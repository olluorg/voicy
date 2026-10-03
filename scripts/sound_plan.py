"""A scene described in plain words → a plan the generator, judge and mixer follow.

    PLANNER_URL=http://127.0.0.1:8091 python scripts/sound_plan.py "описание сцены" plan.json
    PLANNER_URL=... python scripts/sound_plan.py --revise plan.json judge.json plan.next.json [plan.orig.json]
    python scripts/sound_plan.py "описание сцены" plan.json      # transformers, Qwen3-4B

The model decides what sounds, in what order, from where and how loud — in
words. Seconds come from the generated sounds themselves, loudness in LUFS from
the mixer's rules. The plan is checked against the schema, for dangling
references and for words the generator is known to mishear; a broken plan goes
back to the model with the error, twice at most. Through llama-server the
sampling itself is constrained to the schema.
"""
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

import jsonschema

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
# классы AudioSet, которые слышит AST в оценщике: событие обязано попасть в один из ожидаемых
AUDIOSET = json.loads((Path(__file__).with_name("audioset_labels.json")).read_text())
GROUPS = ["engine", "rain", "animal", "music", "speech", "silence"]
ID = {"type": "string", "pattern": "^[a-z][a-z0-9_]*$"}
COMMON = {
    "id": ID,
    "prompt": {"type": "string", "minLength": 20},
    "negative": {"type": "string"},
    "target": {"type": "string", "minLength": 5},
    "perspective": {"enum": ["behind_wall", "room", "near"]},
    "level": {"enum": ["quiet", "medium", "loud"]},
    "allow": {"type": "array", "items": {"enum": GROUPS}},
}
SCHEMA = {
    "type": "object",
    "required": ["title", "duration", "beds", "events"],
    "properties": {
        "title": {"type": "string"},
        "duration": {"type": "number", "minimum": 10, "maximum": 600},
        "beds": {"type": "array", "items": {
            "type": "object",
            "required": ["id", "kind", "prompt", "target", "perspective", "level", "start_after", "start_offset"],
            "properties": {**COMMON,
                           "kind": {"enum": ["wind", "rain", "water", "fire", "room", "crowd", "nature", "machine", "other"]},
                           "start_after": {"type": ["string", "null"]},
                           "start_offset": {"type": "number"},
                           "open": {"type": "array", "items": {
                               "type": "object", "required": ["from_event", "to_event"],
                               "properties": {"from_event": ID, "to_event": ID}}}}}},
        "events": {"type": "array", "minItems": 1, "items": {
            "type": "object",
            "required": ["id", "prompt", "target", "seconds", "after", "gap", "perspective", "level", "expect"],
            "properties": {**COMMON,
                           "seconds": {"type": "number", "minimum": 1, "maximum": 30},
                           "expect": {"type": "array", "minItems": 1, "maxItems": 4, "items": {"enum": AUDIOSET}},
                           "after": {"type": ["string", "null"]},
                           "gap": {"type": "number", "minimum": -10, "maximum": 120}}}},
    },
}

SYSTEM = """You are a sound designer. You plan an ambient sound scene that a text-to-audio model will render piece by piece. The scene description comes in Russian; everything you write is English. Answer with one JSON object and nothing else.

The plan has:
- "title": a short English title.
- "duration": total seconds. Leave 10-20 s after the last event for the scene to breathe and fade.
- "beds": continuous backgrounds (wind, rain, a room's air, a fire burning). Each bed:
  - "kind": wind, rain, water, fire, room, crowd, nature, machine or other;
  - "start_after": null to start with the scene, or the id of an event after which it starts; "start_offset": seconds after that point (may be negative to start before it ends);
  - "open": optional list of {"from_event", "to_event"}: intervals when a bed heard behind a wall is suddenly heard directly, e.g. while a door is open.
- "events": discrete sounds in the order they happen. Each event:
  - "seconds": how long to generate it (a door creak 4-6, a cough 3-4, footsteps across a room 6-10, a fire catching 10-15);
  - "after": null for the first event, else the id of the event it follows; "gap": seconds of pause after that event ends (negative overlaps). The first event's gap is its start time;
  - "expect": 2-4 AudioSet class names that an audio classifier may hear in this sound, exactly as AudioSet names them. List near-synonyms, because the classifier names the same sound differently: a cough ["Cough", "Throat clearing"], a door ["Door", "Creak", "Squeak"], footsteps on wood ["Walk, footsteps", "Creak"], a heavy object set on a table ["Thump, thud", "Thunk", "Knock"]. The checker rejects a take where none of them is heard.
Pacing is set by the mixer: it opens with 20 s of beds, keeps at least 2 s between actions and lets the scene breathe for 45 s after the last one. Give the order and the gaps you mean; use a negative gap only for sounds that truly overlap.
For every bed and event:
- "id": snake_case, unique.
- "prompt": 1-2 sentences for the generator: the physical source, the action, materials, how far it is. Concrete and physical, no emotions, no music. For wind write "broadband roar", "whoosh", "gusts", never "howling" (the generator turns howling into an engine).
- "negative": comma-separated sounds the generator must avoid, e.g. "engine, motor, music" for wind, "rain, hail" for fire.
- "target": 4-8 plain words naming the sound, for an automatic checker, e.g. "an old wooden door creaking open".
- "perspective": "behind_wall" for outdoor sounds when the listener is indoors, "room" for sounds in the listener's room, "near" for sounds right next to the listener.
- "level": "quiet", "medium" or "loud" relative to the scene.
- "allow": optional list from engine, rain, animal, music, speech, silence: things the checker must NOT reject for this sound. A cough needs ["speech"], a dog ["animal"], a quiet room ["silence"], rain ["rain"]. Do not allow anything a sound does not need.

Before answering, think the scene through as a sound designer (silently, do not write it):
1. Every action has physical consequences that are heard. A door that opens is closed again (a separate event); a person who walks in or out closes it right behind them, before doing anything else, unless the description says it is left open. While a door or window to the outside is open, the outdoor bed is heard directly: give that bed an "open" interval from the opening event to the closing event. A fire that is lit keeps burning: after the ignition event add a fire bed that starts near the end of it (start_offset negative, about -3) and lasts to the end.
2. Objects come from the setting and its era, not from a generic modern scene: a Japanese ryokan has sliding paper doors and tatami, not hinged doors and carpet; a medieval tavern has an open hearth and wooden tankards, not a radiator and glasses. Name the material and the kind of object in the prompt. A sound comes from the material that makes it: in a wooden house a door, a floor or a bench creaks by wood rubbing on wood (at the frame, the threshold, between boards), low and dry; squeaks and squeals are metal (hinges, gates, machinery). Do not attribute a wooden creak to iron hinges or metal parts.
3. Small actions that make the described action possible are heard too, when they are typical: lighting a stove means opening its door and striking a match.
4. The listener is inside when the scene happens inside: outdoor weather is "behind_wall" and becomes direct only in an "open" interval.
5. A scene is layered. Indoors there is always a quiet bed of the room itself (kind "room": still air, faint creaks of the building). A fire does not burn steadily the moment it is lit: between the ignition and the burning bed comes a 10-15 s event of the fire catching and growing.
"""

EXAMPLE_IN = "Летний вечер у озера. Плещется вода, квакают лягушки. К берегу подплывает лодка, человек вытаскивает её на песок и уходит по гравийной дорожке."
EXAMPLE_OUT = {
    "title": "Summer evening by the lake",
    "duration": 55,
    "beds": [
        {"id": "lake", "kind": "water", "prompt": "Gentle small waves lapping against a sandy lake shore on a calm summer evening.",
         "negative": "music, engine", "target": "small waves lapping on a lake shore", "perspective": "near",
         "level": "medium", "start_after": None, "start_offset": 0},
        {"id": "frogs", "kind": "nature", "prompt": "A chorus of frogs croaking in reeds some distance away across a quiet lake.",
         "negative": "music, engine", "target": "frogs croaking in the distance", "perspective": "room",
         "level": "quiet", "start_after": None, "start_offset": 0, "allow": ["animal"]},
    ],
    "events": [
        {"id": "oars", "prompt": "Wooden oars dipping and pulling through calm water, a small rowing boat approaching.",
         "negative": "engine, music", "target": "oars rowing a small boat", "seconds": 12, "after": None, "gap": 10,
         "expect": ["Rowboat, canoe, kayak", "Splash, splatter"],
         "perspective": "room", "level": "medium"},
        {"id": "hull_on_sand", "prompt": "The wooden hull of a small boat scraping up onto wet sand.",
         "negative": "music", "target": "a boat hull scraping onto sand", "seconds": 4, "after": "oars", "gap": 0.5,
         "expect": ["Scrape"],
         "perspective": "near", "level": "loud"},
        {"id": "footsteps", "prompt": "Footsteps of a man walking away on a gravel path, crunching steps slowly fading.",
         "negative": "music", "target": "footsteps on gravel walking away", "seconds": 10, "after": "hull_on_sand", "gap": 2,
         "expect": ["Walk, footsteps", "Crunch"],
         "perspective": "near", "level": "medium"},
    ],
}

BANNED = {"howl": "never write 'howl'/'howling': the generator turns it into an engine; write 'roar', 'whoosh', 'gusts'"}


OPENING = re.compile(r"\b(door|window|gate|hatch|shutter)s?\b[^.]*\b(open|opens|opening|opened)\b|"
                     r"\b(open|opens|opening)\b[^.]*\b(door|window|gate|hatch|shutter)s?\b")
CLOSING = re.compile(r"\b(clos|shut|slam)")
INNER = re.compile(r"\b(stove|oven|cupboard|cabinet|drawer|box|chest|wardrobe|fridge)\b")


def opening_problems(plan: dict) -> list[str]:
    """The checklist's first rule, where it can be checked mechanically: an
    opening to the outside is closed again, and while it is open every bed
    behind a wall is heard directly. A 9B model states the rule in its own
    review and then breaks it in the plan, so the code holds it to it."""
    errs = []
    evs = plan["events"]
    walled = [b for b in plan["beds"] if b["perspective"] == "behind_wall"]
    for i, e in enumerate(evs):
        text = e["prompt"].lower()
        if not OPENING.search(text) or INNER.search(text):
            continue                                  # дверца печи на улицу не ведёт
        if CLOSING.search(text):
            errs.append(f"event {e['id']} both opens and closes: split it into an opening and a separate closing event")
            continue
        later = [x for x in evs[i + 1:] if CLOSING.search(x["prompt"].lower()) and not INNER.search(x["prompt"].lower())]
        if not later:
            errs.append(f"event {e['id']} opens to the outside but no later event closes it")
            continue
        between = evs.index(later[0]) - i - 1
        if between > 1:
            errs.append(f"the opening {e['id']} stays open through {between} other events until {later[0]['id']}: "
                        f"whoever walks through closes it right behind them — put {later[0]['id']} right after "
                        f"{e['id']} (or after the footsteps through it), unless the description says it is left open")
        for b in walled:
            if not any(o["from_event"] == e["id"] for o in b.get("open", [])):
                errs.append(f"bed {b['id']} is behind_wall but has no open interval "
                            f"{{\"from_event\": \"{e['id']}\", \"to_event\": \"{later[0]['id']}\"}} while {e['id']} is open")
    lit = [i for i, e in enumerate(evs) if IGNITION.search(e["prompt"].lower())]
    if lit:
        first = evs[lit[0]]["id"]
        later = {e["id"] for e in evs[lit[0]:]}
        for b in plan["beds"]:
            if b["kind"] == "fire" and b["start_after"] not in later:
                errs.append(f"bed {b['id']} burns before the fire is lit: set start_after to \"{evs[lit[-1]]['id']}\" "
                            f"(the last ignition event, after {first}) with start_offset about -3")
    return errs


IGNITION = re.compile(r"\b(ignit|catch(es|ing)? fire|catches|flar(e|es|ing) up|kindling catch|lights? (the|a) fire|bursts? into flame)")


METAL = re.compile(r"\b(iron|metal|steel|hinge|hinges)\b")
NEGATED = re.compile(r"\b(no|without|not|never|free of)\s+(\w+[\s,]+){0,3}?(iron|metal|steel|hinges?)\b[\w\s]*")
WOOD_CREAK = re.compile(r"\bwood(en)?\b[^.]*\bcreak|\bcreak[^.]*\bwood(en)?\b")


def material_problems(plan: dict) -> list[str]:
    """A wooden creak is wood on wood. Written with iron hinges, MOSS renders
    a metal squeak (the izba door, experiments/28)."""
    def affirmed(text):
        """Metal named as a source, not in "no metal squeak" or "without iron hinges"."""
        return METAL.search(NEGATED.sub("", text))

    return [f"{x['id']}: a wooden creak comes from wood rubbing on wood at the frame or between boards; "
            f"remove the iron/metal hinges from the prompt and describe the wood (dry, low, uneven gaps), "
            f"add \"metal squeak, hinge squeal\" to negative"
            for x in plan["beds"] + plan["events"]
            if WOOD_CREAK.search(x["prompt"].lower()) and affirmed(x["prompt"].lower())]


def layer_problems(plan: dict) -> list[str]:
    """Rule 5, mechanically: a room under an indoor scene, and a fire that
    grows before it burns (the hand-made mix the user preferred had both)."""
    errs = []
    indoors = any(x["perspective"] == "behind_wall" for x in plan["beds"] + plan["events"])
    if indoors and not any(b["kind"] == "room" for b in plan["beds"]):
        errs.append("the scene is indoors but has no bed of kind \"room\": add a quiet bed of the room itself "
                    "(still air, faint creaks of the building), level quiet, from the start")
    evs = {e["id"]: e for e in plan["events"]}
    for b in plan["beds"]:
        if b["kind"] == "fire" and b["start_after"] in evs and evs[b["start_after"]]["seconds"] < 8:
            errs.append(f"bed {b['id']} starts burning right after {b['start_after']} ({evs[b['start_after']]['seconds']} s): "
                        f"add a 10-15 s event of the fire catching and growing after {b['start_after']} and start "
                        f"{b['id']} after that event with start_offset about -3")
    return errs


def pacing_problems(plan: dict) -> list[str]:
    """Rough timeline from the planned lengths: the place is established
    before anything happens, and the scene does not idle long after the end."""
    errs, ends = [], {}
    for e in plan["events"]:
        start = (ends[e["after"]] if e["after"] in ends else 0.0) + e["gap"]
        ends[e["id"]] = start + e["seconds"]
    first = plan["events"][0]
    if first["after"] is None and first["gap"] < 8:
        errs.append(f"the first event {first['id']} starts at {first['gap']} s: let the beds establish the place "
                    f"for 8-15 s first (gap 8-15)")
    tail = plan["duration"] - max(ends.values(), default=0)
    if tail > 25:
        errs.append(f"the scene idles {tail:.0f} s after the last event: set duration to about "
                    f"{max(ends.values()) + 15:.0f}")
    if tail < 5:
        errs.append(f"the last event ends {tail:.0f} s before the end: set duration to about {max(ends.values()) + 15:.0f}")
    return errs


def problems(plan: dict) -> list[str]:
    errs = [e.message for e in jsonschema.Draft202012Validator(SCHEMA).iter_errors(plan)]
    if errs:
        return errs[:5]
    for x in plan["beds"] + plan["events"]:
        for word, why in BANNED.items():
            if word in x["prompt"].lower():
                errs.append(f"{x['id']}: {why}")
    events = [e["id"] for e in plan["events"]]
    ids = events + [b["id"] for b in plan["beds"]]
    errs += opening_problems(plan)
    errs += layer_problems(plan)
    errs += material_problems(plan)
    for e in plan["events"]:
        bad = [c for c in e.get("expect", []) if "music" in c.lower()]
        if bad:
            errs.append(f"event {e['id']}: expect {bad} are music genres, not sounds; name the sound itself "
                        f"(e.g. \"Burst, pop\" or \"Crackle\", not \"Pop music\")")
    errs += [f"duplicate id {i}" for i in set(ids) if ids.count(i) > 1]
    for e in plan["events"]:
        if e["after"] is not None and e["after"] not in events:
            errs.append(f"event {e['id']}: after={e['after']} is not an event id")
    for b in plan["beds"]:
        if b["start_after"] is not None and b["start_after"] not in events:
            errs.append(f"bed {b['id']}: start_after={b['start_after']} is not an event id")
        for o in b.get("open", []):
            for k in ("from_event", "to_event"):
                if o[k] not in events:
                    errs.append(f"bed {b['id']}: open.{k}={o[k]} is not an event id")
    return errs


REVIEW = """Review this plan as a strict sound-design supervisor. Go through the four rules of the checklist one by one and, for each, name every violation you find in the plan: a door or window that opens without a separate closing event; an outdoor bed without an "open" interval while the outside is open; an event prompt that describes more than one sound or mentions other elements of the scene; a fire without a burning bed after ignition; objects that do not belong to the setting; a typical small action that is missing. Also check that every prompt describes only its own sound. Be concrete. Do not write the corrected plan yet."""
FIX = "Now send the corrected plan as JSON only, fixing every violation you named."


REVISION = {
    "type": "object", "required": ["revisions"],
    "properties": {"revisions": {"type": "array", "minItems": 1, "items": {
        "type": "object", "required": ["id", "prompt", "negative", "target", "expect"],
        "properties": {"id": ID, "prompt": COMMON["prompt"], "negative": COMMON["negative"],
                       "target": COMMON["target"],
                       "expect": {"type": "array", "minItems": 1, "maxItems": 4, "items": {"enum": AUDIOSET}}}}}},
}
def ask_server(msgs: list, url: str, schema: bool = True, shape: dict | None = None) -> str:
    """llama-server: the reply is constrained to the schema while it is sampled."""
    body = {"messages": msgs, "temperature": 0.25, "top_p": 0.9, "max_tokens": 4000,
            "chat_template_kwargs": {"enable_thinking": False}}
    if schema:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "plan", "schema": shape or SCHEMA}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=900) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def local_asker():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")

    def ask(msgs):
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
        out = model.generate(ids, max_new_tokens=4000, do_sample=True, temperature=0.4, top_p=0.9)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    return ask


def plan(text: str, url: str | None = None) -> dict:
    ask = (lambda m: ask_server(m, url)) if url else local_asker()
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": EXAMPLE_IN},
            {"role": "assistant", "content": json.dumps(EXAMPLE_OUT, ensure_ascii=False)},
            {"role": "user", "content": text}]
    for attempt in range(3):
        reply = ask(msgs).strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            p = json.loads(reply)
            errs = problems(p)
        except json.JSONDecodeError as e:
            errs = [f"not valid JSON: {e}"]
        if not errs and url and attempt == 0 and os.environ.get("PLANNER_REVIEW", "1") == "1":
            # второй проход: модель сама сверяет план с чек-листом и исправляет
            msgs += [{"role": "assistant", "content": reply}, {"role": "user", "content": REVIEW}]
            critique = ask_server(msgs, url, schema=False)
            print("самопроверка:\n" + critique.strip(), file=sys.stderr)
            msgs += [{"role": "assistant", "content": critique}, {"role": "user", "content": FIX}]
            reply = ask(msgs).strip()
            try:
                p = json.loads(reply)
                errs = problems(p)
            except json.JSONDecodeError as e:
                errs = [f"not valid JSON: {e}"]
        if not errs:
            return p
        print(f"попытка {attempt + 1}: {errs}", file=sys.stderr)
        msgs += [{"role": "assistant", "content": reply},
                 {"role": "user", "content": "The plan is invalid: " + "; ".join(errs) + ". Send the corrected JSON only."}]
    raise SystemExit("план так и не прошёл проверку")


def revise(plan_: dict, verdict: dict, url: str, orig: dict | None = None) -> tuple[dict, list[str]]:
    """Elements whose every take was rejected, or that break a rule, go back to
    the model one at a time, with what the judge heard and what is wrong; only
    their prompt, negative and target may change. A revision is kept when its
    own element passes; one stubborn element no longer blocks the rest (a 9B
    model asked for five at once kept leaving one broken, and nothing changed).

    "expect" stays as the first plan had it: allowed to change it, the model
    widened a cough's to Snort and Gasp, and a snort passed as the cough."""
    frozen = {x["id"]: [c for c in (x.get("expect") or []) if "music" not in c.lower()]
              for x in (orig or plan_)["events"]}
    issues = problems(plan_)
    todo = [x["id"] for x in plan_["beds"] + plan_["events"]
            if verdict.get(x["id"], {}).get("all_bad") or any(i.startswith(x["id"] + ":") or f" {x['id']}:" in i
                                                               or f" {x['id']} " in i for i in issues)]
    new = json.loads(json.dumps(plan_))
    changed = []
    for el in todo:
        x = next(y for y in new["beds"] + new["events"] if y["id"] == el)
        takes = verdict.get(el, {}).get("takes", [])
        heard = "\n".join(f"  - {t['heard']} (rejected: {'; '.join(t['why']) or 'no'})" for t in takes)
        own = [i for i in issues if el in i]
        task = (f"Element {el}: prompt \"{x['prompt']}\"; negative \"{x.get('negative', '')}\"; "
                f"target \"{x['target']}\"; expect {x.get('expect', [])}.\n"
                + (f"Every take was rejected; the classifier heard:\n{heard}\n" if takes and verdict[el]["all_bad"] else "")
                + (f"Problems to fix: {'; '.join(own)}\n" if own else "")
                + "Revise only this element: rewrite \"prompt\" so the generator renders the intended sound and "
                  "nothing else (physical process, material, duration, texture); put the sounds it must not be "
                  "into \"negative\"; keep \"target\" a plain name of the sound; send \"expect\" unchanged.")
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Plan:\n" + json.dumps(plan_, ensure_ascii=False)},
                {"role": "user", "content": task}]
        for attempt in range(3):
            reply = ask_server(msgs, url, shape=REVISION)
            try:
                r = next(r for r in json.loads(reply)["revisions"] if r["id"] == el)
            except (StopIteration, json.JSONDecodeError, KeyError):
                errs = [f"send one revision with id {el}"]
            else:
                trial = json.loads(json.dumps(new))
                y = next(z for z in trial["beds"] + trial["events"] if z["id"] == el)
                y.update({k: r[k] for k in ("prompt", "negative", "target")})
                if frozen.get(el):
                    y["expect"] = frozen[el]
                errs = [i for i in problems(trial) if el in i]
                if r["prompt"] == x["prompt"]:
                    errs.append("the prompt did not change")
                if not errs:
                    new = trial
                    changed.append(el)
                    print(f"исправлен {el}: {y['prompt']}", file=sys.stderr)
                    break
            print(f"{el}, попытка {attempt + 1}: {errs}", file=sys.stderr)
            msgs += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": "Invalid: " + "; ".join(errs) + ". Send the corrected JSON only."}]
    return new, changed


def show(p: dict) -> str:
    lines = [f"{p['title']}, {p['duration']} с"]
    for b in p["beds"]:
        lines.append(f"  фон {b['id']} [{b['kind']}, {b['perspective']}, {b['level']}] "
                     f"с {b['start_after'] or 'начала'}{b['start_offset']:+g}"
                     f"{' открыт: ' + str(b['open']) if b.get('open') else ''} — {b['prompt']}")
    for e in p["events"]:
        lines.append(f"  событие {e['id']} [{e['seconds']} с, {e['perspective']}, {e['level']}] "
                     f"после {e['after'] or 'начала'} +{e['gap']} — {e['prompt']}")
    return "\n".join(lines)


if __name__ == "__main__":
    if sys.argv[1] == "--check":
        issues = problems(json.loads(Path(sys.argv[2]).read_text()))
        print("\n".join(issues))
        sys.exit(4 if issues else 0)
    if sys.argv[1] == "--revise":
        old = json.loads(Path(sys.argv[2]).read_text())
        orig = json.loads(Path(sys.argv[5]).read_text()) if len(sys.argv) > 5 else None
        new, changed = revise(old, json.loads(Path(sys.argv[3]).read_text()), os.environ["PLANNER_URL"], orig)
        Path(sys.argv[4]).write_text(json.dumps(new, ensure_ascii=False, indent=1))
        for x in new["beds"] + new["events"]:
            if x["id"] in changed:
                print(f"  {x['id']}: {x['prompt']} | ждём {x.get('expect')}")
        sys.exit(0 if changed else 3)
    p = plan(sys.argv[1], os.environ.get("PLANNER_URL"))
    open(sys.argv[2], "w").write(json.dumps(p, ensure_ascii=False, indent=1))
    print(show(p))
