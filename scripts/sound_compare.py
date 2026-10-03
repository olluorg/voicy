"""One table over scene runs of the same plan with different generators.

    python3 scripts/sound_compare.py NAME=RUN_DIR ...

Per run: how many takes the judge rejected, which elements had no acceptable
take (the least bad went in), and where the scene is. The ear decides; this
says where to listen hardest.
"""
import json
import sys
from pathlib import Path

print("| модель | забраковано | без годных | сцена |")
print("|---|---|---|---|")
for arg in sys.argv[1:]:
    name, run = arg.split("=", 1)
    run = Path(run)
    verdict = json.loads((run / "judge.json").read_text())
    takes = sum(len(v["takes"]) for v in verdict.values())
    bad = sum(1 for v in verdict.values() for t in v["takes"] if t["why"])
    failed = [k for k, v in verdict.items() if v["all_bad"]]
    print(f"| {name} | {bad} из {takes} ({100 * bad / takes:.0f}%) | {', '.join(failed) or '—'} | `{run / 'scene.wav'}` |")
