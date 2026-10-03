// A C face for acestep.cpp (ACE-Step 1.5 on GGML), loaded by voicy-core at
// run time like llama.cpp (src/native/acestep.rs).
//
// acestep.cpp carries its own ggml fork: the VAE needs two ops upstream ggml
// does not have (GGML_OP_SNAKE, GGML_OP_COL2IM_1D). So the fork is linked into
// this library statically and nothing of it is exported: llama.cpp's ggml
// lives in the same process, and two ggml's must not see each other's symbols.
//
// One request runs the whole chain: the LM plans the song (metadata, codes),
// the DiT renders latents, the VAE turns them into 48 kHz stereo.

#include "model-registry.h"
#include "model-store.h"
#include "pipeline-lm.h"
#include "pipeline-synth.h"
#include "pipeline-synth-impl.h"
#include "ggml-backend.h"
#include "request.h"
#include "task-types.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(_WIN32)
#    define VAC_API extern "C" __declspec(dllexport)
#else
#    define VAC_API extern "C" __attribute__((visibility("default")))
#endif

// Stages, as the progress callback reports them.
enum { VAC_LM = 0, VAC_DIT = 1, VAC_VAE = 2 };

// Called between LM tokens, DiT steps and VAE tiles with the stage and how
// many times it was called within that stage. Nonzero return cancels.
typedef int (*vac_progress)(void * user, int stage, int count);

struct vac_ctx {
    ModelStore * store;
    AceLm *      lm;
    AceSynth *   synth;
    std::string  lm_path, text_enc_path, dit_path, vae_path;
    bool         keep_loaded;
    int          vae_chunk;  // 0 — по свободной видеопамяти перед каждым декодированием
};

// The VAE's activations are the peak of the whole chain: on an RTX 3080 a
// chunk of 1024 latent frames took 8.4 GB, 512 — 4.6 GB, 256 — 3.0 GB, about
// 7.4 MB a frame. Beside speech in the same process 1024 ran out and ggml
// aborted the process. So the chunk is the largest that fits what is free —
// counting what the store will drop before the VAE loads, unless it keeps
// everything — with a margin for the desktop and the drivers.
static int pick_vae_chunk(const vac_ctx * c) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        return 1024;  // процессор: память — это ОЗУ
    }
    size_t free = 0, total = 0;
    ggml_backend_dev_memory(dev, &free, &total);
    double avail = (double) free + (c->keep_loaded ? 0.0 : (double) store_vram_bytes(c->store));
    const double per_frame = 7.5e6, fixed = 0.6e9, margin = 0.8e9;
    for (int chunk : { 1024, 512, 256 }) {
        if (fixed + per_frame * chunk + margin <= avail) {
            return chunk;
        }
    }
    return 128;
}

struct poll_state {
    vac_progress cb;
    void *       user;
    int          stage;
    int          count;
    bool         cancelled;
};

static bool poll(void * p) {
    auto * s = (poll_state *) p;
    if (s->cb && s->cb(s->user, s->stage, s->count++) != 0) {
        s->cancelled = true;
    }
    return s->cancelled;
}

static void set_err(char * err, int len, const char * msg) {
    if (err && len > 0) {
        snprintf(err, (size_t) len, "%s", msg);
    }
}

// Open the four models: paths to GGUF files. keep_loaded keeps every module
// in VRAM between requests; otherwise one module is resident at a time and
// each request reloads them. vae_chunk: latent frames the VAE decodes at once;
// 0 picks it before each decode by free VRAM (pick_vae_chunk).
VAC_API vac_ctx * vac_open(const char * lm, const char * text_enc, const char * dit, const char * vae,
                           int keep_loaded, int max_seq, int vae_chunk, char * err, int err_len) {
    auto * c          = new vac_ctx();
    c->keep_loaded    = keep_loaded != 0;
    c->vae_chunk      = vae_chunk;
    c->lm_path        = lm;
    c->text_enc_path  = text_enc;
    c->dit_path       = dit;
    c->vae_path       = vae;
    c->store          = store_create(keep_loaded ? EVICT_NEVER : EVICT_STRICT);

    AceLmParams lp;
    ace_lm_default_params(&lp);
    lp.model_path = c->lm_path.c_str();
    lp.max_batch  = 1;
    if (max_seq > 0) {
        lp.max_seq = max_seq;
    }
    c->lm = ace_lm_load(c->store, &lp);
    if (!c->lm) {
        set_err(err, err_len, "LM did not load");
        store_free(c->store);
        delete c;
        return nullptr;
    }

    AceSynthParams sp;
    ace_synth_default_params(&sp);
    sp.text_encoder_path = c->text_enc_path.c_str();
    sp.dit_path          = c->dit_path.c_str();
    sp.vae_path          = c->vae_path.c_str();
    c->synth             = ace_synth_load(c->store, &sp);
    if (!c->synth) {
        set_err(err, err_len, "DiT, text encoder or VAE did not load");
        ace_lm_free(c->lm);
        store_free(c->store);
        delete c;
        return nullptr;
    }
    return c;
}

// Generate one track. request_json is acestep.cpp's own request format
// (caption, lyrics — "[Instrumental]" for none —, duration, vocal_language,
// seed, lm_seed, ...). On success *samples is planar stereo
// [L0..Ln-1, R0..Rn-1], *n_samples per channel, *sample_rate 48000, and
// *meta the JSON of the request as the LM enriched it (bpm, key, lyrics it
// wrote); free both with vac_free. Returns 0, -1 on error, 1 if cancelled.
VAC_API int vac_generate(vac_ctx * c, const char * request_json, vac_progress cb, void * user, float ** samples,
                         int * n_samples, int * sample_rate, char ** meta, char * err, int err_len) {
    AceRequest req;
    request_init(&req);
    if (!request_parse_json(&req, request_json)) {
        set_err(err, err_len, "malformed request JSON");
        return -1;
    }
    request_resolve_lm_seed(&req);
    request_resolve_seed(&req);

    poll_state ps{ cb, user, VAC_LM, 0, false };
    AceRequest planned;
    if (ace_lm_generate(c->lm, &req, 1, &planned, nullptr, nullptr, poll, &ps, LM_MODE_GENERATE) != 0) {
        if (ps.cancelled) {
            return 1;
        }
        set_err(err, err_len, "LM failed");
        return -1;
    }

    ps.stage            = VAC_DIT;
    ps.count            = 0;
    AceSynthJob * job   = ace_synth_job_run_dit(c->synth, &planned, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, 1,
                                                poll, &ps);
    if (!job) {
        if (ps.cancelled) {
            return 1;
        }
        set_err(err, err_len, "DiT failed");
        return -1;
    }

    ps.stage                  = VAC_VAE;
    ps.count                  = 0;
    c->synth->params.vae_chunk = c->vae_chunk > 0 ? c->vae_chunk : pick_vae_chunk(c);
    fprintf(stderr, "[voicy] VAE chunk %d\n", c->synth->params.vae_chunk);
    AceAudio audio{};
    int      rc = ace_synth_job_run_vae(c->synth, job, &audio, poll, &ps);
    ace_synth_job_free(job);
    if (rc != 0) {
        if (ps.cancelled) {
            return 1;
        }
        set_err(err, err_len, "VAE failed");
        return -1;
    }

    *samples     = audio.samples;  // malloc'ed by acestep: vac_free releases it
    *n_samples   = audio.n_samples;
    *sample_rate = audio.sample_rate;
    std::string m = request_to_json(&planned, true);
    *meta         = (char *) malloc(m.size() + 1);
    memcpy(*meta, m.c_str(), m.size() + 1);
    return 0;
}

VAC_API void vac_free(void * p) {
    free(p);
}

// VRAM the store holds right now, in bytes.
VAC_API size_t vac_vram(const vac_ctx * c) {
    return store_vram_bytes(c->store);
}

VAC_API void vac_close(vac_ctx * c) {
    if (!c) {
        return;
    }
    ace_synth_free(c->synth);
    ace_lm_free(c->lm);
    store_free(c->store);
    delete c;
}
