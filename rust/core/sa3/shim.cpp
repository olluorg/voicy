// A C face for sa3.cpp (Stable Audio 3 on GGML), loaded by voicy-core at run
// time (src/native/sa3.rs). libsa3 has a stable C ABI of its own; this shim
// keeps the Rust side as small as acestep's: five functions, no size-tagged
// structs to mirror. sa3.cpp's ggml fork is linked in statically and hidden,
// as acestep's is: llama.cpp's ggml lives in the same process.

#include "libsa3_v1.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#    define VSA_API extern "C" __declspec(dllexport)
#else
#    define VSA_API extern "C" __attribute__((visibility("default")))
#endif

struct vsa_ctx {
    const sa3_api_v1 * api;
    sa3_context *      ctx;
};

// Called with the stage (0 loading, 1 encoding, 2 sampling, 3 decoding, 4 done)
// and its step of total. Nonzero return cancels.
typedef int (*vsa_progress)(void * user, int stage, int step, int total);

struct cb_state {
    vsa_progress cb;
    void *       user;
    int          cancel;
};

static void SA3_CALL on_progress(void * user, const sa3_progress_v1 * p) {
    auto * s = (cb_state *) user;
    if (s->cb && p && s->cb(s->user, p->stage, p->step, p->total) != 0) {
        s->cancel = 1;
    }
}

static int32_t SA3_CALL should_cancel(void * user) {
    return ((cb_state *) user)->cancel;
}

static void set_err(char * err, int len, const char * msg) {
    if (err && len > 0) {
        snprintf(err, (size_t) len, "%s", msg);
    }
}

// models_dir holds sa3.cpp's GGUF; variant is "medium" or "small-sfx".
VSA_API vsa_ctx * vsa_open(const char * models_dir, const char * variant, char * err, int err_len) {
    const sa3_api_v1 * api = sa3_get_api(SA3_ABI_VERSION_1);
    if (!api) {
        set_err(err, err_len, "libsa3 has no ABI v1");
        return nullptr;
    }
    sa3_error_v1 e;
    memset(&e, 0, sizeof e);
    e.size = sizeof e;
    api->error_init(&e);
    sa3_context_config_v1 cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.size = sizeof cfg;
    api->context_config_init(&cfg);
    cfg.models_dir   = models_dir;
    cfg.variant      = variant;
    cfg.dit_encoding = "f16";
    sa3_context * ctx = nullptr;
    if (api->context_create(&cfg, &ctx, &e) != SA3_STATUS_OK_V1) {
        set_err(err, err_len, e.message);
        return nullptr;
    }
    auto * c = new vsa_ctx{ api, ctx };
    return c;
}

// One piece of music. resident keeps the models in VRAM between calls;
// otherwise they are loaded for each (FRUGAL). On success *samples is planar
// [ch0..., ch1...], *n_samples per channel; free with vsa_free. Returns 0,
// -1 on error, 1 if cancelled.
VSA_API int vsa_generate(vsa_ctx * c, const char * prompt, const char * negative, double seconds, long long seed,
                         int steps, float cfg_scale, int resident, vsa_progress cb, void * user, float ** samples,
                         unsigned long long * n_samples, int * channels, int * sample_rate, char * err, int err_len) {
    const sa3_api_v1 * api = c->api;
    sa3_error_v1       e;
    memset(&e, 0, sizeof e);
    e.size = sizeof e;
    api->error_init(&e);
    sa3_request_v1 r;
    memset(&r, 0, sizeof r);
    r.size = sizeof r;
    api->request_init(&r);
    r.operation        = SA3_OPERATION_GENERATE_V1;
    r.prompt           = prompt;
    r.negative_prompt  = negative && *negative ? negative : nullptr;
    r.duration_seconds = seconds;
    if (seed >= 0) {
        r.seed = seed;
    }
    if (steps > 0) {
        r.steps = steps;
    }
    if (cfg_scale > 0) {
        r.cfg_scale = cfg_scale;
    }
    r.residency = resident ? SA3_RESIDENCY_RESIDENT_V1 : SA3_RESIDENCY_FRUGAL_V1;
    cb_state st{ cb, user, 0 };
    r.on_progress   = on_progress;
    r.should_cancel = should_cancel;
    r.callback_user = &st;

    sa3_result_v1 res;
    memset(&res, 0, sizeof res);
    res.size = sizeof res;
    api->result_init(&res);
    sa3_status_v1 rc = api->generate(c->ctx, &r, &res, &e);
    if (rc == SA3_STATUS_CANCELLED_V1 || st.cancel) {
        api->result_free(&res);
        return 1;
    }
    if (rc != SA3_STATUS_OK_V1) {
        set_err(err, err_len, e.message);
        api->result_free(&res);
        return -1;
    }
    // копия в malloc: освобождать её будет vsa_free, а не таблица libsa3
    size_t n = (size_t) res.n_samples * res.n_channels;
    float * out = (float *) malloc(n * sizeof(float));
    if (res.layout == SA3_AUDIO_PLANAR_V1) {
        memcpy(out, res.samples, n * sizeof(float));
    } else {
        for (size_t i = 0; i < res.n_samples; i++) {
            for (uint32_t ch = 0; ch < res.n_channels; ch++) {
                out[ch * res.n_samples + i] = res.samples[i * res.n_channels + ch];
            }
        }
    }
    *samples     = out;
    *n_samples   = res.n_samples;
    *channels    = (int) res.n_channels;
    *sample_rate = (int) res.sample_rate;
    api->result_free(&res);
    return 0;
}

VSA_API void vsa_free(void * p) {
    free(p);
}

// Drop the weights from VRAM; the context stays usable.
VSA_API void vsa_unload(vsa_ctx * c) {
    c->api->context_unload(c->ctx);
}

VSA_API void vsa_close(vsa_ctx * c) {
    if (!c) {
        return;
    }
    c->api->context_destroy(c->ctx);
    delete c;
}
