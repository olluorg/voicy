// A C face for CTranslate2's Whisper, so Rust can call it without C++.
//
// CTranslate2 has only a C++ API. This file wraps the four calls faster-whisper
// makes — encode, generate, detect_language, align — and nothing else, against
// the prebuilt libctranslate2 that faster-whisper itself ships (4.8.2). Same
// library, same weights: the recogniser is faster-whisper's, only the Python
// around it is gone (rust/src/native/fwhisper.rs).
//
// Built by `voicy setup`, which fetches CTranslate2's headers at the same tag
// (src/setup.rs). Errors come back as text in `err`; every function that
// allocates has its matching free.

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "ctranslate2/models/whisper.h"

using namespace ctranslate2;

// Windows не выносит ничего наружу без явной пометки, unix выносит всё.
// Внутри бинарника (CT2SHIM_STATIC) выносить нечего: вызовы прямые.
#if defined(_WIN32) && !defined(CT2SHIM_STATIC)
#define CT2W __declspec(dllexport)
#else
#define CT2W
#endif

namespace {
void set_err(char* err, size_t len, const std::string& msg) {
  if (err && len) {
    std::strncpy(err, msg.c_str(), len - 1);
    err[len - 1] = 0;
  }
}
}  // namespace

extern "C" {

struct ct2w_gen_opts {
  size_t beam_size;
  float patience;
  float length_penalty;
  float repetition_penalty;
  size_t no_repeat_ngram_size;
  size_t max_length;
  size_t sampling_topk;
  float sampling_temperature;
  size_t num_hypotheses;
  int suppress_blank;
  const int* suppress_tokens;
  size_t n_suppress_tokens;
  size_t max_initial_timestamp_index;
};

CT2W void* ct2w_load(const char* path, int cuda, int device_index, const char* compute_type,
                size_t intra_threads, size_t inter_threads, char* err, size_t errlen) {
  try {
    ReplicaPoolConfig cfg;
    cfg.num_threads_per_replica = intra_threads;
    cfg.max_queued_batches = 0;
    std::vector<int> devices(inter_threads > 0 ? inter_threads : 1, device_index);
    auto* m = new models::Whisper(path, cuda ? Device::CUDA : Device::CPU,
                                  str_to_compute_type(compute_type), devices, false, cfg);
    return m;
  } catch (const std::exception& e) {
    set_err(err, errlen, e.what());
    return nullptr;
  }
}

CT2W void ct2w_free(void* m) { delete static_cast<models::Whisper*>(m); }

CT2W int ct2w_is_multilingual(void* m) { return static_cast<models::Whisper*>(m)->is_multilingual(); }
CT2W size_t ct2w_n_mels(void* m) { return static_cast<models::Whisper*>(m)->n_mels(); }

// features: [n_mels][n_frames] float32, row-major, one item.
CT2W void* ct2w_encode(void* m, const float* features, size_t n_mels, size_t n_frames, char* err, size_t errlen) {
  try {
    std::vector<float> data(features, features + n_mels * n_frames);
    StorageView sv({1, (dim_t)n_mels, (dim_t)n_frames}, data);
    auto out = static_cast<models::Whisper*>(m)->encode(sv, false).get();
    return new StorageView(std::move(out));
  } catch (const std::exception& e) {
    set_err(err, errlen, e.what());
    return nullptr;
  }
}

CT2W void ct2w_free_sv(void* sv) { delete static_cast<StorageView*>(sv); }

// The best hypothesis: its ids (malloc'd), score and the no-speech probability.
CT2W int ct2w_generate(void* m, void* encoded, const size_t* prompt, size_t n_prompt, const ct2w_gen_opts* o,
                  size_t** ids, size_t* n_ids, float* score, float* no_speech_prob, char* err, size_t errlen) {
  try {
    models::WhisperOptions opts;
    opts.beam_size = o->beam_size;
    opts.patience = o->patience;
    opts.length_penalty = o->length_penalty;
    opts.repetition_penalty = o->repetition_penalty;
    opts.no_repeat_ngram_size = o->no_repeat_ngram_size;
    opts.max_length = o->max_length;
    opts.sampling_topk = o->sampling_topk;
    opts.sampling_temperature = o->sampling_temperature;
    opts.num_hypotheses = o->num_hypotheses;
    opts.return_scores = true;
    opts.return_no_speech_prob = true;
    opts.max_initial_timestamp_index = o->max_initial_timestamp_index;
    opts.suppress_blank = o->suppress_blank != 0;
    opts.suppress_tokens.assign(o->suppress_tokens, o->suppress_tokens + o->n_suppress_tokens);
    std::vector<std::vector<size_t>> prompts{std::vector<size_t>(prompt, prompt + n_prompt)};
    auto futures = static_cast<models::Whisper*>(m)->generate(*static_cast<StorageView*>(encoded), prompts, opts);
    auto r = futures[0].get();
    const auto& seq = r.sequences_ids.at(0);
    *n_ids = seq.size();
    *ids = static_cast<size_t*>(std::malloc(sizeof(size_t) * (seq.size() + 1)));
    std::memcpy(*ids, seq.data(), sizeof(size_t) * seq.size());
    *score = r.scores.empty() ? 0.f : r.scores[0];
    *no_speech_prob = r.no_speech_prob;
    return 0;
  } catch (const std::exception& e) {
    set_err(err, errlen, e.what());
    return -1;
  }
}

CT2W void ct2w_free_buf(void* p) { std::free(p); }

// Languages as "<|ru|>\n<|en|>\n…" (malloc'd) and their probabilities, most likely first.
CT2W int ct2w_detect_language(void* m, void* encoded, char** tokens, float** probs, size_t* n, char* err, size_t errlen) {
  try {
    auto r = static_cast<models::Whisper*>(m)->detect_language(*static_cast<StorageView*>(encoded))[0].get();
    std::string joined;
    *n = r.size();
    *probs = static_cast<float*>(std::malloc(sizeof(float) * (r.size() + 1)));
    for (size_t i = 0; i < r.size(); ++i) {
      joined += r[i].first + "\n";
      (*probs)[i] = r[i].second;
    }
    *tokens = static_cast<char*>(std::malloc(joined.size() + 1));
    std::memcpy(*tokens, joined.c_str(), joined.size() + 1);
    return 0;
  } catch (const std::exception& e) {
    set_err(err, errlen, e.what());
    return -1;
  }
}

// Alignment pairs (text index, time index) flattened, and the text token probabilities.
CT2W int ct2w_align(void* m, void* encoded, const size_t* start, size_t n_start, const size_t* text, size_t n_text,
               size_t num_frames, long median_filter_width, long long** pairs, size_t* n_pairs,
               float** probs, size_t* n_probs, char* err, size_t errlen) {
  try {
    std::vector<size_t> start_seq(start, start + n_start);
    std::vector<std::vector<size_t>> text_tokens{std::vector<size_t>(text, text + n_text)};
    auto r = static_cast<models::Whisper*>(m)
                 ->align(*static_cast<StorageView*>(encoded), start_seq, text_tokens, {num_frames},
                         median_filter_width)[0]
                 .get();
    *n_pairs = r.alignments.size();
    *pairs = static_cast<long long*>(std::malloc(sizeof(long long) * (2 * r.alignments.size() + 1)));
    for (size_t i = 0; i < r.alignments.size(); ++i) {
      (*pairs)[2 * i] = r.alignments[i].first;
      (*pairs)[2 * i + 1] = r.alignments[i].second;
    }
    *n_probs = r.text_token_probs.size();
    *probs = static_cast<float*>(std::malloc(sizeof(float) * (r.text_token_probs.size() + 1)));
    std::memcpy(*probs, r.text_token_probs.data(), sizeof(float) * r.text_token_probs.size());
    return 0;
  } catch (const std::exception& e) {
    set_err(err, errlen, e.what());
    return -1;
  }
}

}  // extern "C"
