// Kokoro-82M (hexgrad/Kokoro-82M): StyleTTS 2 with an iSTFTNet vocoder, 24 kHz mono, CPU only.
//
// phoneme ids -> ALBERT (12 passes of one shared layer) -> duration/prosody predictor (BiLSTMs, AdaIN
// residual blocks) -> per-frame alignment -> text encoder -> decoder -> iSTFTNet generator (harmonic source,
// transposed-conv upsampling, snake resblocks, inverse STFT). Mirrors kokoro/model.py, modules.py and
// istftnet.py. Linear and conv weights use the chat models' formats (F16/FP8/FP4, quant.h).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "model.h"

namespace neko {

class TensorStore;

class KokoroModel {
public:
    static constexpr int kSampleRate = 24000;
    static constexpr int kMaxPhonemes = 510;  // plus the two boundary tokens = ALBERT's 512 positions
    static constexpr int kStyleDim = 256;     // voice pack row: decoder style | predictor style

    KokoroModel(const std::string& dir, const LoadOptions& opt, const ProgressFn& progress);
    ~KokoroModel();

    // config.json "vocab": phoneme (one UTF-8 character) -> token id; -1 when the model has no such phoneme.
    int phonemeId(const std::string& utf8Char) const;

    struct Debug {
        std::vector<int> durations;  // frames per token, boundary tokens included
        std::vector<float> f0, n;    // prosody curves (2 x frames)
        // Generator stages, time-major: input [2F][512], harmonic source [600F], its STFT [120F+1][22],
        // conv_post output [120F+1][22].
        std::vector<float> genIn, harSource, har, post;
        // Milliseconds per stage: ALBERT, prosody (durations + F0/N), text encoder + decoder, generator.
        double msAlbert = 0, msProsody = 0, msDecoder = 0, msGenerator = 0;
        // Validation input: use this F0 curve instead of the predicted one (the harmonic source integrates F0
        // into phase, so a 0.1 Hz difference drifts the harmonics; comparing the vocoder needs the same F0).
        const std::vector<float>* forceF0 = nullptr;
        // Validation input: use these durations (boundary tokens included) instead of the predicted ones, so
        // precisions can be compared frame-aligned.
        const std::vector<int>* forceDurations = nullptr;
    };
    // ids: phoneme ids without the boundary tokens (at most kMaxPhonemes). style: kStyleDim floats (the voice
    // pack row for this many phonemes). speed > 1 speaks faster. noise = false disables the source's random
    // noise (validation). Returns 24 kHz samples.
    std::vector<float> synthesize(const std::vector<int>& ids, const float* style, float speed, uint64_t seed,
                                  bool noise = true, Debug* debug = nullptr);

    size_t weightBytes() const { return weightBytes_; }
    std::string weightsLabel() const;
    int threads() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::unordered_map<std::string, int> vocab_;
    size_t weightBytes_ = 0;
    WeightFormat weights_;
};

// A voice pack: [510][1][256] style vectors, one per phoneme count.
class KokoroVoice {
public:
    explicit KokoroVoice(const std::string& path);
    const float* style(int phonemes) const;  // clamped to the pack
    int rows() const { return rows_; }

private:
    std::vector<float> data_;
    int rows_ = 0;
};

}  // namespace neko
