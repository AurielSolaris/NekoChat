// SPDX-FileCopyrightText: 2026 AurielSolaris
// SPDX-License-Identifier: GPL-3.0-or-later

// IBM Granite Speech 5.0 TurboCTC (HF "granite_speech5_ctc": granite-speech-5.0-470m-turboctc and its -nc
// sibling): English speech to text on the CPU.
//
// 16 kHz audio -> 80 HTK mel bins (512-point FFT, 400-sample Hann window, 10 ms hop, centred), log10 floored 8 dB
// below the utterance's peak, plus deltas, two frames stacked (320 values at 50 Hz) -> input_linear -> 16
// conformer blocks (0.5 FF, block self-attention over 128 frames with Shaw relative positions, GLU + depthwise
// conv + BatchNorm, 0.5 FF, LayerNorm); blocks 0 and 1 halve the frame rate (stride-2 conv, pooled residual), so
// the last 14 run at 12.5 Hz. After block 8 the CTC posteriors are fed back (self-conditioning). Greedy CTC over
// 16,384 BPE pieces. Mirrors transformers' modeling/feature_extraction_granite_speech5.py. Weights use the chat
// models' formats (F16/FP8/FP4, quant.h).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "model.h"

namespace neko {

// The CTC vocabulary: SentencePiece-style pieces (-nc) or GPT-2 byte-level BPE (Apache), from tokenizer.json.
class GraniteSpeechTokenizer {
public:
    void load(const std::string& dir, int vocab);
    // CTC-collapsed piece ids (blank and repeats removed) -> text.
    std::string decode(const std::vector<int>& ids) const;

private:
    std::vector<std::string> pieces;  // id -> piece (raw bytes for byte-level vocabularies)
    std::vector<bool> control;        // id -> added token without text (<unk>, <|blank|>, <|tokN|>)
    bool byteLevel = false;
};

class GraniteSpeechModel {
public:
    static constexpr int kSampleRate = 16000;

    GraniteSpeechModel(const std::string& dir, const LoadOptions& opt, const ProgressFn& progress);
    ~GraniteSpeechModel();

    struct Debug {
        std::vector<float> features;  // [frames][320] encoder input
        // With debugLayers = n (1..16), transcribe() stops after block n and leaves its output here
        // ([frames'][1024], before the self-conditioning that follows block 8).
        std::vector<float> hidden;
        int debugLayers = -1;
        std::vector<int> frameIds;    // argmax per output frame, before the CTC collapse
        double msFrontEnd = 0, msEncoder = 0;
        double msMatmul = 0, msAttention = 0, msDepthwise = 0;  // parts of msEncoder
    };

    // 16 kHz mono samples in [-1, 1] -> text. Empty for silence or clips under 0.1 s.
    std::string transcribe(const float* pcm, size_t n, Debug* debug = nullptr);
    size_t weightBytes() const;
    std::string weightsLabel() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace neko
