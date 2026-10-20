// SPDX-FileCopyrightText: 2026 AurielSolaris
// SPDX-License-Identifier: GPL-3.0-or-later

// IBM Granite 4.0-H dense hybrids (HF "granitemoehybrid" without experts: granite-4.0-h-350m, -h-1b) on CPU.
// Pre-norm decoder whose mixer is Mamba2 in most layers and grouped-query attention without positional encoding
// (NoPE) in a few; every layer ends in a SwiGLU MLP. muP multipliers: embeddings * embedding_multiplier, each
// residual branch * residual_multiplier, attention scores * attention_multiplier, logits / logits_scaling.
//
// Mamba layers carry a recurrent state instead of a per-position cache, so the model can only rewind to the
// restore points it saved (see Model::resumeFrom). Attention layers keep an ordinary f16 KV cache.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "model.h"

namespace neko {

struct GraniteConfig {
    int nVocab = 100352;
    int nMaxPos = 32768;
    int nEmbd = 768;
    int nLayer = 32;
    std::vector<bool> attention;  // per layer: attention (true) or Mamba (false)
    int nHead = 12;
    int nKv = 4;
    int headDim = 64;
    int nFf = 2048;
    float eps = 1e-5f;
    float embeddingMult = 1.0f;
    float residualMult = 1.0f;
    float attentionMult = 0.125f;
    float logitsScaling = 1.0f;
    // Mamba2
    int mHeads = 48;
    int mHeadDim = 32;
    int mState = 128;
    int mGroups = 1;
    int mConv = 4;
    bool mConvBias = true;
};

class GraniteModel : public Model {
public:
    // Only a few layers hold a KV cache (8 KB per token in -h-1b), so the usable context can be generous;
    // CPU pages are committed as positions fill.
    static constexpr int kCpuContext = 16384;
    static constexpr int kRestorePoints = 2;  // redo the last reply / edit the last message

    GraniteModel(const std::string& modelDir, const LoadOptions& opt, const ProgressFn& progress);
    ~GraniteModel() override;

    const char* architecture() const override { return "granitemoehybrid"; }
    void forward(const int* tokens, int T, int pos0, float* logits) override;
    std::vector<DebugStage> debugStages() const override;
    int resumeFrom(int pos) override;
    void saveRestorePoint(int pos) override;

private:
    struct Layer {
        bool attention = false;
        std::vector<float> ln1, ln2;
        QMatrix wgu, wdown;  // shared_mlp input_linear (gate|up rows) and output_linear (* residual_multiplier)
        // attention
        QMatrix wqkv, wo;  // q rows pre-scaled so the kernel's 1/sqrt(hd) becomes attention_multiplier
        std::unique_ptr<uint16_t[]> kCache, vCache;
        // Mamba2
        QMatrix inProj, outProj;  // in_proj rows: gate z | xBC | dt
        std::vector<float> convW, convB, dtBias, A, D, norm;  // convW [convDim][mConv], A = -exp(A_log)
        float* ssm = nullptr;   // [mHeads][mHeadDim][mState] in state_
        float* conv = nullptr;  // [convDim][mConv - 1] last raw xBC inputs, oldest first
    };
    struct RestorePoint {
        int pos = -1;
        std::vector<float> state;
    };

    void loadConfig(const std::string& dir);
    void load(TensorStore& ts, const ProgressFn& progress);
    std::string layerName(int l, const char* leaf) const;
    void mamba(Layer& L, int T);
    void clearState();

    int dInner() const { return cfg_.mHeads * cfg_.mHeadDim; }
    int convDim() const { return dInner() + 2 * cfg_.mGroups * cfg_.mState; }
    int inProjWidth() const { return dInner() + convDim() + cfg_.mHeads; }
    int qWidth() const { return cfg_.nHead * cfg_.headDim; }
    int kvWidth() const { return cfg_.nKv * cfg_.headDim; }
    int qkvWidth() const { return qWidth() + 2 * kvWidth(); }

    GraniteConfig cfg_;
    QMatrix embed_, lmHead_;  // lmHead_ empty when tied
    std::vector<float> normF_;
    std::vector<Layer> layers_;
    std::vector<float> state_;  // every Mamba layer's ssm + conv state
    int statePos_ = 0;          // tokens folded into state_
    RestorePoint restore_[kRestorePoints];
    std::vector<float> x_, xn_, proj_, y_, h_, g_;
};

}  // namespace neko
