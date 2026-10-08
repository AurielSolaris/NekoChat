#include "granite_model.h"

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include "common.h"
#include "cpu_ops.h"
#include "json.h"
#include "tensor_store.h"

namespace neko {

namespace {

inline float silu(float v) { return v / (1.0f + std::exp(-v)); }
inline float softplus(float v) { return v > 20.0f ? v : std::log1p(std::exp(v)); }

}  // namespace

void GraniteModel::loadConfig(const std::string& dir) {
    Json c = Json::parseFile(dir + "/config.json");
    if (c.getInt("num_local_experts", 0) > 0)
        fail("Granite mixture-of-experts models (granite-4.0-h-tiny/-small) are not supported yet");
    std::string pe = c.getStr("position_embedding_type", "nope");
    if (pe != "nope") fail("Granite with position_embedding_type '" + pe + "' is not supported (4.0-H uses nope)");
    cfg_.nVocab = int(c.getInt("vocab_size", 100352));
    cfg_.nMaxPos = int(c.getInt("max_position_embeddings", 32768));
    cfg_.nEmbd = int(c.getInt("hidden_size", 768));
    cfg_.nLayer = int(c.getInt("num_hidden_layers", 32));
    cfg_.nHead = int(c.getInt("num_attention_heads", 12));
    cfg_.nKv = int(c.getInt("num_key_value_heads", cfg_.nHead));
    cfg_.headDim = cfg_.nEmbd / cfg_.nHead;
    cfg_.nFf = int(c.getInt("shared_intermediate_size", c.getInt("intermediate_size", 2048)));
    cfg_.eps = float(c.getNum("rms_norm_eps", 1e-5));
    cfg_.embeddingMult = float(c.getNum("embedding_multiplier", 1.0));
    cfg_.residualMult = float(c.getNum("residual_multiplier", 1.0));
    cfg_.attentionMult = float(c.getNum("attention_multiplier", 1.0 / std::sqrt(double(cfg_.headDim))));
    cfg_.logitsScaling = float(c.getNum("logits_scaling", 1.0));
    cfg_.mHeads = int(c.getInt("mamba_n_heads", 48));
    cfg_.mHeadDim = int(c.getInt("mamba_d_head", 32));
    cfg_.mState = int(c.getInt("mamba_d_state", 128));
    cfg_.mGroups = int(c.getInt("mamba_n_groups", 1));
    cfg_.mConv = int(c.getInt("mamba_d_conv", 4));
    cfg_.mConvBias = c["mamba_conv_bias"].boolean(true);
    if (c["mamba_proj_bias"].boolean(false)) fail("Granite Mamba projections with bias are not supported");
    for (auto& t : c["layer_types"].items()) cfg_.attention.push_back(t.str() == "attention");
    if (int(cfg_.attention.size()) != cfg_.nLayer) fail("config.json layer_types does not list every layer");
    if (cfg_.nKv <= 0 || cfg_.nHead % cfg_.nKv != 0) fail("num_attention_heads must be a multiple of num_key_value_heads");
    if (cfg_.mGroups <= 0 || cfg_.mHeads % cfg_.mGroups != 0) fail("mamba_n_heads must be a multiple of mamba_n_groups");
    if (cfg_.mState % 4 != 0) fail("mamba_d_state must be a multiple of 4");
    if (cfg_.mConv < 2) fail("mamba_d_conv must be at least 2");
    shape_.vocab = cfg_.nVocab;
    shape_.context = std::min(cfg_.nMaxPos, kCpuContext);
    shape_.layers = cfg_.nLayer;
    shape_.embd = cfg_.nEmbd;
    shape_.heads = cfg_.nHead;
    shape_.kvHeads = cfg_.nKv;
    shape_.headDim = cfg_.headDim;
}

std::string GraniteModel::layerName(int l, const char* leaf) const {
    return "model.layers." + std::to_string(l) + "." + leaf;
}

GraniteModel::GraniteModel(const std::string& dir, const LoadOptions& opt, const ProgressFn& progress) : Model(opt) {
    progress(0.0f, "Reading config");
    loadConfig(dir);
    pool_ = std::make_unique<ThreadPool>(opt.threads);

    TensorStore ts;
    progress(0.02f, "Mapping checkpoint");
    openCheckpoint(dir, ts);
    if (!ts.has("model.embed_tokens.weight")) fail("checkpoint does not look like Granite (no model.embed_tokens.weight)");
    for (auto& kv : ts.all()) params_ += kv.second.numel();
    if (opt.backend != BackendPref::CPU) note_ = "Granite runs on the CPU (no GPU kernels for Mamba layers yet)\n";
    load(ts, progress);
    progress(1.0f, "Ready");
    NEKO_LOGI("Granite loaded: %lld params, %d layers, ctx %d, %s weights (%zu MB)", (long long)params_, cfg_.nLayer,
              shape_.context, weightsLabel().c_str(), weightBytes_ >> 20);
}

GraniteModel::~GraniteModel() = default;

std::vector<Model::DebugStage> GraniteModel::debugStages() const {
    const int C = cfg_.nEmbd;
    return {{"ln1", C}, {"mixer+res", C}, {"ln2", C}, {"gate|up", 2 * cfg_.nFf}, {"silu*up", cfg_.nFf}, {"mlp+res", C}};
}

// ------------------------------------------------------------------ loading

void GraniteModel::load(TensorStore& ts, const ProgressFn& progress) {
    const int C = cfg_.nEmbd, F = cfg_.nFf, CD = convDim(), K = cfg_.mConv;
    const std::string verb = weightFormat() == WeightFormat::F16 ? "Loading" : "Quantizing";
    progress(0.05f, verb + " embeddings");
    embed_ = matrix(ts, {"model.embed_tokens.weight"});
    if (ts.has("lm_head.weight")) lmHead_ = matrix(ts, {"lm_head.weight"});
    normF_ = ts.toF32("model.norm.weight");
    weightBytes_ = embed_.bytes() + lmHead_.bytes() + normF_.size() * 4;

    // The attention kernel scales scores by 1/sqrt(hd); Granite wants attention_multiplier instead.
    const float qScale = cfg_.attentionMult * std::sqrt(float(cfg_.headDim));
    const float res = cfg_.residualMult;
    size_t stateFloats = 0;
    const size_t ssmFloats = size_t(cfg_.mHeads) * size_t(cfg_.mHeadDim) * size_t(cfg_.mState);
    const size_t convFloats = size_t(CD) * size_t(K - 1);
    int nAttention = 0;
    layers_.resize(size_t(cfg_.nLayer));
    for (int l = 0; l < cfg_.nLayer; l++) {
        progress(0.1f + 0.85f * float(l) / float(cfg_.nLayer), verb + " layer " + std::to_string(l + 1));
        Layer& L = layers_[size_t(l)];
        L.attention = cfg_.attention[size_t(l)];
        L.ln1 = ts.toF32(layerName(l, "input_layernorm.weight"));
        L.ln2 = ts.toF32(layerName(l, "post_attention_layernorm.weight"));
        L.wgu = matrix(ts, {layerName(l, "shared_mlp.input_linear.weight")});
        L.wdown = scaledMatrix(ts, layerName(l, "shared_mlp.output_linear.weight"), res);
        if (L.wgu.rows != 2 * F) fail("shared_mlp.input_linear has an unexpected shape");
        size_t bytes = L.wgu.bytes() + L.wdown.bytes() + (L.ln1.size() + L.ln2.size()) * 4;
        if (L.attention) {
            nAttention++;
            L.wqkv = scaledMatrix(ts, layerName(l, "self_attn.q_proj.weight"), qScale);
            L.wqkv.appendRows(matrix(ts, {layerName(l, "self_attn.k_proj.weight"), layerName(l, "self_attn.v_proj.weight")}));
            L.wo = scaledMatrix(ts, layerName(l, "self_attn.o_proj.weight"), res);
            if (L.wqkv.rows != qkvWidth()) fail("attention projections have an unexpected shape");
            bytes += L.wqkv.bytes() + L.wo.bytes();
            // Uninitialised on purpose: pages are only committed once positions are written.
            const size_t cache = size_t(shape_.context) * size_t(kvWidth());
            L.kCache.reset(new uint16_t[cache]);
            L.vCache.reset(new uint16_t[cache]);
        } else {
            L.inProj = matrix(ts, {layerName(l, "mamba.in_proj.weight")});
            L.outProj = scaledMatrix(ts, layerName(l, "mamba.out_proj.weight"), res);
            if (L.inProj.rows != inProjWidth() || L.outProj.cols != dInner())
                fail("Mamba projections have an unexpected shape (check mamba_n_heads / mamba_d_head)");
            L.convW = ts.toF32(layerName(l, "mamba.conv1d.weight"));
            if (L.convW.size() != size_t(CD) * size_t(K)) fail("mamba.conv1d.weight has an unexpected shape");
            L.convB = cfg_.mConvBias ? ts.toF32(layerName(l, "mamba.conv1d.bias")) : std::vector<float>(size_t(CD), 0.0f);
            L.dtBias = ts.toF32(layerName(l, "mamba.dt_bias"));
            L.A = ts.toF32(layerName(l, "mamba.A_log"));
            for (float& a : L.A) a = -std::exp(a);
            L.D = ts.toF32(layerName(l, "mamba.D"));
            L.norm = ts.toF32(layerName(l, "mamba.norm.weight"));
            bytes += L.inProj.bytes() + L.outProj.bytes() +
                     (L.convW.size() + L.convB.size() + L.dtBias.size() + L.A.size() + L.D.size() + L.norm.size()) * 4;
            stateFloats += ssmFloats + convFloats;
        }
        weightBytes_ += bytes;
    }
    kvBytesPerToken_ = size_t(nAttention) * 2 * size_t(kvWidth()) * 2;

    state_.assign(stateFloats, 0.0f);
    float* p = state_.data();
    for (Layer& L : layers_) {
        if (L.attention) continue;
        L.ssm = p;
        L.conv = p + ssmFloats;
        p += ssmFloats + convFloats;
    }
    stateBytes_ = state_.size() * 4;

    const size_t B = kMaxBatch;
    x_.resize(B * size_t(C));
    xn_.resize(B * size_t(C));
    proj_.resize(B * size_t(std::max(inProjWidth(), qkvWidth())));
    y_.resize(B * size_t(std::max(dInner(), qWidth())));
    h_.resize(B * size_t(2 * F));
    g_.resize(B * size_t(F));
    workBytes_ = (x_.size() + xn_.size() + proj_.size() + y_.size() + h_.size() + g_.size()) * 4;
}

// ------------------------------------------------------------------ recurrent state

void GraniteModel::clearState() {
    std::fill(state_.begin(), state_.end(), 0.0f);
    statePos_ = 0;
}

void GraniteModel::saveRestorePoint(int pos) {
    if (pos != statePos_ || state_.empty()) return;
    for (RestorePoint& r : restore_)
        if (r.pos == pos) return;
    // Replace the oldest (lowest) point; positions only grow within a conversation.
    RestorePoint* slot = &restore_[0];
    for (RestorePoint& r : restore_)
        if (r.pos < slot->pos) slot = &r;
    slot->pos = pos;
    slot->state = state_;
    size_t bytes = state_.size() * 4;
    for (RestorePoint& r : restore_) bytes += r.state.size() * 4;
    stateBytes_ = bytes;
}

// Invariant: every restore point is a prefix of the current history. Forward passes only extend the history;
// rewinding to r drops the points after r, which belong to the history being replaced.
int GraniteModel::resumeFrom(int pos) {
    if (pos >= statePos_) return statePos_;
    const RestorePoint* best = nullptr;
    for (const RestorePoint& r : restore_)
        if (r.pos >= 0 && r.pos <= pos && (!best || r.pos > best->pos)) best = &r;
    if (best) {
        state_ = best->state;
        statePos_ = best->pos;
    } else {
        clearState();
    }
    size_t bytes = state_.size() * 4;
    for (RestorePoint& r : restore_) {
        if (r.pos > statePos_) {
            r.pos = -1;
            std::vector<float>().swap(r.state);
        }
        bytes += r.state.size() * 4;
    }
    stateBytes_ = bytes;
    return statePos_;
}

// ------------------------------------------------------------------ forward

// Mamba2 mixer for T tokens: xn_ -> in_proj -> causal depthwise conv + SiLU on xBC -> selective scan per head
// -> y * silu(z) -> gated RMSNorm -> out_proj, accumulated into x_ (out_proj carries residual_multiplier).
void GraniteModel::mamba(Layer& L, int T) {
    const int C = cfg_.nEmbd, DI = dInner(), CD = convDim(), P = inProjWidth(), K = cfg_.mConv;
    const int H = cfg_.mHeads, HD = cfg_.mHeadDim, N = cfg_.mState, headsPerGroup = cfg_.mHeads / cfg_.mGroups;
    ThreadPool& pool = *pool_;
    cpu::matmul(pool, xn_.data(), T, C, L.inProj, nullptr, proj_.data(), P, 0);

    // Conv over time, channel by channel; the window keeps the last K-1 raw inputs across calls.
    pool.parallelFor(CD, [&](int c0, int c1, int) {
        for (int c = c0; c < c1; c++) {
            const float* w = L.convW.data() + size_t(c) * size_t(K);
            float* win = L.conv + size_t(c) * size_t(K - 1);
            for (int t = 0; t < T; t++) {
                float* v = proj_.data() + size_t(t) * size_t(P) + size_t(DI) + size_t(c);
                const float in = *v;
                float s = L.convB[size_t(c)] + w[K - 1] * in;
                for (int k = 0; k < K - 1; k++) s += w[k] * win[k];
                for (int k = 0; k + 1 < K - 1; k++) win[k] = win[k + 1];
                win[K - 2] = in;
                *v = silu(s);
            }
        }
    });

    // Selective scan: per head h, state S [HD][N] <- S * exp(dt * A) + dt * x (outer) B; y = S C + D x.
    pool.parallelFor(H, [&](int h0, int h1, int) {
        for (int h = h0; h < h1; h++) {
            const int g = h / headsPerGroup;
            float* S = L.ssm + size_t(h) * size_t(HD) * size_t(N);
            for (int t = 0; t < T; t++) {
                const float* row = proj_.data() + size_t(t) * size_t(P);
                const float* xs = row + DI + size_t(h) * size_t(HD);
                const float* Bv = row + DI + DI + size_t(g) * size_t(N);
                const float* Cv = row + DI + DI + size_t(cfg_.mGroups) * size_t(N) + size_t(g) * size_t(N);
                const float dt = softplus(row[DI + CD + h] + L.dtBias[size_t(h)]);
                const float32x4_t dA = vdupq_n_f32(std::exp(dt * L.A[size_t(h)]));
                float* y = y_.data() + size_t(t) * size_t(DI) + size_t(h) * size_t(HD);
                for (int p = 0; p < HD; p++) {
                    float* s = S + size_t(p) * size_t(N);
                    const float32x4_t dx = vdupq_n_f32(dt * xs[p]);
                    float32x4_t acc = vdupq_n_f32(0.0f);
                    for (int n = 0; n < N; n += 4) {
                        float32x4_t sv = vfmaq_f32(vmulq_f32(vld1q_f32(s + n), dA), dx, vld1q_f32(Bv + n));
                        vst1q_f32(s + n, sv);
                        acc = vfmaq_f32(acc, sv, vld1q_f32(Cv + n));
                    }
                    y[p] = vaddvq_f32(acc) + L.D[size_t(h)] * xs[p];
                }
            }
        }
    });

    // Gate with silu(z), then RMSNorm over each group of channels.
    const int GS = DI / cfg_.mGroups;
    for (int t = 0; t < T; t++) {
        float* y = y_.data() + size_t(t) * size_t(DI);
        const float* z = proj_.data() + size_t(t) * size_t(P);
        for (int i = 0; i < DI; i++) y[i] *= silu(z[i]);
        for (int g = 0; g < cfg_.mGroups; g++)
            cpu::rmsNorm(y + size_t(g) * size_t(GS), y + size_t(g) * size_t(GS), L.norm.data() + size_t(g) * size_t(GS), 1,
                         GS, cfg_.eps);
    }
    cpu::matmul(pool, y_.data(), T, DI, L.outProj, nullptr, x_.data(), C, cpu::kAccumulate);
}

void GraniteModel::forward(const int* tokens, int T, int pos0, float* logits) {
    checkBatch(tokens, T, pos0);
    if (pos0 != statePos_)
        fail("Granite: forward at " + std::to_string(pos0) + " but the recurrent state is at " + std::to_string(statePos_));
    const int C = cfg_.nEmbd, H = cfg_.nHead, KV = cfg_.nKv, hd = cfg_.headDim, F = cfg_.nFf;
    const int QW = qkvWidth(), kOff = qWidth(), vOff = qWidth() + kvWidth();
    ThreadPool& pool = *pool_;
    for (int t = 0; t < T; t++) {
        float* x = x_.data() + size_t(t) * size_t(C);
        embed_.dequantizeRow(tokens[t], x);
        for (int c = 0; c < C; c++) x[c] *= cfg_.embeddingMult;
    }
    statePos_ = pos0 + T;  // the Mamba state advances below (a debug stop leaves it unusable; the CLI reloads)
    auto stopAt = [&](size_t l, int op, const std::vector<float>& v, int width) {
        if (debugStage_ != op || int(l) != debugLayers_) return false;
        std::copy(v.begin(), v.begin() + long(T) * width, logits);
        return true;
    };
    for (size_t l = 0; l < layers_.size(); l++) {
        if (debugLayers_ >= 0 && debugStage_ == 0 && int(l) >= debugLayers_) break;
        Layer& L = layers_[l];
        cpu::rmsNorm(x_.data(), xn_.data(), L.ln1.data(), T, C, cfg_.eps);
        if (stopAt(l, 1, xn_, C)) return;
        if (L.attention) {
            cpu::matmul(pool, xn_.data(), T, C, L.wqkv, nullptr, proj_.data(), QW, 0);
            cpu::storeKv(proj_.data() + kOff, proj_.data() + vOff, QW, L.kCache.get(), L.vCache.get(), kvWidth(), T, pos0);
            cpu::attention(pool, proj_.data(), QW, L.kCache.get(), L.vCache.get(), y_.data(), T, H, KV, hd, pos0);
            cpu::matmul(pool, y_.data(), T, qWidth(), L.wo, nullptr, x_.data(), C, cpu::kAccumulate);
        } else {
            mamba(L, T);
        }
        if (stopAt(l, 2, x_, C)) return;
        cpu::rmsNorm(x_.data(), xn_.data(), L.ln2.data(), T, C, cfg_.eps);
        if (stopAt(l, 3, xn_, C)) return;
        cpu::matmul(pool, xn_.data(), T, C, L.wgu, nullptr, h_.data(), 2 * F, 0);
        if (stopAt(l, 4, h_, 2 * F)) return;
        cpu::siluMul(h_.data(), g_.data(), T, F);
        if (stopAt(l, 5, g_, F)) return;
        cpu::matmul(pool, g_.data(), T, F, L.wdown, nullptr, x_.data(), C, cpu::kAccumulate);
        if (stopAt(l, 6, x_, C)) return;
    }
    if (debugLayers_ >= 0) {
        std::copy(x_.begin(), x_.begin() + long(T) * C, logits);
        return;
    }
    cpu::rmsNorm(x_.data() + size_t(T - 1) * size_t(C), xn_.data(), normF_.data(), 1, C, cfg_.eps);
    cpu::matmul(pool, xn_.data(), 1, C, lmHead_.empty() ? embed_ : lmHead_, nullptr, logits, cfg_.nVocab, 0);
    const float inv = 1.0f / cfg_.logitsScaling;
    for (int v = 0; v < cfg_.nVocab; v++) logits[v] *= inv;
}

}  // namespace neko
