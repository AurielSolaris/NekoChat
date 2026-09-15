#include "qwen3_model.h"

#include <algorithm>
#include <cmath>

#include "common.h"
#include "cpu_ops.h"
#include "json.h"
#include "tensor_store.h"

namespace neko {

void Qwen3Model::loadConfig(const std::string& dir) {
    Json c = Json::parseFile(dir + "/config.json");
    cfg_.nVocab = int(c.getInt("vocab_size", 151936));
    cfg_.nMaxPos = int(c.getInt("max_position_embeddings", 40960));
    cfg_.nEmbd = int(c.getInt("hidden_size", 1024));
    cfg_.nLayer = int(c.getInt("num_hidden_layers", 28));
    cfg_.nHead = int(c.getInt("num_attention_heads", 16));
    cfg_.nKv = int(c.getInt("num_key_value_heads", cfg_.nHead));
    cfg_.headDim = int(c.getInt("head_dim", cfg_.nEmbd / cfg_.nHead));
    cfg_.nFf = int(c.getInt("intermediate_size", 3072));
    cfg_.eps = float(c.getNum("rms_norm_eps", 1e-6));
    cfg_.ropeTheta = c.getNum("rope_theta", 1000000.0);
    cfg_.tied = c["tie_word_embeddings"].boolean(false);
    cfg_.qkvBias = c["attention_bias"].boolean(false);
    if (cfg_.nKv <= 0 || cfg_.nHead % cfg_.nKv != 0) fail("num_attention_heads must be a multiple of num_key_value_heads");
    if (cfg_.headDim % 2 != 0) fail("head_dim must be even");
    if (!c["rope_scaling"].isNull()) NEKO_LOGW("rope_scaling is ignored (fine within the default context)");
    shape_.vocab = cfg_.nVocab;
    shape_.context = std::min(cfg_.nMaxPos, kCpuContext);
    shape_.layers = cfg_.nLayer;
    shape_.embd = cfg_.nEmbd;
    shape_.heads = cfg_.nHead;
    shape_.kvHeads = cfg_.nKv;
    shape_.headDim = cfg_.headDim;
}

std::string Qwen3Model::layerName(int l, const char* leaf) const {
    return prefix_ + "layers." + std::to_string(l) + "." + leaf;
}

Qwen3Model::Qwen3Model(const std::string& dir, const LoadOptions& opt, const ProgressFn& progress) : Model(opt) {
    progress(0.0f, "Reading config");
    loadConfig(dir);
    pool_ = std::make_unique<ThreadPool>(opt.threads);

    TensorStore ts;
    progress(0.02f, "Mapping checkpoint");
    openCheckpoint(dir, ts);
    prefix_ = ts.has("model.embed_tokens.weight") ? "model." : "";
    if (!ts.has(prefix_ + "embed_tokens.weight")) fail("checkpoint does not look like Qwen3 (no embed_tokens.weight)");
    if (!ts.has(layerName(0, "self_attn.q_norm.weight"))) fail("checkpoint has no q_norm (Qwen2 is not supported)");
    // Tied checkpoints may still carry a copy of lm_head; the embedding table is used either way.
    head_ = !cfg_.tied && ts.has("lm_head.weight") ? "lm_head.weight" : prefix_ + "embed_tokens.weight";
    for (auto& kv : ts.all())
        if (!TensorStore::isQuantScale(kv.first) && (kv.first != "lm_head.weight" || head_ == "lm_head.weight"))
            params_ += kv.second.numel();
    kvBytesPerToken_ = size_t(cfg_.nLayer) * 2 * size_t(kvWidth()) * 2;

    startGpu(
        opt.backend, progress, [&](gpu::ComputeBackend& be, std::string& why) { return gpuCompatible(be, why); },
        [&] { loadGpu(ts, progress); },
        [&] {
            glayers_.clear();
            gEmbed_.clear();
            gHead_.clear();
        });
    if (!gpu_) loadCpu(ts, progress);
    progress(1.0f, "Ready");
    NEKO_LOGI("Qwen3 loaded: %lld params, %d layers, ctx %d, %s weights (%zu MB), backend %s", (long long)params_,
              cfg_.nLayer, shape_.context, weightsLabel().c_str(), weightBytes_ >> 20, backendName().c_str());
}

Qwen3Model::~Qwen3Model() = default;

std::vector<Model::DebugStage> Qwen3Model::debugStages() const {
    const int C = cfg_.nEmbd;
    return {{"ln1", C},        {"qkv", qkvWidth()}, {"qk-norm+rope", qkvWidth()}, {"attn", qWidth()},
            {"o_proj+res", C}, {"ln2", C},          {"gate|up", 2 * cfg_.nFf},    {"silu*up", cfg_.nFf},
            {"down+res", C}};
}

bool Qwen3Model::gpuCompatible(const gpu::ComputeBackend& be, std::string& why) const {
    const int C = cfg_.nEmbd, F = cfg_.nFf;
    if (C % 8 != 0 || F % 8 != 0 || qWidth() % 8 != 0) { why = "matrix sizes must be multiples of 8"; return false; }
    if (cfg_.headDim > 256) { why = "head dim > 256"; return false; }
    size_t ctx = size_t(std::min(cfg_.nMaxPos, kGpuContext));
    size_t biggest = std::max({size_t(2 * F) * matrixRowBytes(C), size_t(qkvWidth()) * matrixRowBytes(C),
                               size_t(C) * matrixRowBytes(F), size_t(kGpuChunkRows) * matrixRowBytes(C),
                               ctx * size_t(kvWidth()) * 2, size_t(kMaxBatch) * size_t(2 * F) * 4,
                               ctx * size_t(cfg_.headDim) * 4});
    if (biggest > be.maxBufferBytes()) { why = "weights exceed the max storage buffer size"; return false; }
    return true;
}

// ------------------------------------------------------------------ loading

std::vector<float> Qwen3Model::ropeTable(int ctx) const {
    const int hd = cfg_.headDim;
    std::vector<float> t(size_t(ctx) * size_t(hd));
    for (int i = 0; i < hd / 2; i++) {
        double inv = std::pow(cfg_.ropeTheta, -2.0 * i / hd);
        for (int p = 0; p < ctx; p++) {
            double a = p * inv;
            t[size_t(p) * size_t(hd) + size_t(2 * i)] = float(std::cos(a));
            t[size_t(p) * size_t(hd) + size_t(2 * i + 1)] = float(std::sin(a));
        }
    }
    return t;
}

void Qwen3Model::loadCpu(TensorStore& ts, const ProgressFn& progress) {
    shape_.context = std::min(cfg_.nMaxPos, kCpuContext);
    const int C = cfg_.nEmbd, F = cfg_.nFf;
    const std::string verb = weightFormat() == WeightFormat::F16 ? "Loading" : "Quantizing";
    progress(0.05f, verb + " embeddings");
    embed_ = matrix(ts, {prefix_ + "embed_tokens.weight"});
    if (head_ == "lm_head.weight") lmHead_ = matrix(ts, {head_});
    normF_ = ts.toF32(prefix_ + "norm.weight");
    rope_ = ropeTable(shape_.context);
    weightBytes_ = embed_.bytes() + lmHead_.bytes() + normF_.size() * 4;
    layers_.resize(size_t(cfg_.nLayer));
    const size_t cache = size_t(shape_.context) * size_t(kvWidth());
    for (int l = 0; l < cfg_.nLayer; l++) {
        progress(0.1f + 0.85f * float(l) / float(cfg_.nLayer), verb + " layer " + std::to_string(l + 1));
        CpuLayer& L = layers_[size_t(l)];
        L.ln1 = ts.toF32(layerName(l, "input_layernorm.weight"));
        L.ln2 = ts.toF32(layerName(l, "post_attention_layernorm.weight"));
        L.qNorm = ts.toF32(layerName(l, "self_attn.q_norm.weight"));
        L.kNorm = ts.toF32(layerName(l, "self_attn.k_norm.weight"));
        L.wqkv = matrix(ts, {layerName(l, "self_attn.q_proj.weight"), layerName(l, "self_attn.k_proj.weight"),
                             layerName(l, "self_attn.v_proj.weight")});
        if (cfg_.qkvBias) {
            for (const char* b : {"self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias"}) {
                auto v = ts.toF32(layerName(l, b));
                L.bqkv.insert(L.bqkv.end(), v.begin(), v.end());
            }
        }
        L.wo = matrix(ts, {layerName(l, "self_attn.o_proj.weight")});
        L.wgu = matrix(ts, {layerName(l, "mlp.gate_proj.weight"), layerName(l, "mlp.up_proj.weight")});
        L.wdown = matrix(ts, {layerName(l, "mlp.down_proj.weight")});
        weightBytes_ += L.wqkv.bytes() + L.wo.bytes() + L.wgu.bytes() + L.wdown.bytes() +
                        (L.ln1.size() + L.ln2.size() + L.qNorm.size() + L.kNorm.size() + L.bqkv.size()) * 4;
        // Uninitialised on purpose: pages are only committed once positions are written.
        L.kCache.reset(new uint16_t[cache]);
        L.vCache.reset(new uint16_t[cache]);
    }
    const size_t B = kMaxBatch;
    x_.resize(B * size_t(C));
    xn_.resize(B * size_t(C));
    qkv_.resize(B * size_t(qkvWidth()));
    att_.resize(B * size_t(qWidth()));
    h_.resize(B * size_t(2 * F));
    g_.resize(B * size_t(F));
    workBytes_ = (x_.size() + xn_.size() + qkv_.size() + att_.size() + h_.size() + g_.size() + rope_.size()) * 4;
}

void Qwen3Model::loadGpu(TensorStore& ts, const ProgressFn& progress) {
    shape_.context = std::min(cfg_.nMaxPos, kGpuContext);
    const int C = cfg_.nEmbd, F = cfg_.nFf;
    gpu::ComputeBackend& be = *gpu_;
    auto f32 = [&](const std::vector<float>& v) { return weightBuffer(v.size() * 4, v.data()); };
    auto chunked = [&](const std::string& name) {
        QMatrix all = matrix(ts, {name});
        std::vector<Chunk> out;
        for (int r0 = 0; r0 < cfg_.nVocab; r0 += kGpuChunkRows) {
            int rows = std::min(kGpuChunkRows, cfg_.nVocab - r0);
            out.push_back({weightBuffer(all.sliceRows(r0, rows)), r0, rows});
        }
        return out;
    };
    const std::string verb = weightFormat() == WeightFormat::F16 ? "Uploading" : "Quantizing";
    progress(0.05f, verb + " embeddings (" + be.name() + ")");
    gEmbed_ = chunked(prefix_ + "embed_tokens.weight");
    gHead_ = head_ == "lm_head.weight" ? chunked(head_) : gEmbed_;
    gNormF_ = f32(ts.toF32(prefix_ + "norm.weight"));
    std::vector<float> rope = ropeTable(shape_.context);
    gRope_ = workBuffer(rope.size() * 4);
    be.upload(gRope_, 0, rope.data(), rope.size() * 4);
    glayers_.resize(size_t(cfg_.nLayer));
    const size_t cacheBytes = size_t(shape_.context) * size_t(kvWidth()) * 2;
    for (int l = 0; l < cfg_.nLayer; l++) {
        progress(0.1f + 0.85f * float(l) / float(cfg_.nLayer), verb + " layer " + std::to_string(l + 1));
        GpuLayer& L = glayers_[size_t(l)];
        L.ln1 = f32(ts.toF32(layerName(l, "input_layernorm.weight")));
        L.ln2 = f32(ts.toF32(layerName(l, "post_attention_layernorm.weight")));
        std::vector<float> qk = ts.toF32(layerName(l, "self_attn.q_norm.weight"));
        std::vector<float> k = ts.toF32(layerName(l, "self_attn.k_norm.weight"));
        qk.insert(qk.end(), k.begin(), k.end());
        L.qkNorm = f32(qk);
        L.wqkv = weightBuffer(matrix(ts, {layerName(l, "self_attn.q_proj.weight"),
                                          layerName(l, "self_attn.k_proj.weight"),
                                          layerName(l, "self_attn.v_proj.weight")}));
        L.bqkv = nullptr;
        if (cfg_.qkvBias) {
            std::vector<float> b;
            for (const char* n : {"self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias"}) {
                auto v = ts.toF32(layerName(l, n));
                b.insert(b.end(), v.begin(), v.end());
            }
            L.bqkv = f32(b);
        }
        L.wo = weightBuffer(matrix(ts, {layerName(l, "self_attn.o_proj.weight")}));
        L.wgu = weightBuffer(matrix(ts, {layerName(l, "mlp.gate_proj.weight"), layerName(l, "mlp.up_proj.weight")}));
        L.wdown = weightBuffer(matrix(ts, {layerName(l, "mlp.down_proj.weight")}));
        L.kCache = kvBuffer(cacheBytes);
        L.vCache = kvBuffer(cacheBytes);
    }
    const size_t B = kMaxBatch;
    gTokens_ = workBuffer(B * 4);
    gX_ = workBuffer(B * size_t(C) * 4);
    gXn_ = workBuffer(B * size_t(C) * 4);
    gQkv_ = workBuffer(B * size_t(qkvWidth()) * 4);
    gAtt_ = workBuffer(B * size_t(qWidth()) * 4);
    gH_ = workBuffer(B * size_t(2 * F) * 4);
    gG_ = workBuffer(B * size_t(F) * 4);
    gLogits_ = workBuffer(size_t(cfg_.nVocab) * 4, true);
}

// ------------------------------------------------------------------ forward

void Qwen3Model::forward(const int* tokens, int T, int pos0, float* logits) {
    checkBatch(tokens, T, pos0);
    if (gpu_) forwardGpu(tokens, T, pos0, logits);
    else forwardCpu(tokens, T, pos0, logits);
}

void Qwen3Model::forwardCpu(const int* tokens, int T, int pos0, float* logits) {
    const int C = cfg_.nEmbd, H = cfg_.nHead, KV = cfg_.nKv, hd = cfg_.headDim, F = cfg_.nFf;
    const int QW = qkvWidth(), kOff = qWidth(), vOff = qWidth() + kvWidth();
    ThreadPool& pool = *pool_;
    for (int t = 0; t < T; t++) embed_.dequantizeRow(tokens[t], x_.data() + size_t(t) * size_t(C));
    auto stopAt = [&](size_t l, int op, const std::vector<float>& v, int width) {
        if (debugStage_ != op || int(l) != debugLayers_) return false;
        std::copy(v.begin(), v.begin() + long(T) * width, logits);
        return true;
    };
    for (size_t l = 0; l < layers_.size(); l++) {
        if (debugLayers_ >= 0 && debugStage_ == 0 && int(l) >= debugLayers_) break;
        CpuLayer& L = layers_[l];
        cpu::rmsNorm(x_.data(), xn_.data(), L.ln1.data(), T, C, cfg_.eps);
        if (stopAt(l, 1, xn_, C)) return;
        cpu::matmul(pool, xn_.data(), T, C, L.wqkv, L.bqkv.empty() ? nullptr : L.bqkv.data(), qkv_.data(), QW, 0);
        if (stopAt(l, 2, qkv_, QW)) return;
        for (int t = 0; t < T; t++) {
            float* r = qkv_.data() + size_t(t) * size_t(QW);
            cpu::rmsNorm(r, r, L.qNorm.data(), H, hd, cfg_.eps);
            cpu::rmsNorm(r + kOff, r + kOff, L.kNorm.data(), KV, hd, cfg_.eps);
            cpu::rope(r, H + KV, hd, rope_.data() + size_t(pos0 + t) * size_t(hd));  // keys follow the queries
        }
        if (stopAt(l, 3, qkv_, QW)) return;
        cpu::storeKv(qkv_.data() + kOff, qkv_.data() + vOff, QW, L.kCache.get(), L.vCache.get(), kvWidth(), T, pos0);
        cpu::attention(pool, qkv_.data(), QW, L.kCache.get(), L.vCache.get(), att_.data(), T, H, KV, hd, pos0);
        if (stopAt(l, 4, att_, qWidth())) return;
        cpu::matmul(pool, att_.data(), T, qWidth(), L.wo, nullptr, x_.data(), C, cpu::kAccumulate);
        if (stopAt(l, 5, x_, C)) return;
        cpu::rmsNorm(x_.data(), xn_.data(), L.ln2.data(), T, C, cfg_.eps);
        if (stopAt(l, 6, xn_, C)) return;
        cpu::matmul(pool, xn_.data(), T, C, L.wgu, nullptr, h_.data(), 2 * F, 0);
        if (stopAt(l, 7, h_, 2 * F)) return;
        cpu::siluMul(h_.data(), g_.data(), T, F);
        if (stopAt(l, 8, g_, F)) return;
        cpu::matmul(pool, g_.data(), T, F, L.wdown, nullptr, x_.data(), C, cpu::kAccumulate);
        if (stopAt(l, 9, x_, C)) return;
    }
    if (debugLayers_ >= 0) {
        std::copy(x_.begin(), x_.begin() + long(T) * C, logits);
        return;
    }
    cpu::rmsNorm(x_.data() + size_t(T - 1) * size_t(C), xn_.data(), normF_.data(), 1, C, cfg_.eps);
    cpu::matmul(pool, xn_.data(), 1, C, lmHead_.empty() ? embed_ : lmHead_, nullptr, logits, cfg_.nVocab, 0);
}

void Qwen3Model::forwardGpu(const int* tokens, int T, int pos0, float* logits) {
    using gpu::Kernel;
    const int C = cfg_.nEmbd, H = cfg_.nHead, KV = cfg_.nKv, hd = cfg_.headDim, F = cfg_.nFf;
    const int QW = qkvWidth(), QH = qWidth(), KW = kvWidth();
    const int32_t eps = gpu::floatBits(cfg_.eps);
    const int32_t scale = gpu::floatBits(1.0f / std::sqrt(float(hd)));
    gpu::ComputeBackend& be = *gpu_;

    be.upload(gTokens_, 0, tokens, size_t(T) * 4);
    be.begin();
    auto run = [&](Kernel k, gpu::Buffer* a, gpu::Buffer* b, gpu::Buffer* c, gpu::Buffer* d, std::initializer_list<int32_t> p,
                   uint32_t gx, uint32_t gy) {
        gpu::Buffer* binds[4] = {a, b, c, d};
        int32_t params[8] = {};
        std::copy(p.begin(), p.end(), params);
        be.dispatch(k, binds, params, gx, gy);
    };
    auto stopAt = [&](size_t l, int op, gpu::Buffer* b, int width) {
        if (debugStage_ != op || int(l) != debugLayers_) return false;
        be.submitAndWait();
        be.download(b, 0, logits, size_t(T) * size_t(width) * 4);
        return true;
    };
    for (const Chunk& ch : gEmbed_) {
        const int fmt = ch.m.format == WeightFormat::FP8 ? 2 : ch.m.format == WeightFormat::FP4 ? 4 : 0;
        run(Kernel::Embed, gTokens_, ch.m.buf, nullptr, gX_, {T, C, 0, fmt, ch.row0, ch.rows, int(ch.m.rowBytes / 4)},
            uint32_t((C + 63) / 64), uint32_t(T));
    }
    for (size_t l = 0; l < glayers_.size(); l++) {
        if (debugLayers_ >= 0 && debugStage_ == 0 && int(l) >= debugLayers_) break;
        GpuLayer& L = glayers_[l];
        run(Kernel::LayerNorm, gX_, gXn_, L.ln1, nullptr, {C, eps, 0, 1}, uint32_t(T), 1);
        if (stopAt(l, 1, gXn_, C)) return;
        gpuMatmul(L.wqkv, QW, gXn_, L.bqkv, gQkv_, C, QW, T, L.bqkv ? 4 : 0);
        if (stopAt(l, 2, gQkv_, QW)) return;
        run(Kernel::Rope, gQkv_, L.qkNorm, gRope_, nullptr, {QW, H, hd, pos0, eps, 1}, uint32_t(H + KV), uint32_t(T));
        if (stopAt(l, 3, gQkv_, QW)) return;
        run(Kernel::KvStore, gQkv_, L.kCache, L.vCache, nullptr, {T, KW, pos0, 0, QW, QH, QH + KW},
            uint32_t((KW / 2 + 63) / 64), uint32_t(T));
        run(Kernel::Attention, gQkv_, L.kCache, L.vCache, gAtt_, {T, QW, H, pos0, hd, scale, KW}, uint32_t(H), uint32_t(T));
        if (stopAt(l, 4, gAtt_, QH)) return;
        gpuMatmul(L.wo, C, gAtt_, nullptr, gX_, QH, C, T, 2);
        if (stopAt(l, 5, gX_, C)) return;
        run(Kernel::LayerNorm, gX_, gXn_, L.ln2, nullptr, {C, eps, 0, 1}, uint32_t(T), 1);
        if (stopAt(l, 6, gXn_, C)) return;
        gpuMatmul(L.wgu, 2 * F, gXn_, nullptr, gH_, C, 2 * F, T, 0);
        if (stopAt(l, 7, gH_, 2 * F)) return;
        run(Kernel::SiluMul, gH_, gG_, nullptr, nullptr, {T, F}, uint32_t((F + 63) / 64), uint32_t(T));
        if (stopAt(l, 8, gG_, F)) return;
        gpuMatmul(L.wdown, C, gG_, nullptr, gX_, F, C, T, 2);
        if (stopAt(l, 9, gX_, C)) return;
        gpuYield(l);
    }
    if (debugLayers_ >= 0) {
        be.submitAndWait();
        be.download(gX_, 0, logits, size_t(T) * size_t(C) * 4);
        return;
    }
    // Final norm + LM head only for the last token; the head is split across buffers.
    run(Kernel::LayerNorm, gX_, gXn_, gNormF_, nullptr, {C, eps, T - 1, 1}, 1, 1);
    for (const Chunk& ch : gHead_) gpuMatmul(ch.m, ch.rows, gXn_, nullptr, gLogits_, C, cfg_.nVocab, 1, 0, 0, ch.row0);
    be.submitAndWait();
    be.download(gLogits_, 0, logits, size_t(cfg_.nVocab) * 4);
}

}  // namespace neko
