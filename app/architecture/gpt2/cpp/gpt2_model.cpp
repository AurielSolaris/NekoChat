#include "gpt2_model.h"

#include <algorithm>
#include <climits>
#include <cmath>

#include "common.h"
#include "cpu_ops.h"
#include "json.h"
#include "tensor_store.h"

namespace neko {

void Gpt2Model::loadConfig(const std::string& dir) {
    Json c = Json::parseFile(dir + "/config.json");
    cfg_.nVocab = int(c.getInt("vocab_size", 50257));
    cfg_.nCtx = int(c.getInt("n_positions", c.getInt("n_ctx", 1024)));
    cfg_.nEmbd = int(c.getInt("n_embd", 768));
    cfg_.nLayer = int(c.getInt("n_layer", 12));
    cfg_.nHead = int(c.getInt("n_head", 12));
    cfg_.eps = float(c.getNum("layer_norm_epsilon", 1e-5));
    if (cfg_.nEmbd % cfg_.nHead != 0) fail("n_embd not divisible by n_head");
    shape_.vocab = cfg_.nVocab;
    shape_.context = cfg_.nCtx;  // learned position table: the trained length is a hard limit
    shape_.layers = cfg_.nLayer;
    shape_.embd = cfg_.nEmbd;
    shape_.heads = shape_.kvHeads = cfg_.nHead;
    shape_.headDim = cfg_.nEmbd / cfg_.nHead;
}

Gpt2Model::Gpt2Model(const std::string& dir, BackendPref pref, int threads, const ProgressFn& progress) {
    progress(0.0f, "Reading config");
    loadConfig(dir);
    pool_ = std::make_unique<ThreadPool>(threads);

    TensorStore ts;
    progress(0.02f, "Mapping checkpoint");
    openCheckpoint(dir, ts);
    std::string prefix = ts.has("transformer.wte.weight") ? "transformer." : "";
    if (!ts.has(prefix + "wte.weight")) fail("checkpoint does not look like GPT-2 (no wte.weight)");
    for (auto& kv : ts.all())
        if (!ends_with(kv.first, ".attn.bias") && !ends_with(kv.first, ".attn.masked_bias") && kv.first != "lm_head.weight")
            params_ += kv.second.numel();

    startGpu(
        pref, progress, [&](gpu::ComputeBackend& be, std::string& why) { return gpuCompatible(be, why); },
        [&] { loadGpu(ts, prefix, progress); }, [&] { glayers_.clear(); });
    if (!gpu_) loadCpu(ts, prefix, progress);
    progress(1.0f, "Ready");
    NEKO_LOGI("GPT-2 loaded: %lld params, %d layers, ctx %d, backend %s", (long long)params_, cfg_.nLayer,
              shape_.context, backendName().c_str());
}

Gpt2Model::~Gpt2Model() = default;

std::vector<Model::DebugStage> Gpt2Model::debugStages() const {
    const int C = cfg_.nEmbd;
    return {{"ln1", C}, {"qkv", 3 * C}, {"attn", C}, {"proj+res", C}, {"ln2", C}, {"fc+gelu", 4 * C}, {"fc2+res", C}};
}

bool Gpt2Model::gpuCompatible(const gpu::ComputeBackend& be, std::string& why) const {
    const int C = cfg_.nEmbd;
    if (C % 8 != 0) { why = "n_embd must be a multiple of 8"; return false; }
    if (shape_.headDim > 256 || shape_.headDim % 2 != 0) { why = "unsupported head dim"; return false; }
    size_t biggest = size_t(cfg_.nVocab) * size_t(C) * 2;
    biggest = std::max(biggest, size_t(shape_.context) * size_t(C) * 2);
    if (biggest > be.maxBufferBytes()) { why = "weights exceed the max storage buffer size"; return false; }
    return true;
}

// ------------------------------------------------------------------ loading

void Gpt2Model::loadCpu(TensorStore& ts, const std::string& p, const ProgressFn& progress) {
    const int C = cfg_.nEmbd;
    progress(0.05f, "Loading embeddings");
    wte_ = ts.toF16(p + "wte.weight", false);
    wpe_ = ts.toF32(p + "wpe.weight");
    lnfG_ = ts.toF32(p + "ln_f.weight");
    lnfB_ = ts.toF32(p + "ln_f.bias");
    layers_.resize(size_t(cfg_.nLayer));
    for (int l = 0; l < cfg_.nLayer; l++) {
        progress(0.1f + 0.85f * float(l) / float(cfg_.nLayer), "Loading layer " + std::to_string(l + 1));
        std::string h = p + "h." + std::to_string(l) + ".";
        CpuLayer& L = layers_[size_t(l)];
        L.ln1g = ts.toF32(h + "ln_1.weight");
        L.ln1b = ts.toF32(h + "ln_1.bias");
        L.ln2g = ts.toF32(h + "ln_2.weight");
        L.ln2b = ts.toF32(h + "ln_2.bias");
        // HF GPT-2 uses Conv1D ([in][out]); transpose to [out][in] for row-wise dot products.
        L.wqkv = ts.toF16(h + "attn.c_attn.weight", true);
        L.bqkv = ts.toF32(h + "attn.c_attn.bias");
        L.wproj = ts.toF16(h + "attn.c_proj.weight", true);
        L.bproj = ts.toF32(h + "attn.c_proj.bias");
        L.wfc = ts.toF16(h + "mlp.c_fc.weight", true);
        L.bfc = ts.toF32(h + "mlp.c_fc.bias");
        L.wfc2 = ts.toF16(h + "mlp.c_proj.weight", true);
        L.bfc2 = ts.toF32(h + "mlp.c_proj.bias");
        // Uninitialised on purpose: only positions below the current length are ever read.
        L.kCache.reset(new uint16_t[size_t(shape_.context) * size_t(C)]);
        L.vCache.reset(new uint16_t[size_t(shape_.context) * size_t(C)]);
    }
    x_.resize(size_t(kMaxBatch) * size_t(C));
    xn_.resize(x_.size());
    att_.resize(x_.size());
    qkv_.resize(x_.size() * 3);
    h_.resize(x_.size() * 4);
}

void Gpt2Model::loadGpu(TensorStore& ts, const std::string& p, const ProgressFn& progress) {
    const int C = cfg_.nEmbd;
    gpu::ComputeBackend& be = *gpu_;
    auto f32 = [&](const std::string& n) {
        auto v = ts.toF32(n);
        return be.create(v.size() * 4, v.data());
    };
    auto f16 = [&](const std::string& n, bool t) {
        auto v = ts.toF16(n, t);
        return be.create(v.size() * 2, v.data());
    };
    progress(0.05f, std::string("Uploading embeddings (") + be.name() + ")");
    gWte_ = f16(p + "wte.weight", false);
    gWpe_ = f16(p + "wpe.weight", false);
    gLnfG_ = f32(p + "ln_f.weight");
    gLnfB_ = f32(p + "ln_f.bias");
    glayers_.resize(size_t(cfg_.nLayer));
    size_t cacheBytes = size_t(shape_.context) * size_t(C) * 2;
    for (int l = 0; l < cfg_.nLayer; l++) {
        progress(0.1f + 0.85f * float(l) / float(cfg_.nLayer), "Uploading layer " + std::to_string(l + 1));
        std::string h = p + "h." + std::to_string(l) + ".";
        GpuLayer& L = glayers_[size_t(l)];
        L.ln1g = f32(h + "ln_1.weight");
        L.ln1b = f32(h + "ln_1.bias");
        L.ln2g = f32(h + "ln_2.weight");
        L.ln2b = f32(h + "ln_2.bias");
        L.wqkv = f16(h + "attn.c_attn.weight", true);
        L.bqkv = f32(h + "attn.c_attn.bias");
        L.wproj = f16(h + "attn.c_proj.weight", true);
        L.bproj = f32(h + "attn.c_proj.bias");
        L.wfc = f16(h + "mlp.c_fc.weight", true);
        L.bfc = f32(h + "mlp.c_fc.bias");
        L.wfc2 = f16(h + "mlp.c_proj.weight", true);
        L.bfc2 = f32(h + "mlp.c_proj.bias");
        L.kCache = be.create(cacheBytes);
        L.vCache = be.create(cacheBytes);
    }
    size_t act = size_t(kMaxBatch) * size_t(C) * 4;
    gTokens_ = be.create(size_t(kMaxBatch) * 4);
    gX_ = be.create(act);
    gXn_ = be.create(act);
    gAtt_ = be.create(act);
    gQkv_ = be.create(act * 3);
    gH_ = be.create(act * 4);
    gLogits_ = be.create(size_t(cfg_.nVocab) * 4, nullptr, true);
}

// ------------------------------------------------------------------ forward

void Gpt2Model::forward(const int* tokens, int T, int pos0, float* logits) {
    checkBatch(tokens, T, pos0);
    if (gpu_) forwardGpu(tokens, T, pos0, logits);
    else forwardCpu(tokens, T, pos0, logits);
}

void Gpt2Model::forwardCpu(const int* tokens, int T, int pos0, float* logits) {
    const int C = cfg_.nEmbd, H = cfg_.nHead, hd = C / H;
    ThreadPool& pool = *pool_;
    for (int t = 0; t < T; t++) {
        const uint16_t* e = wte_.data() + size_t(tokens[t]) * size_t(C);
        const float* pe = wpe_.data() + size_t(pos0 + t) * size_t(C);
        float* xr = x_.data() + size_t(t) * size_t(C);
        for (int c = 0; c < C; c++) xr[c] = f16_to_f32(e[c]) + pe[c];
    }
    auto stopAt = [&](size_t l, int op, const std::vector<float>& v, int width) {
        if (debugStage_ != op || int(l) != debugLayers_) return false;
        std::copy(v.begin(), v.begin() + long(T) * width, logits);
        return true;
    };
    for (size_t l = 0; l < layers_.size(); l++) {
        if (debugLayers_ >= 0 && debugStage_ == 0 && int(l) >= debugLayers_) break;
        CpuLayer& L = layers_[l];
        cpu::layerNorm(x_.data(), xn_.data(), L.ln1g.data(), L.ln1b.data(), T, C, cfg_.eps);
        if (stopAt(l, 1, xn_, C)) return;
        cpu::matmulF16(pool, xn_.data(), T, C, L.wqkv.data(), L.bqkv.data(), qkv_.data(), 3 * C, 0);
        if (stopAt(l, 2, qkv_, 3 * C)) return;
        cpu::storeKv(qkv_.data() + C, qkv_.data() + 2 * C, 3 * C, L.kCache.get(), L.vCache.get(), C, T, pos0);
        cpu::attention(pool, qkv_.data(), 3 * C, L.kCache.get(), L.vCache.get(), att_.data(), T, H, H, hd, pos0);
        if (stopAt(l, 3, att_, C)) return;
        cpu::matmulF16(pool, att_.data(), T, C, L.wproj.data(), L.bproj.data(), x_.data(), C, cpu::kAccumulate);
        if (stopAt(l, 4, x_, C)) return;
        cpu::layerNorm(x_.data(), xn_.data(), L.ln2g.data(), L.ln2b.data(), T, C, cfg_.eps);
        if (stopAt(l, 5, xn_, C)) return;
        cpu::matmulF16(pool, xn_.data(), T, C, L.wfc.data(), L.bfc.data(), h_.data(), 4 * C, cpu::kGelu);
        if (stopAt(l, 6, h_, 4 * C)) return;
        cpu::matmulF16(pool, h_.data(), T, 4 * C, L.wfc2.data(), L.bfc2.data(), x_.data(), C, cpu::kAccumulate);
        if (stopAt(l, 7, x_, C)) return;
    }
    if (debugLayers_ >= 0) {
        std::copy(x_.begin(), x_.begin() + long(T) * C, logits);
        return;
    }
    cpu::layerNorm(x_.data() + size_t(T - 1) * size_t(C), xn_.data(), lnfG_.data(), lnfB_.data(), 1, C, cfg_.eps);
    cpu::matmulF16(pool, xn_.data(), 1, C, wte_.data(), nullptr, logits, cfg_.nVocab, 0);
}

void Gpt2Model::forwardGpu(const int* tokens, int T, int pos0, float* logits) {
    using gpu::Kernel;
    const int C = cfg_.nEmbd, H = cfg_.nHead, hd = C / H;
    const int32_t eps = gpu::floatBits(cfg_.eps);
    const int32_t scale = gpu::floatBits(1.0f / std::sqrt(float(hd)));
    const uint32_t tTiles = uint32_t((T + 3) / 4);
    const uint32_t cGroups = uint32_t((C + 63) / 64);
    const uint32_t pairGroups = uint32_t((C / 2 + 63) / 64);
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
    run(Kernel::Embed, gTokens_, gWte_, gWpe_, gX_, {T, C, pos0, 1, 0, cfg_.nVocab}, cGroups, uint32_t(T));
    auto stopAt = [&](size_t l, int op, gpu::Buffer* b, int width) {
        if (debugStage_ != op || int(l) != debugLayers_) return false;
        be.submitAndWait();
        be.download(b, 0, logits, size_t(T) * size_t(width) * 4);
        return true;
    };
    for (size_t l = 0; l < glayers_.size(); l++) {
        if (debugLayers_ >= 0 && debugStage_ == 0 && int(l) >= debugLayers_) break;
        GpuLayer& L = glayers_[l];
        run(Kernel::LayerNorm, gX_, gXn_, L.ln1g, L.ln1b, {C, eps, 0, 0}, uint32_t(T), 1);
        if (stopAt(l, 1, gXn_, C)) return;
        run(Kernel::Matmul, gXn_, L.wqkv, L.bqkv, gQkv_, {C, 3 * C, T, 4, 0}, uint32_t(3 * C), tTiles);
        if (stopAt(l, 2, gQkv_, 3 * C)) return;
        run(Kernel::KvStore, gQkv_, L.kCache, L.vCache, nullptr, {T, C, pos0, 0, 3 * C, C, 2 * C}, pairGroups, uint32_t(T));
        run(Kernel::Attention, gQkv_, L.kCache, L.vCache, gAtt_, {T, 3 * C, H, pos0, hd, scale, C}, uint32_t(H),
            uint32_t(T));
        if (stopAt(l, 3, gAtt_, C)) return;
        run(Kernel::Matmul, gAtt_, L.wproj, L.bproj, gX_, {C, C, T, 4 | 2, 0}, uint32_t(C), tTiles);
        if (stopAt(l, 4, gX_, C)) return;
        run(Kernel::LayerNorm, gX_, gXn_, L.ln2g, L.ln2b, {C, eps, 0, 0}, uint32_t(T), 1);
        if (stopAt(l, 5, gXn_, C)) return;
        run(Kernel::Matmul, gXn_, L.wfc, L.bfc, gH_, {C, 4 * C, T, 4 | 1, 0}, uint32_t(4 * C), tTiles);
        if (stopAt(l, 6, gH_, 4 * C)) return;
        run(Kernel::Matmul, gH_, L.wfc2, L.bfc2, gX_, {4 * C, C, T, 4 | 2, 0}, uint32_t(C), tTiles);
        if (stopAt(l, 7, gX_, C)) return;
    }
    if (debugLayers_ >= 0) {
        be.submitAndWait();
        be.download(gX_, 0, logits, size_t(T) * size_t(C) * 4);
        return;
    }
    // Final norm + LM head only for the last token.
    run(Kernel::LayerNorm, gX_, gXn_, gLnfG_, gLnfB_, {C, eps, T - 1, 0}, 1, 1);
    run(Kernel::Matmul, gXn_, gWte_, nullptr, gLogits_, {C, cfg_.nVocab, 1, 0, 0}, uint32_t(cfg_.nVocab), 1);
    be.submitAndWait();
    be.download(gLogits_, 0, logits, size_t(cfg_.nVocab) * 4);
}

}  // namespace neko
