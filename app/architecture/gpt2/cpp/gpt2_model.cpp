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

Gpt2Model::Gpt2Model(const std::string& dir, const LoadOptions& opt, const ProgressFn& progress) : Model(opt) {
    progress(0.0f, "Reading config");
    loadConfig(dir);
    pool_ = std::make_unique<ThreadPool>(opt.threads);

    TensorStore ts;
    progress(0.02f, "Mapping checkpoint");
    openCheckpoint(dir, ts);
    std::string prefix = ts.has("transformer.wte.weight") ? "transformer." : "";
    if (!ts.has(prefix + "wte.weight")) fail("checkpoint does not look like GPT-2 (no wte.weight)");
    for (auto& kv : ts.all())
        if (!ends_with(kv.first, ".attn.bias") && !ends_with(kv.first, ".attn.masked_bias") &&
            kv.first != "lm_head.weight" && !TensorStore::isQuantScale(kv.first))
            params_ += kv.second.numel();
    kvBytesPerToken_ = size_t(cfg_.nLayer) * 2 * size_t(cfg_.nEmbd) * 2;

    startGpu(
        opt.backend, progress, [&](gpu::ComputeBackend& be, std::string& why) { return gpuCompatible(be, why); },
        [&] { loadGpu(ts, prefix, progress); }, [&] { glayers_.clear(); });
    if (!gpu_) loadCpu(ts, prefix, progress);
    progress(1.0f, "Ready");
    NEKO_LOGI("GPT-2 loaded: %lld params, %d layers, ctx %d, %s weights (%zu MB), backend %s", (long long)params_,
              cfg_.nLayer, shape_.context, weightsLabel().c_str(), weightBytes_ >> 20, backendName().c_str());
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
    size_t biggest = size_t(cfg_.nVocab) * matrixRowBytes(C);
    biggest = std::max(biggest, size_t(shape_.context) * size_t(C) * 2);
    if (biggest > be.maxBufferBytes()) { why = "weights exceed the max storage buffer size"; return false; }
    return true;
}

// ------------------------------------------------------------------ loading

QMatrix Gpt2Model::conv1d(TensorStore& ts, const std::string& name) {
    const auto& shape = ts.get(name).shape;
    if (shape.size() != 2) fail("expected a 2-D weight: " + name);
    return matrix(ts.toF16(name, true), int(shape[1]), int(shape[0]));
}

void Gpt2Model::loadCpu(TensorStore& ts, const std::string& p, const ProgressFn& progress) {
    const int C = cfg_.nEmbd;
    const std::string verb = weightFormat() == WeightFormat::F16 ? "Loading" : "Quantizing";
    progress(0.05f, verb + " embeddings");
    wte_ = matrix(ts, {p + "wte.weight"});
    wpe_ = ts.toF32(p + "wpe.weight");
    lnfG_ = ts.toF32(p + "ln_f.weight");
    lnfB_ = ts.toF32(p + "ln_f.bias");
    weightBytes_ = wte_.bytes() + (wpe_.size() + lnfG_.size() + lnfB_.size()) * 4;
    layers_.resize(size_t(cfg_.nLayer));
    for (int l = 0; l < cfg_.nLayer; l++) {
        progress(0.1f + 0.85f * float(l) / float(cfg_.nLayer), verb + " layer " + std::to_string(l + 1));
        std::string h = p + "h." + std::to_string(l) + ".";
        CpuLayer& L = layers_[size_t(l)];
        L.ln1g = ts.toF32(h + "ln_1.weight");
        L.ln1b = ts.toF32(h + "ln_1.bias");
        L.ln2g = ts.toF32(h + "ln_2.weight");
        L.ln2b = ts.toF32(h + "ln_2.bias");
        L.wqkv = conv1d(ts, h + "attn.c_attn.weight");
        L.bqkv = ts.toF32(h + "attn.c_attn.bias");
        L.wproj = conv1d(ts, h + "attn.c_proj.weight");
        L.bproj = ts.toF32(h + "attn.c_proj.bias");
        L.wfc = conv1d(ts, h + "mlp.c_fc.weight");
        L.bfc = ts.toF32(h + "mlp.c_fc.bias");
        L.wfc2 = conv1d(ts, h + "mlp.c_proj.weight");
        L.bfc2 = ts.toF32(h + "mlp.c_proj.bias");
        weightBytes_ += L.wqkv.bytes() + L.wproj.bytes() + L.wfc.bytes() + L.wfc2.bytes() +
                        (L.ln1g.size() * 4 + L.bqkv.size() + L.bproj.size() + L.bfc.size() + L.bfc2.size()) * 4;
        // Uninitialised on purpose: only positions below the current length are ever read.
        L.kCache.reset(new uint16_t[size_t(shape_.context) * size_t(C)]);
        L.vCache.reset(new uint16_t[size_t(shape_.context) * size_t(C)]);
    }
    x_.resize(size_t(kMaxBatch) * size_t(C));
    xn_.resize(x_.size());
    att_.resize(x_.size());
    qkv_.resize(x_.size() * 3);
    h_.resize(x_.size() * 4);
    workBytes_ = (x_.size() + xn_.size() + att_.size() + qkv_.size() + h_.size()) * 4;
}

void Gpt2Model::loadGpu(TensorStore& ts, const std::string& p, const ProgressFn& progress) {
    const int C = cfg_.nEmbd;
    gpu::ComputeBackend& be = *gpu_;
    auto f32 = [&](const std::string& n) {
        auto v = ts.toF32(n);
        return weightBuffer(v.size() * 4, v.data());
    };
    const std::string verb = weightFormat() == WeightFormat::F16 ? "Uploading" : "Quantizing";
    progress(0.05f, verb + " embeddings (" + be.name() + ")");
    gWte_ = weightBuffer(matrix(ts, {p + "wte.weight"}));
    {
        auto wpe = ts.toF16(p + "wpe.weight", false);
        gWpe_ = weightBuffer(wpe.size() * 2, wpe.data());
    }
    gLnfG_ = f32(p + "ln_f.weight");
    gLnfB_ = f32(p + "ln_f.bias");
    glayers_.resize(size_t(cfg_.nLayer));
    size_t cacheBytes = size_t(shape_.context) * size_t(C) * 2;
    for (int l = 0; l < cfg_.nLayer; l++) {
        progress(0.1f + 0.85f * float(l) / float(cfg_.nLayer), verb + " layer " + std::to_string(l + 1));
        std::string h = p + "h." + std::to_string(l) + ".";
        GpuLayer& L = glayers_[size_t(l)];
        L.ln1g = f32(h + "ln_1.weight");
        L.ln1b = f32(h + "ln_1.bias");
        L.ln2g = f32(h + "ln_2.weight");
        L.ln2b = f32(h + "ln_2.bias");
        L.wqkv = weightBuffer(conv1d(ts, h + "attn.c_attn.weight"));
        L.bqkv = f32(h + "attn.c_attn.bias");
        L.wproj = weightBuffer(conv1d(ts, h + "attn.c_proj.weight"));
        L.bproj = f32(h + "attn.c_proj.bias");
        L.wfc = weightBuffer(conv1d(ts, h + "mlp.c_fc.weight"));
        L.bfc = f32(h + "mlp.c_fc.bias");
        L.wfc2 = weightBuffer(conv1d(ts, h + "mlp.c_proj.weight"));
        L.bfc2 = f32(h + "mlp.c_proj.bias");
        L.kCache = kvBuffer(cacheBytes);
        L.vCache = kvBuffer(cacheBytes);
    }
    size_t act = size_t(kMaxBatch) * size_t(C) * 4;
    gTokens_ = workBuffer(size_t(kMaxBatch) * 4);
    gX_ = workBuffer(act);
    gXn_ = workBuffer(act);
    gAtt_ = workBuffer(act);
    gQkv_ = workBuffer(act * 3);
    gH_ = workBuffer(act * 4);
    gLogits_ = workBuffer(size_t(cfg_.nVocab) * 4, true);
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
        float* xr = x_.data() + size_t(t) * size_t(C);
        wte_.dequantizeRow(tokens[t], xr);
        const float* pe = wpe_.data() + size_t(pos0 + t) * size_t(C);
        for (int c = 0; c < C; c++) xr[c] += pe[c];
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
        cpu::matmul(pool, xn_.data(), T, C, L.wqkv, L.bqkv.data(), qkv_.data(), 3 * C, 0);
        if (stopAt(l, 2, qkv_, 3 * C)) return;
        cpu::storeKv(qkv_.data() + C, qkv_.data() + 2 * C, 3 * C, L.kCache.get(), L.vCache.get(), C, T, pos0);
        cpu::attention(pool, qkv_.data(), 3 * C, L.kCache.get(), L.vCache.get(), att_.data(), T, H, H, hd, pos0);
        if (stopAt(l, 3, att_, C)) return;
        cpu::matmul(pool, att_.data(), T, C, L.wproj, L.bproj.data(), x_.data(), C, cpu::kAccumulate);
        if (stopAt(l, 4, x_, C)) return;
        cpu::layerNorm(x_.data(), xn_.data(), L.ln2g.data(), L.ln2b.data(), T, C, cfg_.eps);
        if (stopAt(l, 5, xn_, C)) return;
        cpu::matmul(pool, xn_.data(), T, C, L.wfc, L.bfc.data(), h_.data(), 4 * C, cpu::kGelu);
        if (stopAt(l, 6, h_, 4 * C)) return;
        cpu::matmul(pool, h_.data(), T, 4 * C, L.wfc2, L.bfc2.data(), x_.data(), C, cpu::kAccumulate);
        if (stopAt(l, 7, x_, C)) return;
    }
    if (debugLayers_ >= 0) {
        std::copy(x_.begin(), x_.begin() + long(T) * C, logits);
        return;
    }
    cpu::layerNorm(x_.data() + size_t(T - 1) * size_t(C), xn_.data(), lnfG_.data(), lnfB_.data(), 1, C, cfg_.eps);
    cpu::matmul(pool, xn_.data(), 1, C, wte_, nullptr, logits, cfg_.nVocab, 0);
}

void Gpt2Model::forwardGpu(const int* tokens, int T, int pos0, float* logits) {
    using gpu::Kernel;
    const int C = cfg_.nEmbd, H = cfg_.nHead, hd = C / H;
    const int32_t eps = gpu::floatBits(cfg_.eps);
    const int32_t scale = gpu::floatBits(1.0f / std::sqrt(float(hd)));
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
    const int fmt = gWte_.format == WeightFormat::FP8 ? 2 : gWte_.format == WeightFormat::FP4 ? 4 : 0;
    run(Kernel::Embed, gTokens_, gWte_.buf, gWpe_, gX_, {T, C, pos0, 1 | fmt, 0, cfg_.nVocab, int(gWte_.rowBytes / 4)},
        cGroups, uint32_t(T));
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
        gpuMatmul(L.wqkv, 3 * C, gXn_, L.bqkv, gQkv_, C, 3 * C, T, 4);
        if (stopAt(l, 2, gQkv_, 3 * C)) return;
        run(Kernel::KvStore, gQkv_, L.kCache, L.vCache, nullptr, {T, C, pos0, 0, 3 * C, C, 2 * C}, pairGroups, uint32_t(T));
        run(Kernel::Attention, gQkv_, L.kCache, L.vCache, gAtt_, {T, 3 * C, H, pos0, hd, scale, C}, uint32_t(H),
            uint32_t(T));
        if (stopAt(l, 3, gAtt_, C)) return;
        gpuMatmul(L.wproj, C, gAtt_, L.bproj, gX_, C, C, T, 4 | 2);
        if (stopAt(l, 4, gX_, C)) return;
        run(Kernel::LayerNorm, gX_, gXn_, L.ln2g, L.ln2b, {C, eps, 0, 0}, uint32_t(T), 1);
        if (stopAt(l, 5, gXn_, C)) return;
        gpuMatmul(L.wfc, 4 * C, gXn_, L.bfc, gH_, C, 4 * C, T, 4 | 1);
        if (stopAt(l, 6, gH_, 4 * C)) return;
        gpuMatmul(L.wfc2, C, gH_, L.bfc2, gX_, 4 * C, C, T, 4 | 2);
        if (stopAt(l, 7, gX_, C)) return;
        gpuYield(l);
    }
    if (debugLayers_ >= 0) {
        be.submitAndWait();
        be.download(gX_, 0, logits, size_t(T) * size_t(C) * 4);
        return;
    }
    // Final norm + LM head only for the last token.
    run(Kernel::LayerNorm, gX_, gXn_, gLnfG_, gLnfB_, {C, eps, T - 1, 0}, 1, 1);
    gpuMatmul(gWte_, cfg_.nVocab, gXn_, nullptr, gLogits_, C, cfg_.nVocab, 1, 0);
    be.submitAndWait();
    be.download(gLogits_, 0, logits, size_t(cfg_.nVocab) * 4);
}

}  // namespace neko
