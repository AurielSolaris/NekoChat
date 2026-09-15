// Qwen3 dense models (HF "qwen3": Qwen3-0.6B and up) on CPU, Vulkan or OpenGL ES.
// Pre-norm decoder: RMSNorm, grouped-query attention with per-head q/k RMSNorm and NeoX RoPE, SwiGLU MLP.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "model.h"

namespace neko {

struct Qwen3Config {
    int nVocab = 151936;
    int nMaxPos = 40960;
    int nEmbd = 1024;
    int nLayer = 28;
    int nHead = 16;
    int nKv = 8;
    int headDim = 128;
    int nFf = 3072;
    float eps = 1e-6f;
    double ropeTheta = 1000000.0;
    bool tied = true;
    bool qkvBias = false;
};

class Qwen3Model : public Model {
public:
    // KV memory grows with the context (f16: 2 * layers * kvHeads * headDim * 2 bytes per token), so the usable
    // context is capped well below the trained 40K positions. CPU caches are committed lazily as they fill.
    static constexpr int kCpuContext = 8192;
    static constexpr int kGpuContext = 4096;
    static constexpr int kGpuChunkRows = 32768;  // embedding/LM-head rows per buffer (dispatch and SSBO limits)

    Qwen3Model(const std::string& modelDir, const LoadOptions& opt, const ProgressFn& progress);
    ~Qwen3Model() override;

    const char* architecture() const override { return "qwen3"; }
    void forward(const int* tokens, int T, int pos0, float* logits) override;
    std::vector<DebugStage> debugStages() const override;

private:
    struct CpuLayer {
        std::vector<float> ln1, ln2, qNorm, kNorm, bqkv;
        QMatrix wqkv, wo, wgu, wdown;                 // [out][in]; wqkv = q|k|v rows, wgu = gate|up rows
        std::unique_ptr<uint16_t[]> kCache, vCache;  // f16 [ctx][nKv * headDim]
    };
    struct GpuLayer {
        gpu::Buffer *ln1, *ln2, *qkNorm, *bqkv, *kCache, *vCache;
        GpuMatrix wqkv, wo, wgu, wdown;
    };
    struct Chunk {
        GpuMatrix m;
        int row0, rows;
    };

    void loadConfig(const std::string& dir);
    bool gpuCompatible(const gpu::ComputeBackend& be, std::string& why) const;
    void loadCpu(TensorStore& ts, const ProgressFn& progress);
    void loadGpu(TensorStore& ts, const ProgressFn& progress);
    void forwardCpu(const int* tokens, int T, int pos0, float* logits);
    void forwardGpu(const int* tokens, int T, int pos0, float* logits);
    std::vector<float> ropeTable(int ctx) const;  // [pos][headDim / 2] (cos, sin)
    std::string layerName(int l, const char* leaf) const;

    int qkvWidth() const { return (cfg_.nHead + 2 * cfg_.nKv) * cfg_.headDim; }
    int qWidth() const { return cfg_.nHead * cfg_.headDim; }
    int kvWidth() const { return cfg_.nKv * cfg_.headDim; }

    Qwen3Config cfg_;
    std::string prefix_;  // "model." for HF checkpoints
    std::string head_;    // tensor used as LM head (embed_tokens when tied)

    // CPU path
    QMatrix embed_, lmHead_;  // lmHead_ empty when tied to embed_
    std::vector<float> normF_, rope_;
    std::vector<CpuLayer> layers_;
    std::vector<float> x_, xn_, qkv_, att_, h_, g_;

    // GPU path
    std::vector<GpuLayer> glayers_;
    std::vector<Chunk> gEmbed_, gHead_;  // gHead_ repeats gEmbed_ when tied
    gpu::Buffer *gNormF_ = nullptr, *gRope_ = nullptr, *gTokens_ = nullptr, *gX_ = nullptr, *gXn_ = nullptr,
                *gQkv_ = nullptr, *gAtt_ = nullptr, *gH_ = nullptr, *gG_ = nullptr, *gLogits_ = nullptr;
};

}  // namespace neko
