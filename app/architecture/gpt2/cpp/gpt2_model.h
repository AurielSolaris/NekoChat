// GPT-2 (HF "gpt2" family: small/medium/large/xl and fine-tunes) on CPU, Vulkan or OpenGL ES.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "model.h"

namespace neko {

struct Gpt2Config {
    int nVocab = 50257;
    int nCtx = 1024;
    int nEmbd = 768;
    int nLayer = 12;
    int nHead = 12;
    float eps = 1e-5f;
};

class Gpt2Model : public Model {
public:
    Gpt2Model(const std::string& modelDir, const LoadOptions& opt, const ProgressFn& progress);
    ~Gpt2Model() override;

    const char* architecture() const override { return "gpt2"; }
    void forward(const int* tokens, int T, int pos0, float* logits) override;
    std::vector<DebugStage> debugStages() const override;

private:
    struct CpuLayer {
        std::vector<float> ln1g, ln1b, ln2g, ln2b, bqkv, bproj, bfc, bfc2;
        QMatrix wqkv, wproj, wfc, wfc2;              // [out][in]
        std::unique_ptr<uint16_t[]> kCache, vCache;  // f16 [ctx][C], committed lazily as positions fill
    };
    struct GpuLayer {
        gpu::Buffer *ln1g, *ln1b, *ln2g, *ln2b, *bqkv, *bproj, *bfc, *bfc2;
        GpuMatrix wqkv, wproj, wfc, wfc2;
        gpu::Buffer *kCache, *vCache;
    };

    void loadConfig(const std::string& dir);
    bool gpuCompatible(const gpu::ComputeBackend& be, std::string& why) const;
    void loadCpu(TensorStore& ts, const std::string& prefix, const ProgressFn& progress);
    void loadGpu(TensorStore& ts, const std::string& prefix, const ProgressFn& progress);
    // HF GPT-2 stores linear layers as Conv1D ([in][out]); returns them as [out][in].
    QMatrix conv1d(TensorStore& ts, const std::string& name);
    void forwardCpu(const int* tokens, int T, int pos0, float* logits);
    void forwardGpu(const int* tokens, int T, int pos0, float* logits);

    Gpt2Config cfg_;

    // CPU path
    QMatrix wte_;  // [V][C]: embedding table and tied LM head
    std::vector<float> wpe_, lnfG_, lnfB_;
    std::vector<CpuLayer> layers_;
    std::vector<float> x_, xn_, qkv_, att_, h_;

    // GPU path
    std::vector<GpuLayer> glayers_;
    GpuMatrix gWte_;
    gpu::Buffer *gWpe_ = nullptr, *gLnfG_ = nullptr, *gLnfB_ = nullptr;
    gpu::Buffer *gTokens_ = nullptr, *gX_ = nullptr, *gXn_ = nullptr, *gQkv_ = nullptr, *gAtt_ = nullptr, *gH_ = nullptr,
                *gLogits_ = nullptr;
};

}  // namespace neko
