// Architecture-independent model interface. Each architecture (gpt2/, qwen3/) implements Model
// for CPU and GPU; the engine only sees this interface.
#pragma once

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gpu/compute.h"
#include "quant.h"
#include "thread_pool.h"

namespace neko {

enum class BackendPref : int { Auto = 0, Vulkan = 1, OpenGL = 2, CPU = 3 };

using ProgressFn = std::function<void(float fraction, const std::string& stage)>;

class TensorStore;

struct LoadOptions {
    BackendPref backend = BackendPref::Auto;
    int threads = 0;  // 0 = one per performance core
    WeightFormat weights = WeightFormat::F16;
};

struct ModelShape {
    int vocab = 0;
    int context = 0;  // usable context (may be below the model's trained maximum to bound KV memory)
    int layers = 0;
    int embd = 0;
    int heads = 0;
    int kvHeads = 0;
    int headDim = 0;
};

// Bytes held by a loaded model. The KV cache fills as the conversation grows: CPU pages are committed as
// positions fill (resident = used), the GPU reserves the whole context up front (resident = capacity).
struct MemoryStats {
    size_t weights = 0;
    size_t kvUsed = 0;
    size_t kvCapacity = 0;
    size_t kvResident = 0;
    size_t workspace = 0;  // activations, logits

    size_t total() const { return weights + kvResident + workspace; }
};

// A weight matrix in GPU memory; the format picks the matmul kernel.
struct GpuMatrix {
    gpu::Buffer* buf = nullptr;
    WeightFormat format = WeightFormat::F16;
    size_t rowBytes = 0;
};

class Model {
public:
    static constexpr int kMaxBatch = 64;  // prompt tokens per forward call

    virtual ~Model() = default;

    virtual const char* architecture() const = 0;
    // Processes tokens[0..T) at positions [pos0, pos0 + T) and writes the last token's logits (vocab floats).
    virtual void forward(const int* tokens, int T, int pos0, float* logits) = 0;

    const ModelShape& shape() const { return shape_; }
    int contextLength() const { return shape_.context; }
    std::string backendName() const { return gpu_ ? gpu_->name() : "CPU"; }
    std::string deviceName() const;
    const std::string& backendNote() const { return note_; }
    int64_t parameterCount() const { return params_; }
    const std::string& checkpointFormat() const { return format_; }
    // Precision the weights are stored in, e.g. "FP8" ("FP8 + FP16" when some matrices could not be quantized).
    std::string weightsLabel() const;
    WeightFormat weightFormat() const { return weights_; }
    MemoryStats memory(int cachedTokens) const;

    // Validation (nekochat_cli --bisect): when n >= 0, forward() stops after n layers and writes the hidden
    // states [T][embd] instead of logits. With stage s >= 1 it stops inside layer n after debugStages()[s - 1]
    // and dumps that intermediate ([T][width] floats).
    struct DebugStage {
        const char* name;
        int width;
    };
    virtual std::vector<DebugStage> debugStages() const = 0;
    void setDebugLayers(int n, int stage = 0) {
        debugLayers_ = n;
        debugStage_ = stage;
    }

protected:
    explicit Model(const LoadOptions& opt) : weights_(opt.weights) {}

    // Maps every checkpoint file in dir (all *.safetensors shards, else the first .pt/.pth/.bin).
    void openCheckpoint(const std::string& dir, TensorStore& ts);
    void checkBatch(const int* tokens, int T, int pos0) const;

    // Backend selection: preferred GPU API first, then the other one, else CPU. A GPU is only used after
    // compatible() and the kernel self-test pass; load() uploads the weights and may throw, in which case
    // unload() drops partial state and the next option is tried. Reasons end up in backendNote().
    void startGpu(BackendPref pref, const ProgressFn& progress,
                  const std::function<bool(gpu::ComputeBackend&, std::string&)>& compatible,
                  const std::function<void()>& load, const std::function<void()>& unload);

    // Weights in the requested format: the rows of all named tensors, concatenated. CPU loaders add what they
    // keep to weightBytes_; GPU uploads are counted by weightBuffer().
    QMatrix matrix(TensorStore& ts, std::initializer_list<std::string> names);
    // Same from an f16 [rows][cols] array (GPT-2's Conv1D weights after transposing).
    QMatrix matrix(const std::vector<uint16_t>& f16, int rows, int cols);
    size_t matrixRowBytes(int cols) const { return quantRowBytes(formatFor(cols, weights_), cols); }

    // GPU allocations, counted by kind for memory().
    gpu::Buffer* weightBuffer(size_t bytes, const void* data);
    GpuMatrix weightBuffer(const QMatrix& m);
    gpu::Buffer* kvBuffer(size_t bytes);
    gpu::Buffer* workBuffer(size_t bytes, bool readback = false);

    // y[t][nOffset + n] (+)= act(x[xRow0 + t] . W[n] + bias[n]) for n < rows. flags as in matmul.comp.
    void gpuMatmul(const GpuMatrix& w, int rows, gpu::Buffer* x, gpu::Buffer* bias, gpu::Buffer* y, int K, int yStride,
                   int T, int flags, int xRow0 = 0, int nOffset = 0);
    // Lets other GPU work (the UI, the keyboard) run between layers; see ComputeBackend::flush().
    void gpuYield(size_t layer) {
        if (gpu_ && layer % kLayersPerSubmit == kLayersPerSubmit - 1) gpu_->flush();
    }

    static constexpr size_t kLayersPerSubmit = 1;

    ModelShape shape_;
    int64_t params_ = 0;
    std::string note_;
    std::string format_;
    std::unique_ptr<ThreadPool> pool_;
    std::unique_ptr<gpu::ComputeBackend> gpu_;
    int debugLayers_ = -1;
    int debugStage_ = 0;

    // Memory bookkeeping: weights and workspace are counted as they are allocated; KV per token is set by the
    // architecture, kvPreallocated_ when the whole context is reserved up front (GPU).
    size_t weightBytes_ = 0;
    size_t workBytes_ = 0;
    size_t kvBytesPerToken_ = 0;
    bool kvPreallocated_ = false;

private:
    static WeightFormat formatFor(int cols, WeightFormat f) { return cols % kQuantBlock == 0 ? f : WeightFormat::F16; }
    void noteFormat(WeightFormat f) { used_ |= 1u << int(f); }

    WeightFormat weights_;
    unsigned used_ = 0;  // bit per WeightFormat actually stored (F16 fallback for widths that aren't % 32)
};

// config.json "model_type" ("gpt2", "qwen3", ...).
std::string readModelType(const std::string& modelDir);

std::unique_ptr<Model> loadModel(const std::string& modelType, const std::string& modelDir, const LoadOptions& opt,
                                 const ProgressFn& progress);

}  // namespace neko
