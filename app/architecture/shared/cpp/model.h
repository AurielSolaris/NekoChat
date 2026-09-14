// Architecture-independent model interface. Each architecture (gpt2/, qwen3/) implements Model
// for CPU and GPU; the engine only sees this interface.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gpu/compute.h"
#include "thread_pool.h"

namespace neko {

enum class BackendPref : int { Auto = 0, Vulkan = 1, OpenGL = 2, CPU = 3 };

using ProgressFn = std::function<void(float fraction, const std::string& stage)>;

class TensorStore;

struct ModelShape {
    int vocab = 0;
    int context = 0;  // usable context (may be below the model's trained maximum to bound KV memory)
    int layers = 0;
    int embd = 0;
    int heads = 0;
    int kvHeads = 0;
    int headDim = 0;
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
    // Maps every checkpoint file in dir (all *.safetensors shards, else the first .pt/.pth/.bin).
    void openCheckpoint(const std::string& dir, TensorStore& ts);
    void checkBatch(const int* tokens, int T, int pos0) const;

    // Backend selection: preferred GPU API first, then the other one, else CPU. A GPU is only used after
    // compatible() and the kernel self-test pass; load() uploads the weights and may throw, in which case
    // unload() drops partial state and the next option is tried. Reasons end up in backendNote().
    void startGpu(BackendPref pref, const ProgressFn& progress,
                  const std::function<bool(gpu::ComputeBackend&, std::string&)>& compatible,
                  const std::function<void()>& load, const std::function<void()>& unload);

    ModelShape shape_;
    int64_t params_ = 0;
    std::string note_;
    std::string format_;
    std::unique_ptr<ThreadPool> pool_;
    std::unique_ptr<gpu::ComputeBackend> gpu_;
    int debugLayers_ = -1;
    int debugStage_ = 0;
};

// config.json "model_type" ("gpt2", "qwen3", ...).
std::string readModelType(const std::string& modelDir);

std::unique_ptr<Model> loadModel(const std::string& modelType, const std::string& modelDir, BackendPref pref,
                                 int threads, const ProgressFn& progress);

}  // namespace neko
