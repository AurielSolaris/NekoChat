#include "model.h"

#include <dirent.h>

#include <algorithm>

#include "common.h"
#include "gpt2_model.h"
#include "json.h"
#include "qwen3_model.h"
#include "tensor_store.h"

namespace neko {

std::string Model::deviceName() const {
    if (gpu_) return gpu_->deviceName();
    return "NEON x" + std::to_string(pool_->size());
}

void Model::openCheckpoint(const std::string& dir, TensorStore& ts) {
    std::vector<std::string> st, pt;
    DIR* d = opendir(dir.c_str());
    if (!d) fail("cannot open model folder " + dir);
    while (dirent* e = readdir(d)) {
        std::string n = e->d_name;
        if (ends_with(n, ".safetensors")) st.push_back(dir + "/" + n);
        else if (ends_with(n, ".pt") || ends_with(n, ".pth") || ends_with(n, ".bin")) pt.push_back(dir + "/" + n);
    }
    closedir(d);
    std::sort(st.begin(), st.end());
    std::sort(pt.begin(), pt.end());
    if (st.empty() && pt.empty()) fail("no .safetensors or .pt checkpoint in " + dir);
    // Prefer safetensors (zero-copy, no pickle); PyTorch checkpoints are single files.
    std::vector<std::string> files = st.empty() ? std::vector<std::string>{pt.front()} : st;
    format_ = st.empty() ? "pytorch" : "safetensors";
    for (auto& f : files) ts.addFile(f);
}

void Model::checkBatch(const int* tokens, int T, int pos0) const {
    if (T <= 0 || T > kMaxBatch) fail("forward: bad batch size");
    if (pos0 + T > shape_.context) fail("forward: context overflow");
    for (int t = 0; t < T; t++)
        if (tokens[t] < 0 || tokens[t] >= shape_.vocab) fail("forward: token id out of range");
}

void Model::startGpu(BackendPref pref, const ProgressFn& progress,
                     const std::function<bool(gpu::ComputeBackend&, std::string&)>& compatible,
                     const std::function<void()>& load, const std::function<void()>& unload) {
    std::vector<BackendPref> order;
    switch (pref) {
        case BackendPref::Vulkan: case BackendPref::Auto: order = {BackendPref::Vulkan, BackendPref::OpenGL}; break;
        case BackendPref::OpenGL: order = {BackendPref::OpenGL, BackendPref::Vulkan}; break;
        case BackendPref::CPU: break;
    }
    for (BackendPref b : order) {
        const bool vk = b == BackendPref::Vulkan;
        const char* label = vk ? "Vulkan" : "OpenGL ES";
        std::string err;
        progress(0.04f, std::string("Starting ") + label);
        std::unique_ptr<gpu::ComputeBackend> be = vk ? gpu::createVulkan(err) : gpu::createGles(err);
        if (be && !compatible(*be, err)) be.reset();
        if (be && !gpu::selfTest(*be, err)) be.reset();
        if (!be) {
            note_ += std::string(label) + ": " + err + "\n";
            NEKO_LOGW("%s unavailable: %s", label, err.c_str());
            continue;
        }
        // Self-test buffers are tiny; recreate the backend so the model starts from a clean allocation state.
        be.reset();
        be = vk ? gpu::createVulkan(err) : gpu::createGles(err);
        if (!be) continue;
        gpu_ = std::move(be);
        try {
            load();
            return;
        } catch (const std::exception& e) {
            note_ += std::string(label) + ": " + e.what() + "\n";
            NEKO_LOGW("%s load failed: %s", label, e.what());
            unload();
            gpu_.reset();
        }
    }
}

std::string readModelType(const std::string& dir) {
    Json c = Json::parseFile(dir + "/config.json");
    std::string type = c.getStr("model_type", "");
    if (!type.empty()) return type;
    for (auto& a : c["architectures"].items()) {
        if (starts_with(a.str(), "GPT2")) return "gpt2";
        if (starts_with(a.str(), "Qwen3")) return "qwen3";
    }
    return "unknown";
}

std::unique_ptr<Model> loadModel(const std::string& type, const std::string& dir, BackendPref pref, int threads,
                                 const ProgressFn& progress) {
    if (type == "gpt2") return std::make_unique<Gpt2Model>(dir, pref, threads, progress);
    if (type == "qwen3") return std::make_unique<Qwen3Model>(dir, pref, threads, progress);
    fail("unsupported architecture '" + type + "' (NekoChat runs GPT-2 and Qwen3 models)");
}

}  // namespace neko
