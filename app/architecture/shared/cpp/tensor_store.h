// Read-only view of checkpoint tensors from .safetensors or PyTorch (.pt/.pth/.bin) files.
// Files are mmapped; tensors are exposed as pointers into the mapping (zero copy).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace neko {

enum class DType { F32, F16, BF16, F64, I64, I32, I8, U8, Bool };

size_t dtypeSize(DType t);
const char* dtypeName(DType t);

class MappedFile {
public:
    explicit MappedFile(const std::string& path);
    ~MappedFile();
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }
    const std::string& path() const { return path_; }

private:
    std::string path_;
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
};

struct TensorView {
    DType dtype = DType::F32;
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr;  // contiguous row-major
    int64_t numel() const {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
};

class TensorStore {
public:
    // Loads every checkpoint file of one model folder (sharded safetensors supported).
    void addFile(const std::string& path);

    bool has(const std::string& name) const { return tensors_.count(name) != 0; }
    const TensorView& get(const std::string& name) const;
    const std::unordered_map<std::string, TensorView>& all() const { return tensors_; }

    // Converts any float dtype to f32.
    std::vector<float> toF32(const std::string& name) const;
    // Converts to f16 bits; when transpose2d is set a [R][C] tensor is returned as [C][R].
    std::vector<uint16_t> toF16(const std::string& name, bool transpose2d) const;

    // Drops the mmaps (call after weights were copied into their final layout).
    void release() {
        tensors_.clear();
        files_.clear();
    }

private:
    void loadSafetensors(std::shared_ptr<MappedFile> f);
    void loadTorchZip(std::shared_ptr<MappedFile> f);

    std::vector<std::shared_ptr<MappedFile>> files_;
    std::unordered_map<std::string, TensorView> tensors_;
};

}  // namespace neko
