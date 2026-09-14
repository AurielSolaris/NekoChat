// GPU compute abstraction implemented by the Vulkan and OpenGL ES 3.1 backends.
// Both run the same GLSL kernels (shaders/*.comp); Vulkan gets SPIR-V compiled at build time,
// GLES compiles the embedded source at runtime.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace neko::gpu {

// Order must match NEKO_KERNELS in architecture/CMakeLists.txt.
enum class Kernel : int { Embed = 0, LayerNorm, Matmul, KvStore, Attention, Rope, SiluMul, Count };

struct Buffer {
    size_t size = 0;
    virtual ~Buffer() = default;
};

class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;

    virtual const char* name() const = 0;
    virtual std::string deviceName() const = 0;
    virtual size_t maxBufferBytes() const = 0;

    // Buffers are owned by the backend and live until it is destroyed.
    // readback hints that the host will read the buffer every step (cached memory is preferred).
    virtual Buffer* create(size_t bytes, const void* init = nullptr, bool readback = false) = 0;
    virtual void upload(Buffer* b, size_t offset, const void* src, size_t bytes) = 0;
    virtual void download(Buffer* b, size_t offset, void* dst, size_t bytes) = 0;

    // Recording: begin(), any number of dispatch() (each one is ordered after the previous), submitAndWait().
    virtual void begin() = 0;
    virtual void dispatch(Kernel k, Buffer* const bindings[4], const int32_t params[8], uint32_t gx, uint32_t gy) = 0;
    virtual void submitAndWait() = 0;
};

std::unique_ptr<ComputeBackend> createVulkan(std::string& error);
std::unique_ptr<ComputeBackend> createGles(std::string& error);

// Runs every kernel on small random inputs and compares with the CPU reference.
bool selfTest(ComputeBackend& backend, std::string& error);

inline int32_t floatBits(float f) {
    int32_t i;
    static_assert(sizeof i == sizeof f, "");
    __builtin_memcpy(&i, &f, 4);
    return i;
}

}  // namespace neko::gpu
