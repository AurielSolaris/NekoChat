// NekoChat shared native core: logging, fp16 helpers, small utilities.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <stdexcept>

#if defined(__ANDROID__)
#include <android/log.h>
#define NEKO_LOGI(...) __android_log_print(ANDROID_LOG_INFO, "NekoChat", __VA_ARGS__)
#define NEKO_LOGW(...) __android_log_print(ANDROID_LOG_WARN, "NekoChat", __VA_ARGS__)
#define NEKO_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "NekoChat", __VA_ARGS__)
#else
#include <cstdio>
#define NEKO_LOGI(...) (std::fprintf(stderr, "[I] " __VA_ARGS__), std::fputc('\n', stderr))
#define NEKO_LOGW(...) (std::fprintf(stderr, "[W] " __VA_ARGS__), std::fputc('\n', stderr))
#define NEKO_LOGE(...) (std::fprintf(stderr, "[E] " __VA_ARGS__), std::fputc('\n', stderr))
#endif

namespace neko {

// All loader/engine failures are reported as NekoError and surfaced to Kotlin as a message.
struct NekoError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

[[noreturn]] inline void fail(const std::string& msg) { throw NekoError(msg); }

inline float f16_to_f32(uint16_t h) {
    uint32_t sign = uint32_t(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {  // subnormal: normalise
            exp = 127 - 15 + 1;
            while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
            mant &= 0x3ff;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// Round-to-nearest-even float -> half.
inline uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = int32_t((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffff;
    if (((x >> 23) & 0xff) == 0xff) return uint16_t(sign | 0x7c00 | (mant ? 0x200 : 0));
    if (exp >= 31) return uint16_t(sign | 0x7c00);
    if (exp <= 0) {
        if (exp < -10) return uint16_t(sign);
        mant |= 0x800000;
        uint32_t shift = uint32_t(14 - exp);
        uint32_t half = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1);
        uint32_t mid = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (half & 1))) half++;
        return uint16_t(sign | half);
    }
    uint32_t half = sign | (uint32_t(exp) << 10) | (mant >> 13);
    uint32_t rem = mant & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (half & 1))) half++;
    return uint16_t(half);
}

inline float bf16_to_f32(uint16_t b) {
    uint32_t bits = uint32_t(b) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// Opens a file read-only. Model folders picked through Android's Storage Access Framework are
// exposed as symlinks to "/proc/self/fd/N" (descriptors opened by the Kotlin side); for those the
// existing descriptor is dup'ed instead of re-opening the path, which scoped storage would deny.
int openReadOnly(const std::string& path);

inline bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline bool starts_with(const std::string& s, const std::string& prefix) {
    return s.compare(0, prefix.size(), prefix) == 0;
}

}  // namespace neko
