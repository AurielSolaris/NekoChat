#include "quant.h"

#include <algorithm>
#include <cmath>

#include "common.h"
#include "thread_pool.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace neko {

namespace {

inline uint16_t halfBits(float f) {
#if defined(__aarch64__)
    __fp16 h = static_cast<__fp16>(f);
    uint16_t b;
    std::memcpy(&b, &h, 2);
    return b;
#else
    return f32_to_f16(f);
#endif
}

inline float halfValue(uint16_t b) {
#if defined(__aarch64__)
    __fp16 h;
    std::memcpy(&h, &b, 2);
    return static_cast<float>(h);
#else
    return f16_to_f32(b);
#endif
}

// E4M3 is an f16 with the exponent rebased by 8 and 3 mantissa bits: code c decodes as the f16 bit pattern
// (sign << 15 | (c & 0x7F) << 7) times 256, subnormals included. Encoding goes the same way back.
inline uint8_t encodeFp8(float y) {
    uint16_t h = halfBits(y * (1.0f / kFp8Factor));
    uint32_t mag = h & 0x7FFFu;
    uint32_t c = (mag + 0x3Fu + ((mag >> 7) & 1u)) >> 7;  // round to nearest even over the 7 dropped bits
    if (c > 0x7Eu) c = 0x7Eu;                               // 448 (0x7F is NaN)
    return uint8_t(c | ((h >> 8) & 0x80u));
}

inline float decodeFp8(uint8_t c) {
    return halfValue(uint16_t(((c & 0x80u) << 8) | ((c & 0x7Fu) << 7))) * kFp8Factor;
}

// E2M1 magnitudes by code: 0, 0.5, 1, 1.5, 2, 3, 4, 6. As f16 bits (sign << 15 | (c & 7) << 9) they decode to
// value / 16384, subnormal 0.5 included.
inline uint8_t encodeFp4(float y) {
    float a = std::fabs(y);
    int c = (a > 0.25f) + (a > 0.75f) + (a > 1.25f) + (a > 1.75f) + (a > 2.5f) + (a > 3.5f) + (a > 5.0f);
    return uint8_t(c == 0 ? 0 : (c | (y < 0 ? 8 : 0)));
}

inline float decodeFp4(uint8_t c) {
    return halfValue(uint16_t(((c & 8u) << 12) | ((c & 7u) << 9))) * kFp4Factor;
}

constexpr float kFp8Max = 448.0f;
constexpr float kFp4Max = 6.0f;

// Scale so the block's largest magnitude maps to the format's largest code. Returns the f16 scale bits and
// 1 / scale (0 for an all-zero block). The rounded f16 scale is what the kernels use, so codes are chosen for it.
inline uint16_t blockScale(const float* x, float maxCode, float& inv) {
    float amax = 0;
    for (int i = 0; i < kQuantBlock; i++) amax = std::max(amax, std::fabs(x[i]));
    uint16_t s = halfBits(amax / maxCode);
    float sv = halfValue(s);
    inv = sv > 0 ? 1.0f / sv : 0.0f;
    return sv > 0 ? s : 0;
}

void quantizeRowFp8(const float* x, int cols, uint8_t* out) {
    const int nb = cols / kQuantBlock;
    auto* scales = reinterpret_cast<uint16_t*>(out + cols);
    for (int b = 0; b < nb; b++) {
        const float* xb = x + b * kQuantBlock;
        float inv;
        scales[b] = blockScale(xb, kFp8Max, inv);
        for (int i = 0; i < kQuantBlock; i++) out[b * kQuantBlock + i] = encodeFp8(xb[i] * inv);
    }
}

// Nearest E2M1 magnitude of a >= 0: 0, 0.5, 1, 1.5 below 1.75; 2, 3 below 3.5; 4 below 5; else 6.
inline float fp4Round(float a) {
    if (a < 1.75f) return std::nearbyint(a * 2.0f) * 0.5f;
    if (a < 3.5f) return std::nearbyint(a);
    return a < 5.0f ? 4.0f : 6.0f;
}

// Squared error of a block quantized with scale s, plus the sums that give the least-squares scale for the
// codes it picked: s* = sum(|x| r) / sum(r^2).
struct Fp4Fit {
    float err = 0, xr = 0, rr = 0;
};

#if defined(__aarch64__)
inline float32x4_t fp4RoundV(float32x4_t a) {
    float32x4_t lo = vmulq_n_f32(vrndnq_f32(vmulq_n_f32(a, 2.0f)), 0.5f);
    float32x4_t mid = vrndnq_f32(a);
    float32x4_t hi = vbslq_f32(vcltq_f32(a, vdupq_n_f32(5.0f)), vdupq_n_f32(4.0f), vdupq_n_f32(6.0f));
    return vbslq_f32(vcltq_f32(a, vdupq_n_f32(1.75f)), lo, vbslq_f32(vcltq_f32(a, vdupq_n_f32(3.5f)), mid, hi));
}

inline Fp4Fit fitFp4(const float32x4_t ax[8], float s) {
    const float inv = 1.0f / s;
    float32x4_t err = vdupq_n_f32(0), xr = err, rr = err;
    for (int i = 0; i < 8; i++) {
        float32x4_t r = fp4RoundV(vmulq_n_f32(ax[i], inv));
        float32x4_t d = vfmsq_n_f32(ax[i], r, s);
        err = vfmaq_f32(err, d, d);
        xr = vfmaq_f32(xr, ax[i], r);
        rr = vfmaq_f32(rr, r, r);
    }
    return {vaddvq_f32(err), vaddvq_f32(xr), vaddvq_f32(rr)};
}
#else
inline Fp4Fit fitFp4(const float* ax, float s) {
    Fp4Fit f;
    const float inv = 1.0f / s;
    for (int i = 0; i < kQuantBlock; i++) {
        float r = fp4Round(ax[i] * inv), d = ax[i] - r * s;
        f.err += d * d;
        f.xr += ax[i] * r;
        f.rr += r * r;
    }
    return f;
}
#endif

// FP4 has only 7 magnitudes, so the scale matters: starting from max / 6 (nothing clipped), two least-squares
// refits of the scale to the chosen codes usually trade a little clipping of the largest weight for a finer grid
// everywhere else. The candidate with the smallest error wins; scales are compared after f16 rounding.
uint16_t fp4BlockScale(const float* xb) {
    float amax = 0;
    for (int i = 0; i < kQuantBlock; i++) amax = std::max(amax, std::fabs(xb[i]));
    uint16_t best = halfBits(amax / kFp4Max);
    if (halfValue(best) <= 0) return 0;
#if defined(__aarch64__)
    float32x4_t ax[8];
    for (int i = 0; i < 8; i++) ax[i] = vabsq_f32(vld1q_f32(xb + 4 * i));
#else
    float ax[kQuantBlock];
    for (int i = 0; i < kQuantBlock; i++) ax[i] = std::fabs(xb[i]);
#endif
    Fp4Fit fit = fitFp4(ax, halfValue(best));
    float bestErr = fit.err;
    for (int iter = 0; iter < 2 && fit.rr > 0; iter++) {
        uint16_t s = halfBits(fit.xr / fit.rr);
        if (halfValue(s) <= 0 || s == best) break;
        fit = fitFp4(ax, halfValue(s));
        if (fit.err >= bestErr) break;
        bestErr = fit.err;
        best = s;
    }
    return best;
}

void quantizeRowFp4(const float* x, int cols, uint8_t* out) {
    const int nb = cols / kQuantBlock;
    auto* scales = reinterpret_cast<uint16_t*>(out + cols / 2);
    for (int b = 0; b < nb; b++) {
        const float* xb = x + b * kQuantBlock;
        uint16_t s = fp4BlockScale(xb);
        scales[b] = s;
        const float inv = s ? 1.0f / halfValue(s) : 0.0f;
        uint8_t* o = out + b * (kQuantBlock / 2);
        for (int j = 0; j < 16; j++) o[j] = uint8_t(encodeFp4(xb[j] * inv) | (encodeFp4(xb[16 + j] * inv) << 4));
    }
}

void rowToHalf(const float* x, int cols, uint16_t* out) {
    int i = 0;
#if defined(__aarch64__)
    for (; i + 4 <= cols; i += 4) vst1_u16(out + i, vreinterpret_u16_f16(vcvt_f16_f32(vld1q_f32(x + i))));
#endif
    for (; i < cols; i++) out[i] = halfBits(x[i]);
}

}  // namespace

const char* weightFormatName(WeightFormat f) {
    switch (f) {
        case WeightFormat::F16: return "FP16";
        case WeightFormat::FP8: return "FP8";
        case WeightFormat::FP4: return "FP4";
    }
    return "?";
}

size_t quantRowBytes(WeightFormat f, int cols) {
    const size_t c = size_t(cols);
    const size_t scales = (c / kQuantBlock * 2 + 15) / 16 * 16;
    switch (f) {
        case WeightFormat::F16: return c * 2;
        case WeightFormat::FP8: return c + scales;
        case WeightFormat::FP4: return c / 2 + scales;
    }
    return 0;
}

void QMatrix::appendRows(const QMatrix& m) {
    if (empty()) {
        *this = m;
        return;
    }
    if (m.format != format || m.cols != cols) fail("appendRows: format or width mismatch");
    data.insert(data.end(), m.data.begin(), m.data.end());
    rows += m.rows;
}

QMatrix QMatrix::sliceRows(int r0, int n) const {
    QMatrix s;
    s.format = format;
    s.rows = n;
    s.cols = cols;
    s.rowBytes = rowBytes;
    s.data.assign(row(r0), row(r0) + size_t(n) * rowBytes);
    return s;
}

void QMatrix::dequantizeRow(int r, float* out) const {
    const uint8_t* p = row(r);
    switch (format) {
        case WeightFormat::F16: {
            auto* h = reinterpret_cast<const uint16_t*>(p);
            for (int c = 0; c < cols; c++) out[c] = halfValue(h[c]);
            return;
        }
        case WeightFormat::FP8: {
            auto* scales = reinterpret_cast<const uint16_t*>(p + cols);
            for (int c = 0; c < cols; c++) out[c] = decodeFp8(p[c]) * halfValue(scales[c / kQuantBlock]);
            return;
        }
        case WeightFormat::FP4: {
            auto* scales = reinterpret_cast<const uint16_t*>(p + cols / 2);
            for (int c = 0; c < cols; c++) {
                const int b = c / kQuantBlock, j = c % kQuantBlock;
                uint8_t byte = p[b * 16 + (j & 15)];
                out[c] = decodeFp4(j < 16 ? byte & 0xF : byte >> 4) * halfValue(scales[b]);
            }
            return;
        }
    }
}

QMatrix quantizeMatrix(WeightFormat f, int rows, int cols, const std::function<void(int, float*)>& read,
                       ThreadPool* pool) {
    if (cols % kQuantBlock != 0) f = WeightFormat::F16;
    QMatrix m;
    m.format = f;
    m.rows = rows;
    m.cols = cols;
    m.rowBytes = quantRowBytes(f, cols);
    m.data.assign(m.rowBytes * size_t(rows), 0);  // zeroes the scale padding
    auto work = [&](int r0, int r1, int) {
        std::vector<float> tmp(static_cast<size_t>(cols));
        for (int r = r0; r < r1; r++) {
            read(r, tmp.data());
            uint8_t* out = m.data.data() + size_t(r) * m.rowBytes;
            switch (f) {
                case WeightFormat::F16: rowToHalf(tmp.data(), cols, reinterpret_cast<uint16_t*>(out)); break;
                case WeightFormat::FP8: quantizeRowFp8(tmp.data(), cols, out); break;
                case WeightFormat::FP4: quantizeRowFp4(tmp.data(), cols, out); break;
            }
        }
    };
    if (pool) pool->parallelFor(rows, work);
    else work(0, rows, 0);
    return m;
}

}  // namespace neko
