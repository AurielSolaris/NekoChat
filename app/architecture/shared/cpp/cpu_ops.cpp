#include "cpu_ops.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "common.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

namespace neko::cpu {

float gelu(float x) {
    // GPT-2 "gelu_new" (tanh approximation).
    return 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}

namespace {

// Computes dot products of one f16 weight row against up to 4 activation rows.
inline void dotRow4(const uint16_t* w, const float* x0, const float* x1, const float* x2, const float* x3, int nt, int K,
                    float out[4]) {
    int k = 0;
#if defined(__aarch64__)
    float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0), a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
    float32x4_t b0 = vdupq_n_f32(0), b1 = vdupq_n_f32(0), b2 = vdupq_n_f32(0), b3 = vdupq_n_f32(0);
    for (; k + 8 <= K; k += 8) {
        float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(w + k));
        float32x4_t wl = vcvt_f32_f16(vget_low_f16(h));
        float32x4_t wh = vcvt_high_f32_f16(h);
        a0 = vfmaq_f32(a0, wl, vld1q_f32(x0 + k));
        b0 = vfmaq_f32(b0, wh, vld1q_f32(x0 + k + 4));
        if (nt > 1) {
            a1 = vfmaq_f32(a1, wl, vld1q_f32(x1 + k));
            b1 = vfmaq_f32(b1, wh, vld1q_f32(x1 + k + 4));
        }
        if (nt > 2) {
            a2 = vfmaq_f32(a2, wl, vld1q_f32(x2 + k));
            b2 = vfmaq_f32(b2, wh, vld1q_f32(x2 + k + 4));
        }
        if (nt > 3) {
            a3 = vfmaq_f32(a3, wl, vld1q_f32(x3 + k));
            b3 = vfmaq_f32(b3, wh, vld1q_f32(x3 + k + 4));
        }
    }
    out[0] = vaddvq_f32(vaddq_f32(a0, b0));
    out[1] = vaddvq_f32(vaddq_f32(a1, b1));
    out[2] = vaddvq_f32(vaddq_f32(a2, b2));
    out[3] = vaddvq_f32(vaddq_f32(a3, b3));
#else
    out[0] = out[1] = out[2] = out[3] = 0;
#endif
    const float* xs[4] = {x0, x1, x2, x3};
    for (; k < K; k++) {
        float wf = f16_to_f32(w[k]);
        for (int t = 0; t < nt; t++) out[t] += wf * xs[t][k];
    }
}

// sum(a[d] * b[d]) with b in f16.
inline float dotF16(const float* a, const uint16_t* b, int n) {
    int d = 0;
    float s = 0;
#if defined(__aarch64__)
    float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0);
    for (; d + 8 <= n; d += 8) {
        float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(b + d));
        s0 = vfmaq_f32(s0, vld1q_f32(a + d), vcvt_f32_f16(vget_low_f16(h)));
        s1 = vfmaq_f32(s1, vld1q_f32(a + d + 4), vcvt_high_f32_f16(h));
    }
    s = vaddvq_f32(vaddq_f32(s0, s1));
#endif
    for (; d < n; d++) s += a[d] * f16_to_f32(b[d]);
    return s;
}

// acc[d] += p * v[d] with v in f16.
inline void axpyF16(float p, const uint16_t* v, float* acc, int n) {
    int d = 0;
#if defined(__aarch64__)
    float32x4_t pv = vdupq_n_f32(p);
    for (; d + 8 <= n; d += 8) {
        float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(v + d));
        vst1q_f32(acc + d, vfmaq_f32(vld1q_f32(acc + d), pv, vcvt_f32_f16(vget_low_f16(h))));
        vst1q_f32(acc + d + 4, vfmaq_f32(vld1q_f32(acc + d + 4), pv, vcvt_high_f32_f16(h)));
    }
#endif
    for (; d < n; d++) acc[d] += p * f16_to_f32(v[d]);
}

inline void toHalf(const float* src, uint16_t* dst, int n) {
    int i = 0;
#if defined(__aarch64__)
    for (; i + 4 <= n; i += 4) vst1_u16(dst + i, vreinterpret_u16_f16(vcvt_f16_f32(vld1q_f32(src + i))));
#endif
    for (; i < n; i++) dst[i] = f32_to_f16(src[i]);
}

}  // namespace

void matmulF16(ThreadPool& pool, const float* x, int T, int K, const uint16_t* W, const float* bias, float* y, int N,
               int flags) {
    pool.parallelFor(N, [&](int n0, int n1, int) {
        for (int n = n0; n < n1; n++) {
            const uint16_t* w = W + size_t(n) * size_t(K);
            float b = bias ? bias[n] : 0.0f;
            for (int t0 = 0; t0 < T; t0 += 4) {
                int nt = std::min(4, T - t0);
                const float* xr = x + size_t(t0) * size_t(K);
                float acc[4];
                dotRow4(w, xr, xr + (nt > 1 ? K : 0), xr + (nt > 2 ? 2 * K : 0), xr + (nt > 3 ? 3 * K : 0), nt, K, acc);
                for (int j = 0; j < nt; j++) {
                    float v = acc[j] + b;
                    if (flags & kGelu) v = gelu(v);
                    float& dst = y[size_t(t0 + j) * size_t(N) + size_t(n)];
                    dst = (flags & kAccumulate) ? dst + v : v;
                }
            }
        }
    });
}

namespace {

#if defined(__aarch64__)
inline float halfValue(uint16_t b) {
    __fp16 h;
    std::memcpy(&h, &b, 2);
    return static_cast<float>(h);
}

inline void finishRow(const float* sums, int tn, int tc, int n, const float* bias, float* y, int N, int flags) {
    const float b = bias ? bias[n] : 0.0f;
    for (int t = 0; t < tn; t++) {
        float v = sums[t] + b;
        if (flags & kGelu) v = gelu(v);
        float& dst = y[size_t(tc + t) * size_t(N) + size_t(n)];
        dst = (flags & kAccumulate) ? dst + v : v;
    }
}

// 16 FP8 (E4M3) codes -> 16 floats (value / 256). The f16 pattern sign << 15 | (code & 0x7F) << 7 has the
// high byte sign | (code & 0x7E) >> 1 and the low byte code << 7; a zip interleaves them into halves.
inline void fp8x16(const uint8_t* p, float32x4_t w[4]) {
    uint8x16_t q = vld1q_u8(p);
    uint8x16_t hi = vbslq_u8(vdupq_n_u8(0x80), q, vshrq_n_u8(vshlq_n_u8(q, 1), 2));
    uint8x16_t lo = vshlq_n_u8(q, 7);
    float16x8_t a = vreinterpretq_f16_u8(vzip1q_u8(lo, hi)), b = vreinterpretq_f16_u8(vzip2q_u8(lo, hi));
    w[0] = vcvt_f32_f16(vget_low_f16(a));
    w[1] = vcvt_high_f32_f16(a);
    w[2] = vcvt_f32_f16(vget_low_f16(b));
    w[3] = vcvt_high_f32_f16(b);
}

inline float32x4_t dotBlock(const float32x4_t w[8], const float* xt) {
    float32x4_t p = vmulq_f32(w[0], vld1q_f32(xt));
    float32x4_t q = vmulq_f32(w[1], vld1q_f32(xt + 4));
    p = vfmaq_f32(p, w[2], vld1q_f32(xt + 8));
    q = vfmaq_f32(q, w[3], vld1q_f32(xt + 12));
    p = vfmaq_f32(p, w[4], vld1q_f32(xt + 16));
    q = vfmaq_f32(q, w[5], vld1q_f32(xt + 20));
    p = vfmaq_f32(p, w[6], vld1q_f32(xt + 24));
    q = vfmaq_f32(q, w[7], vld1q_f32(xt + 28));
    return vaddq_f32(p, q);
}

// FP8: every block is decoded once per row and applied to all T tokens (prefill does not decode a row once per
// token); decode (T = 1) keeps its accumulator in a register.
void matmulFp8(ThreadPool& pool, const float* x, int T, int K, const QMatrix& W, const float* bias, float* y, int N,
               int flags) {
    constexpr int kTile = 64;
    const int nb = K / kQuantBlock;
    pool.parallelFor(W.rows, [&](int n0, int n1, int) {
        float32x4_t acc[kTile];
        float sums[kTile];
        for (int n = n0; n < n1; n++) {
            const uint8_t* row = W.row(n);
            const uint16_t* scales = reinterpret_cast<const uint16_t*>(row + size_t(K));
            if (T == 1) {
                float32x4_t a = vdupq_n_f32(0);
                for (int blk = 0; blk < nb; blk++) {
                    float32x4_t w[8];
                    fp8x16(row + size_t(blk) * kQuantBlock, w);
                    fp8x16(row + size_t(blk) * kQuantBlock + 16, w + 4);
                    a = vfmaq_n_f32(a, dotBlock(w, x + size_t(blk) * kQuantBlock), halfValue(scales[blk]));
                }
                sums[0] = vaddvq_f32(a) * kFp8Factor;
                finishRow(sums, 1, 0, n, bias, y, N, flags);
                continue;
            }
            for (int tc = 0; tc < T; tc += kTile) {
                const int tn = std::min(kTile, T - tc);
                for (int t = 0; t < tn; t++) acc[t] = vdupq_n_f32(0);
                for (int blk = 0; blk < nb; blk++) {
                    float32x4_t w[8];
                    fp8x16(row + size_t(blk) * kQuantBlock, w);
                    fp8x16(row + size_t(blk) * kQuantBlock + 16, w + 4);
                    const float s = halfValue(scales[blk]);
                    const float* xb = x + size_t(tc) * size_t(K) + size_t(blk) * kQuantBlock;
                    for (int t = 0; t < tn; t++) acc[t] = vfmaq_n_f32(acc[t], dotBlock(w, xb + size_t(t) * size_t(K)), s);
                }
                for (int t = 0; t < tn; t++) sums[t] = vaddvq_f32(acc[t]) * kFp8Factor;
                finishRow(sums, tn, tc, n, bias, y, N, flags);
            }
        }
    });
}

// FP4 runs in integers: activations are quantized per 32-block to int8 (scale = max / 127), codes map to twice
// their value ({0, ±1, ±2, ±3, ±4, ±6, ±8, ±12}), and each block is one int8 dot product scaled by
// weightScale / 2 * activationScale. The dot product uses SDOT where the core has it (ARMv8.2 dotprod).
struct Int8Rows {
    std::vector<int8_t> q;
    std::vector<float> s;
};

void quantizeActivations(const float* x, int T, int K, Int8Rows& out) {
    const int nb = K / kQuantBlock;
    out.q.resize(size_t(T) * size_t(K));
    out.s.resize(size_t(T) * size_t(nb));
    for (int t = 0; t < T; t++) {
        for (int b = 0; b < nb; b++) {
            const float* xb = x + size_t(t) * size_t(K) + size_t(b) * kQuantBlock;
            float32x4_t v[8];
            float32x4_t m = vdupq_n_f32(0);
            for (int i = 0; i < 8; i++) {
                v[i] = vld1q_f32(xb + 4 * i);
                m = vmaxq_f32(m, vabsq_f32(v[i]));
            }
            const float amax = vmaxvq_f32(m);
            const float s = amax / 127.0f;
            const float inv = s > 0 ? 1.0f / s : 0.0f;
            out.s[size_t(t) * size_t(nb) + size_t(b)] = s;
            int8_t* qb = out.q.data() + size_t(t) * size_t(K) + size_t(b) * kQuantBlock;
            for (int i = 0; i < 8; i += 2) {
                int32x4_t a = vcvtnq_s32_f32(vmulq_n_f32(v[i], inv));
                int32x4_t c = vcvtnq_s32_f32(vmulq_n_f32(v[i + 1], inv));
                int16x8_t h = vcombine_s16(vqmovn_s32(a), vqmovn_s32(c));
                vst1_s8(qb + 4 * i, vqmovn_s16(h));
            }
        }
    }
}

const int8_t kFp4Twice[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

inline int32x4_t dotI8Mull(int8x16_t w0, int8x16_t x0, int8x16_t w1, int8x16_t x1) {
    // |products| <= 12 * 127, so two of them still fit in int16.
    int16x8_t p0 = vmlal_high_s8(vmull_s8(vget_low_s8(w0), vget_low_s8(x0)), w0, x0);
    int16x8_t p1 = vmlal_high_s8(vmull_s8(vget_low_s8(w1), vget_low_s8(x1)), w1, x1);
    return vpadalq_s16(vpaddlq_s16(p0), p1);
}

__attribute__((target("dotprod"))) inline int32x4_t dotI8Sdot(int8x16_t w0, int8x16_t x0, int8x16_t w1,
                                                              int8x16_t x1) {
    return vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0, x0), w1, x1);
}

template <bool kSdot>
__attribute__((target("dotprod"))) void matmulFp4Rows(const Int8Rows& xq, int T, int K, const QMatrix& W,
                                                      const float* bias, float* y, int N, int flags, int n0, int n1) {
    constexpr int kTile = 64;
    const int nb = K / kQuantBlock;
    const int8x16_t table = vld1q_s8(kFp4Twice);
    const uint8x16_t low = vdupq_n_u8(0x0F);
    float32x4_t acc[kTile];
    float sums[kTile];
    for (int n = n0; n < n1; n++) {
        const uint8_t* row = W.row(n);
        const uint16_t* scales = reinterpret_cast<const uint16_t*>(row + size_t(K) / 2);
        for (int tc = 0; tc < T; tc += kTile) {
            const int tn = std::min(kTile, T - tc);
            for (int t = 0; t < tn; t++) acc[t] = vdupq_n_f32(0);
            for (int blk = 0; blk < nb; blk++) {
                uint8x16_t c = vld1q_u8(row + size_t(blk) * (kQuantBlock / 2));
                int8x16_t w0 = vqtbl1q_s8(table, vandq_u8(c, low));  // weights 0..15
                int8x16_t w1 = vqtbl1q_s8(table, vshrq_n_u8(c, 4));  // weights 16..31
                const float ws = halfValue(scales[blk]) * 0.5f;
                for (int t = 0; t < tn; t++) {
                    const size_t xi = size_t(tc + t) * size_t(K) + size_t(blk) * kQuantBlock;
                    int8x16_t x0 = vld1q_s8(xq.q.data() + xi), x1 = vld1q_s8(xq.q.data() + xi + 16);
                    int32x4_t d = kSdot ? dotI8Sdot(w0, x0, w1, x1) : dotI8Mull(w0, x0, w1, x1);
                    acc[t] = vfmaq_n_f32(acc[t], vcvtq_f32_s32(d), ws * xq.s[size_t(tc + t) * size_t(nb) + size_t(blk)]);
                }
            }
            for (int t = 0; t < tn; t++) sums[t] = vaddvq_f32(acc[t]);
            finishRow(sums, tn, tc, n, bias, y, N, flags);
        }
    }
}

bool hasDotProd() {
    static const bool has = (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
    return has;
}

void matmulFp4(ThreadPool& pool, const float* x, int T, int K, const QMatrix& W, const float* bias, float* y, int N,
               int flags) {
    static thread_local Int8Rows buffer;
    Int8Rows& xq = buffer;  // named reference: a lambda would otherwise see each worker's own thread_local
    quantizeActivations(x, T, K, xq);
    const bool sdot = hasDotProd();
    pool.parallelFor(W.rows, [&](int n0, int n1, int) {
        if (sdot) matmulFp4Rows<true>(xq, T, K, W, bias, y, N, flags, n0, n1);
        else matmulFp4Rows<false>(xq, T, K, W, bias, y, N, flags, n0, n1);
    });
}
#endif

// Portable fallback: dequantize each row, then plain dot products.
void matmulGeneric(ThreadPool& pool, const float* x, int T, int K, const QMatrix& W, const float* bias, float* y, int N,
                   int flags) {
    pool.parallelFor(W.rows, [&](int n0, int n1, int) {
        std::vector<float> w(static_cast<size_t>(K));
        for (int n = n0; n < n1; n++) {
            W.dequantizeRow(n, w.data());
            for (int t = 0; t < T; t++) {
                const float* xt = x + size_t(t) * size_t(K);
                float v = bias ? bias[n] : 0.0f;
                for (int k = 0; k < K; k++) v += xt[k] * w[size_t(k)];
                if (flags & kGelu) v = gelu(v);
                float& dst = y[size_t(t) * size_t(N) + size_t(n)];
                dst = (flags & kAccumulate) ? dst + v : v;
            }
        }
    });
}

}  // namespace

void matmul(ThreadPool& pool, const float* x, int T, int K, const QMatrix& W, const float* bias, float* y, int N,
            int flags) {
    if (W.cols != K) fail("matmul: weight width mismatch");
    switch (W.format) {
        case WeightFormat::F16: matmulF16(pool, x, T, K, W.f16(), bias, y, N, flags); return;
#if defined(__aarch64__)
        case WeightFormat::FP8: matmulFp8(pool, x, T, K, W, bias, y, N, flags); return;
        case WeightFormat::FP4: matmulFp4(pool, x, T, K, W, bias, y, N, flags); return;
#else
        default: matmulGeneric(pool, x, T, K, W, bias, y, N, flags); return;
#endif
    }
}

void layerNorm(const float* x, float* y, const float* g, const float* b, int T, int C, float eps) {
    for (int t = 0; t < T; t++) {
        const float* xr = x + size_t(t) * size_t(C);
        float* yr = y + size_t(t) * size_t(C);
        double mean = 0;
        for (int c = 0; c < C; c++) mean += xr[c];
        mean /= C;
        double var = 0;
        for (int c = 0; c < C; c++) {
            double d = xr[c] - mean;
            var += d * d;
        }
        var /= C;
        float r = float(1.0 / std::sqrt(var + eps));
        float m = float(mean);
        for (int c = 0; c < C; c++) yr[c] = (xr[c] - m) * r * g[c] + b[c];
    }
}

void rmsNorm(const float* x, float* y, const float* w, int rows, int C, float eps) {
    for (int r = 0; r < rows; r++) {
        const float* xr = x + size_t(r) * size_t(C);
        float* yr = y + size_t(r) * size_t(C);
        double ss = 0;
        for (int c = 0; c < C; c++) ss += double(xr[c]) * xr[c];
        float s = float(1.0 / std::sqrt(ss / C + eps));
        for (int c = 0; c < C; c++) yr[c] = xr[c] * s * w[c];
    }
}

void rope(float* x, int nHeads, int hd, const float* cs) {
    const int half = hd / 2;
    for (int h = 0; h < nHeads; h++) {
        float* v = x + size_t(h) * size_t(hd);
        for (int i = 0; i < half; i++) {
            float c = cs[2 * i], s = cs[2 * i + 1];
            float a = v[i], b = v[i + half];
            v[i] = a * c - b * s;
            v[i + half] = a * s + b * c;
        }
    }
}

void siluMul(const float* h, float* out, int T, int I) {
    for (int t = 0; t < T; t++) {
        const float* gate = h + size_t(t) * 2 * size_t(I);
        const float* up = gate + I;
        float* o = out + size_t(t) * size_t(I);
        for (int i = 0; i < I; i++) o[i] = gate[i] / (1.0f + std::exp(-gate[i])) * up[i];
    }
}

void storeKv(const float* k, const float* v, int srcStride, uint16_t* kCache, uint16_t* vCache, int width, int T,
             int pos0) {
    for (int t = 0; t < T; t++) {
        size_t dst = size_t(pos0 + t) * size_t(width);
        toHalf(k + size_t(t) * size_t(srcStride), kCache + dst, width);
        toHalf(v + size_t(t) * size_t(srcStride), vCache + dst, width);
    }
}

void attention(ThreadPool& pool, const float* q, int qStride, const uint16_t* kCache, const uint16_t* vCache,
               float* out, int T, int nHead, int nKv, int hd, int pos0) {
    const int kvW = nKv * hd, group = nHead / nKv;
    const float scale = 1.0f / std::sqrt(float(hd));
    pool.parallelFor(nHead * T, [&](int i0, int i1, int) {
        std::vector<float> scores(static_cast<size_t>(pos0 + T));
        std::vector<float> acc(static_cast<size_t>(hd));
        for (int i = i0; i < i1; i++) {
            int h = i % nHead, t = i / nHead;
            int L = pos0 + t + 1;
            const float* qh = q + size_t(t) * size_t(qStride) + size_t(h) * size_t(hd);
            const size_t kvOff = size_t(h / group) * size_t(hd);
            float mx = -INFINITY;
            for (int j = 0; j < L; j++) {
                float s = dotF16(qh, kCache + size_t(j) * size_t(kvW) + kvOff, hd) * scale;
                scores[size_t(j)] = s;
                mx = std::max(mx, s);
            }
            float sum = 0;
            for (int j = 0; j < L; j++) {
                float e = std::exp(scores[size_t(j)] - mx);
                scores[size_t(j)] = e;
                sum += e;
            }
            std::fill(acc.begin(), acc.end(), 0.0f);
            for (int j = 0; j < L; j++) axpyF16(scores[size_t(j)], vCache + size_t(j) * size_t(kvW) + kvOff, acc.data(), hd);
            float inv = 1.0f / sum;
            float* o = out + size_t(t) * size_t(nHead) * size_t(hd) + size_t(h) * size_t(hd);
            for (int d = 0; d < hd; d++) o[d] = acc[size_t(d)] * inv;
        }
    });
}

}  // namespace neko::cpu
