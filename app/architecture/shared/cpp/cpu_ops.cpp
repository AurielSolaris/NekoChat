#include "cpu_ops.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "common.h"

#if defined(__aarch64__)
#include <arm_neon.h>
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
