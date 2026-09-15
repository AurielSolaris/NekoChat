// CPU (NEON) transformer kernels. Weights are [N][K] (f16 or FP8/FP4 blocks, see quant.h), activations f32,
// KV caches f16.
#pragma once

#include <cstdint>

#include "quant.h"
#include "thread_pool.h"

namespace neko::cpu {

enum MatmulFlags { kGelu = 1, kAccumulate = 2 };

// y[t][n] (+)= act(x[t] . W[n] + bias[n]) for t < T. bias may be null. N is the row stride of y.
void matmulF16(ThreadPool& pool, const float* x, int T, int K, const uint16_t* W, const float* bias, float* y, int N,
               int flags);

// Same for any weight format; W is [N][K].
void matmul(ThreadPool& pool, const float* x, int T, int K, const QMatrix& W, const float* bias, float* y, int N,
            int flags);

void layerNorm(const float* x, float* y, const float* gamma, const float* beta, int T, int C, float eps);

// y = x * rsqrt(mean(x^2) + eps) * w for each of `rows` rows of C floats. y may alias x.
void rmsNorm(const float* x, float* y, const float* w, int rows, int C, float eps);

// NeoX-style rotary embedding on nHeads consecutive heads of hd floats. cs holds hd/2 (cos, sin) pairs.
void rope(float* x, int nHeads, int hd, const float* cs);

// SwiGLU: out[t][i] = silu(h[t][i]) * h[t][I + i] (each h row holds the gate then the up projection).
void siluMul(const float* h, float* out, int T, int I);

// Writes the keys/values of T tokens (rows of srcStride floats) into f16 caches [ctx][width] at pos0.
void storeKv(const float* k, const float* v, int srcStride, uint16_t* kCache, uint16_t* vCache, int width, int T,
             int pos0);

// Causal grouped-query attention for T new tokens at pos0 over caches that already hold positions < pos0 + T.
// q rows have stride qStride (nHead heads of hd); caches are [ctx][nKv * hd]; out is [T][nHead * hd].
void attention(ThreadPool& pool, const float* q, int qStride, const uint16_t* kCache, const uint16_t* vCache,
               float* out, int T, int nHead, int nKv, int hd, int pos0);

float gelu(float x);

}  // namespace neko::cpu
