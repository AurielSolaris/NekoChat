// SPDX-FileCopyrightText: 2026 AurielSolaris
// SPDX-License-Identifier: GPL-3.0-or-later

#include "kokoro_model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>

#include "common.h"
#include "cpu_ops.h"
#include "json.h"
#include "tensor_store.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace neko {

namespace {

constexpr float kPi = 3.14159265358979323846f;

// Activations are time-major: [T][C], one row of C channels per frame.
using Vec = std::vector<float>;

struct Linear {
    QMatrix w;  // [out][in]
    Vec b;
    int in = 0, out = 0;
};

// Style-conditioned affine parameters, computed once per utterance: (1 + gamma) * norm(x) + beta.
struct StyleFc {
    Vec w, b;  // [2C][styleDim], f32 (tiny)
    int channels = 0;
};

// Conv1d. Each kernel tap k is its own [out][in] matrix, so a stride-1 conv is K accumulating matmuls over a
// zero-padded input and needs no im2col copy.
struct Conv {
    std::vector<QMatrix> taps;
    Vec b;
    int in = 0, out = 0, k = 1, stride = 1, pad = 0, dil = 1;
};

// ConvTranspose1d: one [out * K][in] matmul, then overlap-add.
struct ConvT {
    QMatrix w;
    Vec b;
    int in = 0, out = 0, k = 1, stride = 1, pad = 0;
};

// Depthwise ConvTranspose1d (kernel 3, stride 2, padding 1, output_padding 1): doubles the length.
struct DwUp {
    Vec w, b;  // [C][3], [C]
    int c = 0;
};

struct Lstm {
    QMatrix wih[2];  // [4H][in] per direction (PyTorch gate order i, f, g, o)
    Vec whh[2];      // [4H][H], f32: the recurrence feeds its own output back
    Vec bias[2];     // b_ih + b_hh
    int in = 0, h = 0;
};

struct AdainResBlk {  // istftnet.AdainResBlk1d
    StyleFc n1, n2;
    Conv c1, c2, sc;
    bool learnedSc = false, upsample = false;
    DwUp pool;
    int in = 0, out = 0;
};

struct AdaResBlock1 {  // istftnet.AdaINResBlock1 (snake activations)
    Conv c1[3], c2[3];
    StyleFc a1[3], a2[3];
    Vec alpha1[3], alpha2[3];
};

}  // namespace

struct KokoroModel::Impl {
    std::unique_ptr<ThreadPool> pool;
    WeightFormat fmt = WeightFormat::F16;
    size_t bytes = 0;

    // ALBERT
    Vec wordEmb, posEmb, typeEmb, embLnG, embLnB;  // embeddings are 128 wide
    Linear embMap;                                  // 128 -> 768
    Linear q, k, v, dense, ffn, ffnOut;
    Vec attLnG, attLnB, fullLnG, fullLnB;
    Linear bertEncoder;  // 768 -> 512

    // Prosody predictor
    Lstm durLstm[3];
    StyleFc durNorm[3];
    Lstm predLstm, shared;
    Linear durProj;
    AdainResBlk f0[3], nblk[3];
    Conv f0Proj, nProj;

    // Text encoder
    Vec teEmb;  // [178][512]
    Conv teCnn[3];
    Vec teLnG[3], teLnB[3];
    Lstm teLstm;

    // Decoder
    Conv f0Conv, nConv, asrRes;
    AdainResBlk encode, decode[4];

    // Generator
    Vec srcW, srcB;  // m_source.l_linear: 9 -> 1
    Conv noiseConv[2];
    AdaResBlock1 noiseRes[2];
    ConvT ups[2];
    AdaResBlock1 res[6];
    Conv convPost;

    // -------------------------------------------------------------- loading helpers

    QMatrix matrix(int rows, int cols, const float* w) {
        QMatrix m = quantizeMatrix(fmt, rows, cols, [w, cols](int r, float* out) {
            std::copy(w + size_t(r) * size_t(cols), w + size_t(r + 1) * size_t(cols), out);
        }, pool.get());
        bytes += m.bytes();
        return m;
    }

    Vec f32(TensorStore& ts, const std::string& name) {
        Vec v = ts.toF32(name);
        bytes += v.size() * 4;
        return v;
    }

    Linear linear(TensorStore& ts, const std::string& p, bool bias = true) {
        const auto& shape = ts.get(p + ".weight").shape;
        Linear l;
        l.out = int(shape[0]);
        l.in = int(shape[1]);
        Vec w = ts.toF32(p + ".weight");
        l.w = matrix(l.out, l.in, w.data());
        if (bias) l.b = f32(ts, p + ".bias");
        return l;
    }

    StyleFc styleFc(TensorStore& ts, const std::string& p) {
        StyleFc s;
        s.w = f32(ts, p + ".weight");
        s.b = f32(ts, p + ".bias");
        s.channels = int(s.b.size() / 2);
        return s;
    }

    // Weight-normalised weights (weight_g, weight_v) are folded: w = g * v / ||v||, the norm taken over every
    // dimension but the first (per output channel for Conv1d, per input channel for ConvTranspose1d).
    Vec weight(TensorStore& ts, const std::string& p, std::vector<int64_t>& shape) {
        if (!ts.has(p + ".weight_v")) {
            shape = ts.get(p + ".weight").shape;
            return ts.toF32(p + ".weight");
        }
        shape = ts.get(p + ".weight_v").shape;
        Vec v = ts.toF32(p + ".weight_v"), g = ts.toF32(p + ".weight_g");
        const size_t per = v.size() / size_t(shape[0]);
        for (size_t o = 0; o < size_t(shape[0]); o++) {
            double ss = 0;
            for (size_t i = 0; i < per; i++) ss += double(v[o * per + i]) * v[o * per + i];
            const float s = float(g[o] / std::sqrt(ss));
            for (size_t i = 0; i < per; i++) v[o * per + i] *= s;
        }
        return v;
    }

    Conv conv(TensorStore& ts, const std::string& p, int stride, int pad, int dil, bool bias = true) {
        std::vector<int64_t> shape;
        Vec w = weight(ts, p, shape);  // [out][in][K]
        Conv c;
        c.out = int(shape[0]);
        c.in = int(shape[1]);
        c.k = int(shape[2]);
        c.stride = stride;
        c.pad = pad;
        c.dil = dil;
        Vec tap(size_t(c.out) * size_t(c.in));
        for (int t = 0; t < c.k; t++) {
            for (int o = 0; o < c.out; o++)
                for (int i = 0; i < c.in; i++) tap[size_t(o) * c.in + i] = w[(size_t(o) * c.in + i) * c.k + t];
            c.taps.push_back(matrix(c.out, c.in, tap.data()));
        }
        if (bias && ts.has(p + ".bias")) c.b = f32(ts, p + ".bias");
        return c;
    }

    ConvT convT(TensorStore& ts, const std::string& p, int stride, int pad) {
        std::vector<int64_t> shape;
        Vec w = weight(ts, p, shape);  // [in][out][K]
        ConvT c;
        c.in = int(shape[0]);
        c.out = int(shape[1]);
        c.k = int(shape[2]);
        c.stride = stride;
        c.pad = pad;
        Vec m(size_t(c.out) * c.k * c.in);  // row (o, t), column i
        for (int i = 0; i < c.in; i++)
            for (int o = 0; o < c.out; o++)
                for (int t = 0; t < c.k; t++) m[(size_t(o) * c.k + t) * c.in + i] = w[(size_t(i) * c.out + o) * c.k + t];
        c.w = matrix(c.out * c.k, c.in, m.data());
        c.b = f32(ts, p + ".bias");
        return c;
    }

    Lstm lstm(TensorStore& ts, const std::string& p) {
        Lstm l;
        for (int d = 0; d < 2; d++) {
            const std::string s = d ? "_reverse" : "";
            const auto& shape = ts.get(p + ".weight_ih_l0" + s).shape;
            l.h = int(shape[0]) / 4;
            l.in = int(shape[1]);
            Vec wih = ts.toF32(p + ".weight_ih_l0" + s);
            l.wih[d] = matrix(4 * l.h, l.in, wih.data());
            l.whh[d] = f32(ts, p + ".weight_hh_l0" + s);
            Vec bi = ts.toF32(p + ".bias_ih_l0" + s), bh = ts.toF32(p + ".bias_hh_l0" + s);
            for (size_t i = 0; i < bi.size(); i++) bi[i] += bh[i];
            l.bias[d] = std::move(bi);
        }
        return l;
    }

    AdainResBlk adainResBlk(TensorStore& ts, const std::string& p, bool upsample) {
        AdainResBlk b;
        b.upsample = upsample;
        b.c1 = conv(ts, p + ".conv1", 1, 1, 1);
        b.c2 = conv(ts, p + ".conv2", 1, 1, 1);
        b.in = b.c1.in;
        b.out = b.c2.out;
        b.n1 = styleFc(ts, p + ".norm1.fc");
        b.n2 = styleFc(ts, p + ".norm2.fc");
        b.learnedSc = b.in != b.out;
        if (b.learnedSc) b.sc = conv(ts, p + ".conv1x1", 1, 0, 1, false);
        if (upsample) {
            std::vector<int64_t> shape;
            b.pool.w = weight(ts, p + ".pool", shape);  // [C][1][3]
            b.pool.b = f32(ts, p + ".pool.bias");
            b.pool.c = int(shape[0]);
            bytes += b.pool.w.size() * 4;
        }
        return b;
    }

    AdaResBlock1 resBlock1(TensorStore& ts, const std::string& p, int kernel) {
        AdaResBlock1 r;
        const int dil[3] = {1, 3, 5};
        for (int j = 0; j < 3; j++) {
            const std::string s = std::to_string(j);
            r.c1[j] = conv(ts, p + ".convs1." + s, 1, dil[j] * (kernel - 1) / 2, dil[j]);
            r.c2[j] = conv(ts, p + ".convs2." + s, 1, (kernel - 1) / 2, 1);
            r.a1[j] = styleFc(ts, p + ".adain1." + s + ".fc");
            r.a2[j] = styleFc(ts, p + ".adain2." + s + ".fc");
            r.alpha1[j] = f32(ts, p + ".alpha1." + s);
            r.alpha2[j] = f32(ts, p + ".alpha2." + s);
        }
        return r;
    }

    // -------------------------------------------------------------- ops

    // y[T][out] (+)= x[T][in] . W^T (+ b)
    void mm(const float* x, int T, const QMatrix& w, const Vec* b, float* y, bool accumulate) {
        if (T <= 0) return;
        if (T >= 16) {  // enough rows to be worth unpacking the weights into f32 panels
            convGemm(packPanels({w}), 1, w.cols, w.rows, x, T, 0, b, y, accumulate);
            return;
        }
        cpu::matmul(*pool, x, T, w.cols, w, b && !b->empty() ? b->data() : nullptr, y, w.rows,
                    accumulate ? cpu::kAccumulate : 0);
    }

    Vec linearFwd(const Linear& l, const Vec& x, int T) {
        Vec y(size_t(T) * size_t(l.out));
        mm(x.data(), T, l.w, &l.b, y.data(), false);
        return y;
    }

    // ---- Convolution GEMM. Convolutions here run over thousands of frames, so weights are worth unpacking per
    // call: the stored (possibly FP8/FP4) taps are dequantized into f32 panels of 16 outputs,
    // panel[p][tap][k][16], and a 4-frame x 16-output register tile accumulates over every tap and input
    // channel before anything is written.
    static constexpr int kPanel = 16;

    Vec packPanels(const std::vector<QMatrix>& taps) {
        const int nt = int(taps.size()), in = taps[0].cols, out = taps[0].rows;
        const int npan = (out + kPanel - 1) / kPanel;
        Vec panels(size_t(npan) * nt * in * kPanel, 0.0f);
        // Whole panels per thread: splitting by output row would have every thread writing interleaved lanes
        // of the same cache lines.
        pool->parallelFor(npan, [&](int p0, int p1, int) {
            Vec row(static_cast<size_t>(in));
            for (int p = p0; p < p1; p++)
                for (int lane = 0; lane < kPanel && p * kPanel + lane < out; lane++)
                    for (int j = 0; j < nt; j++) {
                        taps[size_t(j)].dequantizeRow(p * kPanel + lane, row.data());
                        float* dst = panels.data() + ((size_t(p) * nt + j) * in) * kPanel + lane;
                        for (int k = 0; k < in; k++) dst[size_t(k) * kPanel] = row[size_t(k)];
                    }
        });
        return panels;
    }

    // y[t][n] (+)= b[n] + sum_j sum_k xp[(t + j*dil)][k] * W_j[n][k] for t < T. xp holds T + (taps-1)*dil rows.
    void convGemm(const Vec& panels, int taps, int in, int out, const float* xp, int T, int dil, const Vec* bias,
                  float* y, bool accumulate) {
        const int npan = (out + kPanel - 1) / kPanel;
        const int blocks = (T + 3) / 4;
        pool->parallelFor(blocks, [&](int b0, int b1, int) {
            for (int p = 0; p < npan; p++) {
                const int n0 = p * kPanel, valid = std::min(kPanel, out - n0);
                const float* wp = panels.data() + size_t(p) * taps * in * kPanel;
                for (int blk = b0; blk < b1; blk++) {
                    const int t0 = blk * 4, nt = std::min(4, T - t0);
                    float acc[4][kPanel];
#if defined(__aarch64__)
                    float32x4_t c[4][4];
                    for (int r = 0; r < 4; r++)
                        for (int q = 0; q < 4; q++) c[r][q] = vdupq_n_f32(0);
                    for (int j = 0; j < taps; j++) {
                        const float* x0 = xp + size_t(t0 + j * dil) * in;
                        const float* x1 = nt > 1 ? x0 + in : x0;
                        const float* x2 = nt > 2 ? x0 + 2 * size_t(in) : x0;
                        const float* x3 = nt > 3 ? x0 + 3 * size_t(in) : x0;
                        const float* w = wp + size_t(j) * in * kPanel;
                        int k = 0;
                        // Four input channels at a time: one vector load per frame, then lane-indexed FMAs.
                        for (; k + 4 <= in; k += 4) {
                            const float32x4_t v0 = vld1q_f32(x0 + k), v1 = vld1q_f32(x1 + k), v2 = vld1q_f32(x2 + k),
                                              v3 = vld1q_f32(x3 + k);
#define NEKO_KSTEP(L)                                                                                            \
    {                                                                                                            \
        const float32x4_t w0 = vld1q_f32(w), w1 = vld1q_f32(w + 4), w2 = vld1q_f32(w + 8), w3 = vld1q_f32(w + 12); \
        c[0][0] = vfmaq_laneq_f32(c[0][0], w0, v0, L); c[0][1] = vfmaq_laneq_f32(c[0][1], w1, v0, L);            \
        c[0][2] = vfmaq_laneq_f32(c[0][2], w2, v0, L); c[0][3] = vfmaq_laneq_f32(c[0][3], w3, v0, L);            \
        c[1][0] = vfmaq_laneq_f32(c[1][0], w0, v1, L); c[1][1] = vfmaq_laneq_f32(c[1][1], w1, v1, L);            \
        c[1][2] = vfmaq_laneq_f32(c[1][2], w2, v1, L); c[1][3] = vfmaq_laneq_f32(c[1][3], w3, v1, L);            \
        c[2][0] = vfmaq_laneq_f32(c[2][0], w0, v2, L); c[2][1] = vfmaq_laneq_f32(c[2][1], w1, v2, L);            \
        c[2][2] = vfmaq_laneq_f32(c[2][2], w2, v2, L); c[2][3] = vfmaq_laneq_f32(c[2][3], w3, v2, L);            \
        c[3][0] = vfmaq_laneq_f32(c[3][0], w0, v3, L); c[3][1] = vfmaq_laneq_f32(c[3][1], w1, v3, L);            \
        c[3][2] = vfmaq_laneq_f32(c[3][2], w2, v3, L); c[3][3] = vfmaq_laneq_f32(c[3][3], w3, v3, L);            \
        w += kPanel;                                                                                             \
    }
                            NEKO_KSTEP(0)
                            NEKO_KSTEP(1)
                            NEKO_KSTEP(2)
                            NEKO_KSTEP(3)
#undef NEKO_KSTEP
                        }
                        for (; k < in; k++, w += kPanel) {
                            const float32x4_t w0 = vld1q_f32(w), w1 = vld1q_f32(w + 4), w2 = vld1q_f32(w + 8),
                                              w3 = vld1q_f32(w + 12);
                            const float a0 = x0[k], a1 = x1[k], a2 = x2[k], a3 = x3[k];
                            c[0][0] = vfmaq_n_f32(c[0][0], w0, a0); c[0][1] = vfmaq_n_f32(c[0][1], w1, a0);
                            c[0][2] = vfmaq_n_f32(c[0][2], w2, a0); c[0][3] = vfmaq_n_f32(c[0][3], w3, a0);
                            c[1][0] = vfmaq_n_f32(c[1][0], w0, a1); c[1][1] = vfmaq_n_f32(c[1][1], w1, a1);
                            c[1][2] = vfmaq_n_f32(c[1][2], w2, a1); c[1][3] = vfmaq_n_f32(c[1][3], w3, a1);
                            c[2][0] = vfmaq_n_f32(c[2][0], w0, a2); c[2][1] = vfmaq_n_f32(c[2][1], w1, a2);
                            c[2][2] = vfmaq_n_f32(c[2][2], w2, a2); c[2][3] = vfmaq_n_f32(c[2][3], w3, a2);
                            c[3][0] = vfmaq_n_f32(c[3][0], w0, a3); c[3][1] = vfmaq_n_f32(c[3][1], w1, a3);
                            c[3][2] = vfmaq_n_f32(c[3][2], w2, a3); c[3][3] = vfmaq_n_f32(c[3][3], w3, a3);
                        }
                    }
                    for (int r = 0; r < 4; r++)
                        for (int q = 0; q < 4; q++) vst1q_f32(acc[r] + 4 * q, c[r][q]);
#else
                    for (int r = 0; r < 4; r++)
                        for (int q = 0; q < kPanel; q++) acc[r][q] = 0;
                    for (int j = 0; j < taps; j++)
                        for (int r = 0; r < nt; r++) {
                            const float* xr = xp + size_t(t0 + r + j * dil) * in;
                            const float* w = wp + size_t(j) * in * kPanel;
                            for (int k = 0; k < in; k++)
                                for (int q = 0; q < kPanel; q++) acc[r][q] += xr[k] * w[size_t(k) * kPanel + q];
                        }
#endif
                    for (int r = 0; r < nt; r++) {
                        float* yr = y + size_t(t0 + r) * out + n0;
                        for (int q = 0; q < valid; q++) {
                            const float v = acc[r][q] + (bias && !bias->empty() ? (*bias)[size_t(n0 + q)] : 0.0f);
                            yr[q] = accumulate ? yr[q] + v : v;
                        }
                    }
                }
            }
        });
    }

    // Conv1d over [T][in] rows; returns [Tout][out].
    Vec convFwd(const Conv& c, const Vec& x, int T, int& Tout) {
        Tout = (T + 2 * c.pad - c.dil * (c.k - 1) - 1) / c.stride + 1;
        Vec y(size_t(std::max(Tout, 0)) * size_t(c.out));
        if (Tout <= 0) return y;
        if (c.stride == 1) {
            // Zero-padded copy: tap t reads rows [t*dil, t*dil + Tout).
            const int Tp = T + 2 * c.pad;
            Vec xp(size_t(Tp) * size_t(c.in), 0.0f);
            std::copy(x.begin(), x.begin() + long(T) * c.in, xp.begin() + long(c.pad) * c.in);
            convGemm(packPanels(c.taps), c.k, c.in, c.out, xp.data(), Tout, c.dil, &c.b, y.data(), false);
            return y;
        }
        // Strided (only the small F0/N and noise convs): gather each tap's rows, then accumulate.
        const int Tp = T + 2 * c.pad;
        Vec xp(size_t(Tp) * size_t(c.in), 0.0f);
        std::copy(x.begin(), x.begin() + long(T) * c.in, xp.begin() + long(c.pad) * c.in);
        Vec gather(size_t(Tout) * size_t(c.in));
        for (int t = 0; t < c.k; t++) {
            for (int o = 0; o < Tout; o++) {
                const float* r = xp.data() + size_t(o * c.stride + t * c.dil) * size_t(c.in);
                std::copy(r, r + c.in, gather.data() + size_t(o) * c.in);
            }
            mm(gather.data(), Tout, c.taps[size_t(t)], t == 0 ? &c.b : nullptr, y.data(), t > 0);
        }
        return y;
    }

    Vec convTFwd(const ConvT& c, const Vec& x, int T, int& Tout) {
        Tout = (T - 1) * c.stride - 2 * c.pad + c.k;
        Vec z(size_t(T) * size_t(c.out) * size_t(c.k));
        convGemm(packPanels({c.w}), 1, c.in, c.out * c.k, x.data(), T, 0, nullptr, z.data(), false);
        Vec y(size_t(Tout) * size_t(c.out));
        for (int o = 0; o < Tout; o++)
            std::copy(c.b.begin(), c.b.end(), y.begin() + long(o) * c.out);
        for (int t = 0; t < T; t++) {
            const float* zr = z.data() + size_t(t) * size_t(c.out) * size_t(c.k);
            for (int k = 0; k < c.k; k++) {
                const int o = t * c.stride - c.pad + k;
                if (o < 0 || o >= Tout) continue;
                float* yr = y.data() + size_t(o) * size_t(c.out);
                for (int ch = 0; ch < c.out; ch++) yr[ch] += zr[size_t(ch) * c.k + k];
            }
        }
        return y;
    }

    static Vec dwUp(const DwUp& d, const Vec& x, int T) {
        Vec y(size_t(2 * T) * size_t(d.c));
        for (int o = 0; o < 2 * T; o++) std::copy(d.b.begin(), d.b.end(), y.begin() + long(o) * d.c);
        for (int t = 0; t < T; t++)
            for (int k = 0; k < 3; k++) {
                const int o = 2 * t - 1 + k;
                if (o < 0 || o >= 2 * T) continue;
                const float* xr = x.data() + size_t(t) * d.c;
                float* yr = y.data() + size_t(o) * d.c;
                for (int ch = 0; ch < d.c; ch++) yr[ch] += xr[ch] * d.w[size_t(ch) * 3 + k];
            }
        return y;
    }

    // gamma | beta for this utterance's style.
    static Vec styleParams(const StyleFc& f, const float* s, int sdim) {
        Vec gb(size_t(2 * f.channels));
        for (int i = 0; i < 2 * f.channels; i++) {
            float acc = f.b[size_t(i)];
            const float* w = f.w.data() + size_t(i) * size_t(sdim);
            for (int j = 0; j < sdim; j++) acc += w[j] * s[j];
            gb[size_t(i)] = acc;
        }
        return gb;
    }

    // Splits [0, T) into one chunk per worker (vocoder tensors run to 16K frames x 128 channels).
    template <typename F>
    void forRows(int T, F&& fn) {
        const int chunks = std::min(T, pool->size() * 4);
        pool->parallelFor(chunks, [&](int c0, int c1, int) {
            for (int c = c0; c < c1; c++) fn(int(int64_t(T) * c / chunks), int(int64_t(T) * (c + 1) / chunks));
        });
    }

    // InstanceNorm1d (per channel over time, eps 1e-5) followed by (1 + gamma) * x + beta. Statistics are
    // gathered per time chunk in parallel (mean first, then the centred second moment, both in double).
    void adaIn(Vec& x, int T, int C, const Vec& gb) {
        const int chunks = std::min(T, pool->size() * 4);
        std::vector<double> part(size_t(chunks) * C);
        auto reduce = [&](auto&& rowFn) {
            std::fill(part.begin(), part.end(), 0.0);
            pool->parallelFor(chunks, [&](int c0, int c1, int) {
                for (int ch = c0; ch < c1; ch++) {
                    double* acc = part.data() + size_t(ch) * C;
                    for (int t = int(int64_t(T) * ch / chunks); t < int(int64_t(T) * (ch + 1) / chunks); t++)
                        rowFn(x.data() + size_t(t) * C, acc);
                }
            });
            std::vector<double> total(static_cast<size_t>(C), 0.0);
            for (int ch = 0; ch < chunks; ch++)
                for (int c = 0; c < C; c++) total[size_t(c)] += part[size_t(ch) * C + c];
            return total;
        };
        std::vector<double> mean = reduce([&](const float* r, double* acc) {
            for (int c = 0; c < C; c++) acc[c] += r[c];
        });
        for (double& m : mean) m /= T;
        std::vector<double> var = reduce([&](const float* r, double* acc) {
            for (int c = 0; c < C; c++) {
                const double d = r[c] - mean[size_t(c)];
                acc[c] += d * d;
            }
        });
        Vec scale(static_cast<size_t>(C)), shift(static_cast<size_t>(C));
        for (int c = 0; c < C; c++) {
            const float rstd = float(1.0 / std::sqrt(var[size_t(c)] / T + 1e-5));
            scale[size_t(c)] = (1.0f + gb[size_t(c)]) * rstd;
            shift[size_t(c)] = gb[size_t(C + c)] - scale[size_t(c)] * float(mean[size_t(c)]);
        }
        forRows(T, [&](int t0, int t1) {
            for (int t = t0; t < t1; t++) {
                float* r = x.data() + size_t(t) * C;
                for (int c = 0; c < C; c++) r[c] = r[c] * scale[size_t(c)] + shift[size_t(c)];
            }
        });
    }

    static void leaky(Vec& x, float slope) {
        for (float& v : x) v = v > 0 ? v : v * slope;
    }

    // sin(x) to ~4e-6: reduce to [-pi, pi], fold to [-pi/2, pi/2], odd polynomial (Taylor to x^9).
    static inline float fastSin(float x) {
        constexpr float kTwoPi = 6.28318530717958647f, kInvTwoPi = 0.159154943091895336f, kHalfPi = 1.57079632679489662f;
        float r = x - kTwoPi * std::nearbyint(x * kInvTwoPi);
        if (r > kHalfPi) r = kPi - r;
        else if (r < -kHalfPi) r = -kPi - r;
        const float r2 = r * r;
        return r * (1.0f + r2 * (-1.0f / 6 + r2 * (1.0f / 120 + r2 * (-1.0f / 5040 + r2 * (1.0f / 362880)))));
    }

    Vec adainResBlkFwd(const AdainResBlk& b, const Vec& x, int T, const float* s, int& Tout) {
        // residual
        Vec h = x;
        adaIn(h, T, b.in, styleParams(b.n1, s, 128));
        leaky(h, 0.2f);
        int Th = T;
        if (b.upsample) {
            h = dwUp(b.pool, h, T);
            Th = 2 * T;
        }
        int T1, T2;
        Vec r = convFwd(b.c1, h, Th, T1);
        adaIn(r, T1, b.c1.out, styleParams(b.n2, s, 128));
        leaky(r, 0.2f);
        r = convFwd(b.c2, r, T1, T2);
        // shortcut: nearest x2, then 1x1 conv when the width changes
        Vec sc;
        int Ts = T;
        if (b.upsample) {
            sc.resize(size_t(2 * T) * size_t(b.in));
            for (int t = 0; t < 2 * T; t++)
                std::copy(x.begin() + long(t / 2) * b.in, x.begin() + long(t / 2 + 1) * b.in, sc.begin() + long(t) * b.in);
            Ts = 2 * T;
        } else {
            sc = x;
        }
        if (b.learnedSc) {
            int To;
            sc = convFwd(b.sc, sc, Ts, To);
        }
        const float inv = 1.0f / std::sqrt(2.0f);
        for (size_t i = 0; i < r.size(); i++) r[i] = (r[i] + sc[i]) * inv;
        Tout = T2;
        return r;
    }

    // Snake1D: x + sin(a x)^2 / a, per channel a.
    void snake(Vec& x, int T, int C, const Vec& alpha) {
        Vec inv(static_cast<size_t>(C));
        for (int c = 0; c < C; c++) inv[size_t(c)] = 1.0f / alpha[size_t(c)];
        forRows(T, [&](int t0, int t1) {
            for (int t = t0; t < t1; t++) {
                float* r = x.data() + size_t(t) * C;
                for (int c = 0; c < C; c++) {
                    const float s = fastSin(alpha[size_t(c)] * r[c]);
                    r[c] += s * s * inv[size_t(c)];
                }
            }
        });
    }

    Vec resBlock1Fwd(const AdaResBlock1& rb, Vec x, int T, int C, const float* s) {
        for (int j = 0; j < 3; j++) {
            Vec xt = x;
            adaIn(xt, T, C, styleParams(rb.a1[j], s, 128));
            snake(xt, T, C, rb.alpha1[j]);
            int To;
            xt = convFwd(rb.c1[j], xt, T, To);
            adaIn(xt, T, C, styleParams(rb.a2[j], s, 128));
            snake(xt, T, C, rb.alpha2[j]);
            xt = convFwd(rb.c2[j], xt, T, To);
            for (size_t i = 0; i < x.size(); i++) x[i] += xt[i];
        }
        return x;
    }

    static float sigmoid(float v) { return 1.0f / (1.0f + std::exp(-v)); }

    // Bidirectional single-layer LSTM over [T][in]; returns [T][2H] (forward | backward).
    Vec lstmFwd(const Lstm& l, const Vec& x, int T) {
        const int H = l.h, G = 4 * H;
        Vec gin[2];
        for (int d = 0; d < 2; d++) {
            gin[d].resize(size_t(T) * size_t(G));
            mm(x.data(), T, l.wih[d], &l.bias[d], gin[d].data(), false);
        }
        Vec out(size_t(T) * size_t(2 * H));
        pool->parallelFor(2, [&](int d0, int d1, int) {
            for (int d = d0; d < d1; d++) {
                Vec h(static_cast<size_t>(H), 0.0f), c(static_cast<size_t>(H), 0.0f), g(static_cast<size_t>(G));
                const float* whh = l.whh[d].data();
                for (int step = 0; step < T; step++) {
                    const int t = d == 0 ? step : T - 1 - step;
                    const float* gi = gin[d].data() + size_t(t) * G;
                    for (int r = 0; r < G; r++) {
                        const float* w = whh + size_t(r) * H;
                        float acc = 0;
                        int j = 0;
#if defined(__aarch64__)
                        float32x4_t a = vdupq_n_f32(0), b = vdupq_n_f32(0);
                        for (; j + 8 <= H; j += 8) {
                            a = vfmaq_f32(a, vld1q_f32(w + j), vld1q_f32(h.data() + j));
                            b = vfmaq_f32(b, vld1q_f32(w + j + 4), vld1q_f32(h.data() + j + 4));
                        }
                        acc = vaddvq_f32(vaddq_f32(a, b));
#endif
                        for (; j < H; j++) acc += w[j] * h[size_t(j)];
                        g[size_t(r)] = gi[r] + acc;
                    }
                    float* o = out.data() + size_t(t) * size_t(2 * H) + size_t(d * H);
                    for (int j = 0; j < H; j++) {
                        const float ig = sigmoid(g[size_t(j)]), fg = sigmoid(g[size_t(H + j)]);
                        const float gg = std::tanh(g[size_t(2 * H + j)]), og = sigmoid(g[size_t(3 * H + j)]);
                        c[size_t(j)] = fg * c[size_t(j)] + ig * gg;
                        h[size_t(j)] = og * std::tanh(c[size_t(j)]);
                        o[j] = h[size_t(j)];
                    }
                }
            }
        });
        return out;
    }

    static void layerNorm(Vec& x, int T, int C, const float* g, const float* b, float eps) {
        for (int t = 0; t < T; t++) {
            float* r = x.data() + size_t(t) * C;
            double mean = 0, var = 0;
            for (int c = 0; c < C; c++) mean += r[c];
            mean /= C;
            for (int c = 0; c < C; c++) var += (r[c] - mean) * (r[c] - mean);
            const float rstd = float(1.0 / std::sqrt(var / C + eps)), m = float(mean);
            for (int c = 0; c < C; c++) r[c] = (r[c] - m) * rstd * (g ? g[c] : 1.0f) + (b ? b[c] : 0.0f);
        }
    }

    // -------------------------------------------------------------- blocks

    Vec albert(const std::vector<int>& ids) {
        const int T = int(ids.size()), E = 128, D = 768, NH = 12, HD = 64;
        Vec e(size_t(T) * E);
        for (int t = 0; t < T; t++)
            for (int c = 0; c < E; c++)
                e[size_t(t) * E + c] = wordEmb[size_t(ids[size_t(t)]) * E + c] + posEmb[size_t(t) * E + c] + typeEmb[size_t(c)];
        layerNorm(e, T, E, embLnG.data(), embLnB.data(), 1e-12f);
        Vec x = linearFwd(embMap, e, T);
        Vec qv(size_t(T) * D), kv(qv.size()), vv(qv.size()), ctx(qv.size());
        const float scale = 1.0f / std::sqrt(float(HD));
        // ALBERT runs one shared layer 12 times: unpack its weights once per utterance, not once per pass.
        const Vec pq = packPanels({q.w}), pk = packPanels({k.w}), pv = packPanels({v.w}), pd = packPanels({dense.w}),
                  pf = packPanels({ffn.w}), po = packPanels({ffnOut.w});
        auto dense1 = [&](const Vec& panels, const Linear& l, const float* in, float* out, bool acc) {
            convGemm(panels, 1, l.in, l.out, in, T, 0, &l.b, out, acc);
        };
        for (int layer = 0; layer < 12; layer++) {
            dense1(pq, q, x.data(), qv.data(), false);
            dense1(pk, k, x.data(), kv.data(), false);
            dense1(pv, v, x.data(), vv.data(), false);
            pool->parallelFor(NH, [&](int h0, int h1, int) {
                Vec sc(static_cast<size_t>(T));
                for (int h = h0; h < h1; h++)
                    for (int i = 0; i < T; i++) {
                        const float* qi = qv.data() + size_t(i) * D + size_t(h) * HD;
                        float mx = -INFINITY;
                        for (int j = 0; j < T; j++) {
                            const float* kj = kv.data() + size_t(j) * D + size_t(h) * HD;
                            float s = 0;
                            for (int d = 0; d < HD; d++) s += qi[d] * kj[d];
                            sc[size_t(j)] = s * scale;
                            mx = std::max(mx, sc[size_t(j)]);
                        }
                        float sum = 0;
                        for (int j = 0; j < T; j++) sum += (sc[size_t(j)] = std::exp(sc[size_t(j)] - mx));
                        float* o = ctx.data() + size_t(i) * D + size_t(h) * HD;
                        std::fill(o, o + HD, 0.0f);
                        for (int j = 0; j < T; j++) {
                            const float p = sc[size_t(j)] / sum;
                            const float* vj = vv.data() + size_t(j) * D + size_t(h) * HD;
                            for (int d = 0; d < HD; d++) o[d] += p * vj[d];
                        }
                    }
            });
            Vec a = x;  // attention output = LN(x + dense(ctx))
            dense1(pd, dense, ctx.data(), a.data(), true);
            layerNorm(a, T, D, attLnG.data(), attLnB.data(), 1e-12f);
            Vec f(size_t(T) * size_t(ffn.out));
            dense1(pf, ffn, a.data(), f.data(), false);
            for (float& val : f) val = cpu::gelu(val);
            dense1(po, ffnOut, f.data(), a.data(), true);  // a += ffn_output(...)
            layerNorm(a, T, D, fullLnG.data(), fullLnB.data(), 1e-12f);
            x = std::move(a);
        }
        return x;
    }

    static Vec catStyle(const Vec& x, int T, int C, const float* s) {
        Vec y(size_t(T) * size_t(C + 128));
        for (int t = 0; t < T; t++) {
            std::copy(x.begin() + long(t) * C, x.begin() + long(t + 1) * C, y.begin() + long(t) * (C + 128));
            std::copy(s, s + 128, y.begin() + long(t) * (C + 128) + C);
        }
        return y;
    }

    Vec textEncoder(const std::vector<int>& ids) {
        const int T = int(ids.size()), C = 512;
        Vec x(size_t(T) * C);
        for (int t = 0; t < T; t++)
            std::copy(teEmb.begin() + long(ids[size_t(t)]) * C, teEmb.begin() + long(ids[size_t(t)] + 1) * C, x.begin() + long(t) * C);
        for (int i = 0; i < 3; i++) {
            int To;
            x = convFwd(teCnn[i], x, T, To);
            layerNorm(x, T, C, teLnG[i].data(), teLnB[i].data(), 1e-5f);
            leaky(x, 0.2f);
        }
        return lstmFwd(teLstm, x, T);
    }

    // -------------------------------------------------------------- generator

    // SineGen + SourceModuleHnNSF for frame-rate f0 (length L): 9 harmonics, sample rate 24000, 300 samples
    // per frame, voiced above 10 Hz. Follows istftnet.py exactly, including its interpolation quirks: the
    // phase is integrated at frame rate and linearly upsampled, and the random initial phase added to the first
    // sample is lost in the downsampling (so it is omitted here).
    Vec harmonicSource(const Vec& f0, bool noise, uint64_t seed) {
        const int L = int(f0.size()), up = 300, S = L * up, NH = 9;
        std::vector<float> phase(size_t(L) * NH);
        for (int h = 0; h < NH; h++) {
            double acc = 0;
            for (int t = 0; t < L; t++) {
                float r = f0[size_t(t)] * float(h + 1) / 24000.0f;
                r = r - std::floor(r);  // Python %: result in [0, 1)
                acc += r;
                phase[size_t(t) * NH + h] = float(acc) * 2.0f * kPi * float(up);
            }
        }
        std::mt19937_64 rng(seed);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        Vec out(static_cast<size_t>(S));
        for (int n = 0; n < S; n++) {
            // The phases reach ~5e5 rad, where one float step is ~0.03 rad: interpolate and take the sine in
            // double, or the harmonics drift audibly from the reference.
            double src = (double(n) + 0.5) / double(up) - 0.5;
            if (src < 0) src = 0;
            const int i0 = int(src), i1 = std::min(i0 + 1, L - 1);
            const double w = src - double(i0);
            const bool voiced = f0[size_t(n / up)] > 10.0f;
            const float amp = voiced ? 0.003f : 0.1f / 3.0f;
            float acc = srcB[0];
            for (int h = 0; h < NH; h++) {
                const double p = double(phase[size_t(i0) * NH + h]) * (1 - w) + double(phase[size_t(i1) * NH + h]) * w;
                const float sine = float(std::sin(p)) * 0.1f * (voiced ? 1.0f : 0.0f) + (noise ? amp * gauss(rng) : 0.0f);
                acc += srcW[size_t(h)] * sine;
            }
            out[size_t(n)] = std::tanh(acc);
        }
        return out;
    }

    static Vec hann() {
        Vec w(20);
        for (int n = 0; n < 20; n++) w[size_t(n)] = 0.5f - 0.5f * std::cos(2.0f * kPi * float(n) / 20.0f);
        return w;
    }

    // e^(-2 pi i k j / 20) for k = 0..10, j = 0..19.
    struct Twiddles {
        double c[11][20], s[11][20];
        Twiddles() {
            for (int k = 0; k <= 10; k++)
                for (int j = 0; j < 20; j++) {
                    c[k][j] = std::cos(2.0 * M_PI * k * j / 20);
                    s[k][j] = std::sin(2.0 * M_PI * k * j / 20);
                }
        }
    };
    static const Twiddles& twiddles() {
        static const Twiddles t;
        return t;
    }

    // torch.stft(n_fft=20, hop=5, hann, center=True, reflect): [frames][22] = magnitude(11) | phase(11).
    Vec stft(const Vec& x, int& frames) {
        const int N = 20, hop = 5, pad = 10, n = int(x.size());
        frames = n / hop + 1;
        const Vec w = hann();
        const Twiddles& tw = twiddles();
        auto at = [&](int i) {  // reflect padding
            if (i < 0) i = -i;
            if (i >= n) i = 2 * (n - 1) - i;
            return x[size_t(i)];
        };
        Vec out(size_t(frames) * 22);
        forRows(frames, [&](int f0, int f1) {
            for (int f = f0; f < f1; f++) {
                float seg[20];
                for (int j = 0; j < N; j++) seg[j] = at(f * hop - pad + j) * w[size_t(j)];
                for (int k = 0; k <= 10; k++) {
                    double re = 0, im = 0;
                    for (int j = 0; j < N; j++) {
                        re += seg[j] * tw.c[k][j];
                        im -= seg[j] * tw.s[k][j];
                    }
                    out[size_t(f) * 22 + k] = float(std::sqrt(re * re + im * im));
                    out[size_t(f) * 22 + 11 + k] = float(std::atan2(im, re));
                }
            }
        });
        return out;
    }

    // torch.istft of magnitude * e^(i phase): [frames][11] each -> (frames - 1) * 5 samples.
    static Vec istft(const Vec& mag, const Vec& ph, int frames) {
        const int N = 20, hop = 5, pad = 10;
        const Vec w = hann();
        const Twiddles& tw = twiddles();
        const int full = N + hop * (frames - 1);
        Vec y(size_t(full), 0.0f), env(size_t(full), 0.0f);
        for (int f = 0; f < frames; f++) {
            const float* m = mag.data() + size_t(f) * 11;
            const float* p = ph.data() + size_t(f) * 11;
            // irfft: DC and Nyquist use only their real parts; cos(p + a) = cos p cos a - sin p sin a.
            double re[11], im[11];
            for (int k = 0; k <= 10; k++) {
                re[k] = m[k] * std::cos(p[k]);
                im[k] = m[k] * std::sin(p[k]);
            }
            for (int j = 0; j < N; j++) {
                double s = re[0] + re[10] * ((j & 1) ? -1.0 : 1.0);
                for (int k = 1; k < 10; k++) s += 2.0 * (re[k] * tw.c[k][j] - im[k] * tw.s[k][j]);
                y[size_t(f * hop + j)] += float(s / N) * w[size_t(j)];
                env[size_t(f * hop + j)] += w[size_t(j)] * w[size_t(j)];
            }
        }
        const int len = hop * (frames - 1);
        Vec out(static_cast<size_t>(std::max(len, 0)));
        for (int i = 0; i < len; i++) {
            const float e = env[size_t(i + pad)];
            out[size_t(i)] = e > 1e-11f ? y[size_t(i + pad)] / e : 0.0f;
        }
        return out;
    }

    Vec generator(Vec x, int T, const float* s, const Vec& f0, bool noise, uint64_t seed, Debug* debug) {
        if (debug) debug->genIn = x;
        Vec har0 = harmonicSource(f0, noise, seed);
        int frames;
        Vec har = stft(har0, frames);  // [frames][22]
        if (debug) {
            debug->harSource = har0;
            debug->har = har;
        }
        int C = 512;
        for (int i = 0; i < 2; i++) {
            leaky(x, 0.1f);
            int Ts;
            Vec xs = convFwd(noiseConv[i], har, frames, Ts);
            const int cOut = noiseConv[i].out;
            xs = resBlock1Fwd(noiseRes[i], std::move(xs), Ts, cOut, s);
            int Tu;
            x = convTFwd(ups[i], x, T, Tu);
            C = ups[i].out;
            if (i == 1) {  // ReflectionPad1d((1, 0))
                Vec p(size_t(Tu + 1) * size_t(C));
                std::copy(x.begin() + C, x.begin() + 2 * C, p.begin());
                std::copy(x.begin(), x.end(), p.begin() + C);
                x = std::move(p);
                Tu += 1;
            }
            if (Tu != Ts) fail("kokoro: generator length mismatch");
            for (size_t j = 0; j < x.size(); j++) x[j] += xs[j];
            T = Tu;
            Vec sum;
            for (int j = 0; j < 3; j++) {
                Vec r = resBlock1Fwd(res[i * 3 + j], x, T, C, s);
                if (sum.empty()) sum = std::move(r);
                else for (size_t q2 = 0; q2 < sum.size(); q2++) sum[q2] += r[q2];
            }
            for (float& val : sum) val /= 3.0f;
            x = std::move(sum);
        }
        leaky(x, 0.01f);
        int To;
        Vec post = convFwd(convPost, x, T, To);  // [T][22]
        if (debug) debug->post = post;
        Vec mag(size_t(To) * 11), ph(size_t(To) * 11);
        for (int t = 0; t < To; t++)
            for (int k = 0; k < 11; k++) {
                mag[size_t(t) * 11 + k] = std::exp(post[size_t(t) * 22 + k]);
                ph[size_t(t) * 11 + k] = std::sin(post[size_t(t) * 22 + 11 + k]);
            }
        return istft(mag, ph, To);
    }
};

// ------------------------------------------------------------------ KokoroModel

KokoroModel::KokoroModel(const std::string& dir, const LoadOptions& opt, const ProgressFn& progress)
    : impl_(std::make_unique<Impl>()), weights_(opt.weights) {
    Impl& m = *impl_;
    m.fmt = opt.weights;
    m.pool = std::make_unique<ThreadPool>(opt.threads);

    progress(0.0f, "Reading voice config");
    Json cfg = Json::parseFile(dir + "/config.json");
    for (auto& kv : cfg["vocab"].members()) vocab_[kv.first] = int(kv.second.i64());
    if (vocab_.empty()) fail("Kokoro config.json has no vocab");

    std::string ckpt = dir + "/kokoro-v1_0.pth";
    TensorStore ts;
    progress(0.05f, "Mapping voice model");
    ts.addFile(ckpt);
    const std::string B = "bert.module.", P = "predictor.module.", T = "text_encoder.module.", D = "decoder.module.";
    if (!ts.has(B + "embeddings.word_embeddings.weight")) fail("this doesn't look like a Kokoro checkpoint");

    progress(0.1f, "Loading voice text model");
    m.wordEmb = m.f32(ts, B + "embeddings.word_embeddings.weight");
    m.posEmb = m.f32(ts, B + "embeddings.position_embeddings.weight");
    m.typeEmb = m.f32(ts, B + "embeddings.token_type_embeddings.weight");
    m.embLnG = m.f32(ts, B + "embeddings.LayerNorm.weight");
    m.embLnB = m.f32(ts, B + "embeddings.LayerNorm.bias");
    m.embMap = m.linear(ts, B + "encoder.embedding_hidden_mapping_in");
    const std::string L = B + "encoder.albert_layer_groups.0.albert_layers.0.";
    m.q = m.linear(ts, L + "attention.query");
    m.k = m.linear(ts, L + "attention.key");
    m.v = m.linear(ts, L + "attention.value");
    m.dense = m.linear(ts, L + "attention.dense");
    m.attLnG = m.f32(ts, L + "attention.LayerNorm.weight");
    m.attLnB = m.f32(ts, L + "attention.LayerNorm.bias");
    m.ffn = m.linear(ts, L + "ffn");
    m.ffnOut = m.linear(ts, L + "ffn_output");
    m.fullLnG = m.f32(ts, L + "full_layer_layer_norm.weight");
    m.fullLnB = m.f32(ts, L + "full_layer_layer_norm.bias");
    m.bertEncoder = m.linear(ts, "bert_encoder.module");

    progress(0.25f, "Loading voice prosody model");
    for (int i = 0; i < 3; i++) {
        m.durLstm[i] = m.lstm(ts, P + "text_encoder.lstms." + std::to_string(2 * i));
        m.durNorm[i] = m.styleFc(ts, P + "text_encoder.lstms." + std::to_string(2 * i + 1) + ".fc");
    }
    m.predLstm = m.lstm(ts, P + "lstm");
    m.shared = m.lstm(ts, P + "shared");
    m.durProj = m.linear(ts, P + "duration_proj.linear_layer");
    for (int i = 0; i < 3; i++) {
        m.f0[i] = m.adainResBlk(ts, P + "F0." + std::to_string(i), i == 1);
        m.nblk[i] = m.adainResBlk(ts, P + "N." + std::to_string(i), i == 1);
    }
    m.f0Proj = m.conv(ts, P + "F0_proj", 1, 0, 1);
    m.nProj = m.conv(ts, P + "N_proj", 1, 0, 1);

    progress(0.4f, "Loading voice text encoder");
    m.teEmb = m.f32(ts, T + "embedding.weight");
    for (int i = 0; i < 3; i++) {
        const std::string c = T + "cnn." + std::to_string(i);
        m.teCnn[i] = m.conv(ts, c + ".0", 1, 2, 1);
        m.teLnG[i] = m.f32(ts, c + ".1.gamma");
        m.teLnB[i] = m.f32(ts, c + ".1.beta");
    }
    m.teLstm = m.lstm(ts, T + "lstm");

    progress(0.5f, "Loading voice decoder");
    m.f0Conv = m.conv(ts, D + "F0_conv", 2, 1, 1);
    m.nConv = m.conv(ts, D + "N_conv", 2, 1, 1);
    m.asrRes = m.conv(ts, D + "asr_res.0", 1, 0, 1);
    m.encode = m.adainResBlk(ts, D + "encode", false);
    for (int i = 0; i < 4; i++) m.decode[i] = m.adainResBlk(ts, D + "decode." + std::to_string(i), i == 3);

    progress(0.7f, "Loading voice vocoder");
    const std::string G = D + "generator.";
    m.srcW = m.f32(ts, G + "m_source.l_linear.weight");
    m.srcB = m.f32(ts, G + "m_source.l_linear.bias");
    m.noiseConv[0] = m.conv(ts, G + "noise_convs.0", 6, 3, 1);
    m.noiseConv[1] = m.conv(ts, G + "noise_convs.1", 1, 0, 1);
    m.noiseRes[0] = m.resBlock1(ts, G + "noise_res.0", 7);
    m.noiseRes[1] = m.resBlock1(ts, G + "noise_res.1", 11);
    m.ups[0] = m.convT(ts, G + "ups.0", 10, 5);
    m.ups[1] = m.convT(ts, G + "ups.1", 6, 3);
    const int kernels[3] = {3, 7, 11};
    for (int i = 0; i < 6; i++) m.res[i] = m.resBlock1(ts, G + "resblocks." + std::to_string(i), kernels[i % 3]);
    m.convPost = m.conv(ts, G + "conv_post", 1, 3, 1);
    weightBytes_ = m.bytes;
    progress(1.0f, "Voice ready");
    NEKO_LOGI("Kokoro loaded: %s weights, %zu MB", weightsLabel().c_str(), weightBytes_ >> 20);
}

KokoroModel::~KokoroModel() = default;

std::string KokoroModel::weightsLabel() const { return weightFormatName(weights_); }

int KokoroModel::threads() const { return impl_->pool->size(); }

int KokoroModel::phonemeId(const std::string& c) const {
    auto it = vocab_.find(c);
    return it == vocab_.end() ? -1 : it->second;
}

std::vector<float> KokoroModel::synthesize(const std::vector<int>& phonemeIds, const float* style, float speed,
                                           uint64_t seed, bool noise, Debug* debug) {
    Impl& m = *impl_;
    if (phonemeIds.empty()) return {};
    if (int(phonemeIds.size()) > kMaxPhonemes) fail("kokoro: too many phonemes");
    std::vector<int> ids;
    ids.reserve(phonemeIds.size() + 2);
    ids.push_back(0);
    ids.insert(ids.end(), phonemeIds.begin(), phonemeIds.end());
    ids.push_back(0);
    const int T = int(ids.size());
    const float* sDec = style;         // ref_s[:, :128]
    const float* sPred = style + 128;  // ref_s[:, 128:]

    auto clock = std::chrono::steady_clock::now();
    auto lap = [&](double* slot) {
        auto now = std::chrono::steady_clock::now();
        if (debug && slot) *slot = std::chrono::duration<double, std::milli>(now - clock).count();
        clock = now;
    };

    // Durations.
    Vec bert = m.albert(ids);
    lap(debug ? &debug->msAlbert : nullptr);
    Vec dEn = m.linearFwd(m.bertEncoder, bert, T);  // [T][512]
    Vec d = Impl::catStyle(dEn, T, 512, sPred);              // [T][640]
    for (int i = 0; i < 3; i++) {
        Vec h = m.lstmFwd(m.durLstm[i], d, T);  // [T][512]
        Vec gb = Impl::styleParams(m.durNorm[i], sPred, 128);
        Impl::layerNorm(h, T, 512, nullptr, nullptr, 1e-5f);
        for (int t = 0; t < T; t++)
            for (int c = 0; c < 512; c++) {
                float& val = h[size_t(t) * 512 + c];
                val = (1.0f + gb[size_t(c)]) * val + gb[size_t(512 + c)];
            }
        d = Impl::catStyle(h, T, 512, sPred);
    }
    Vec x = m.lstmFwd(m.predLstm, d, T);
    Vec logits = m.linearFwd(m.durProj, x, T);  // [T][50]
    std::vector<int> dur(static_cast<size_t>(T));
    int frames = 0;
    for (int t = 0; t < T; t++) {
        float sum = 0;
        for (int j = 0; j < 50; j++) sum += Impl::sigmoid(logits[size_t(t) * 50 + j]);
        dur[size_t(t)] = std::max(1, int(std::nearbyint(sum / speed)));
        if (debug && debug->forceDurations && debug->forceDurations->size() == dur.size())
            dur[size_t(t)] = (*debug->forceDurations)[size_t(t)];
        frames += dur[size_t(t)];
    }

    // Frame-level features: each token's row repeated for its duration.
    auto expand = [&](const Vec& src, int C) {
        Vec out(size_t(frames) * size_t(C));
        int f = 0;
        for (int t = 0; t < T; t++)
            for (int r = 0; r < dur[size_t(t)]; r++, f++)
                std::copy(src.begin() + long(t) * C, src.begin() + long(t + 1) * C, out.begin() + long(f) * C);
        return out;
    };
    Vec en = expand(d, 640);
    Vec sh = m.lstmFwd(m.shared, en, frames);  // [F][512]
    auto prosody = [&](AdainResBlk* blocks, const Conv& proj) {
        Vec y = sh;
        int Tc = frames;
        for (int i = 0; i < 3; i++) {
            int To;
            y = m.adainResBlkFwd(blocks[i], y, Tc, sPred, To);
            Tc = To;
        }
        int To;
        return m.convFwd(proj, y, Tc, To);  // [2F][1]
    };
    Vec f0 = prosody(m.f0, m.f0Proj), nCurve = prosody(m.nblk, m.nProj);
    if (debug && debug->forceF0 && debug->forceF0->size() == f0.size()) f0 = *debug->forceF0;
    lap(debug ? &debug->msProsody : nullptr);

    Vec asr = expand(m.textEncoder(ids), 512);  // [F][512]

    // Decoder.
    int Tf;
    Vec f0c = m.convFwd(m.f0Conv, f0, 2 * frames, Tf), nc = m.convFwd(m.nConv, nCurve, 2 * frames, Tf);
    auto cat = [&](const Vec& a, int ca, const Vec* extra, int ce) {
        const int C = ca + ce + 2;
        Vec out(size_t(frames) * size_t(C));
        for (int t = 0; t < frames; t++) {
            float* r = out.data() + size_t(t) * C;
            std::copy(a.begin() + long(t) * ca, a.begin() + long(t + 1) * ca, r);
            if (extra) std::copy(extra->begin() + long(t) * ce, extra->begin() + long(t + 1) * ce, r + ca);
            r[ca + ce] = f0c[size_t(t)];
            r[ca + ce + 1] = nc[size_t(t)];
        }
        return out;
    };
    int To;
    Vec h = m.adainResBlkFwd(m.encode, cat(asr, 512, nullptr, 0), frames, sDec, To);  // [F][1024]
    Vec asrRes = m.convFwd(m.asrRes, asr, frames, To);                                 // [F][64]
    int Th = frames;
    for (int i = 0; i < 4; i++) h = m.adainResBlkFwd(m.decode[i], cat(h, 1024, &asrRes, 64), frames, sDec, Th);
    // h: [2F][512]

    lap(debug ? &debug->msDecoder : nullptr);
    if (debug) {
        debug->durations = dur;
        debug->f0 = f0;
        debug->n = nCurve;
    }
    Vec pcm = m.generator(std::move(h), Th, sDec, f0, noise, seed, debug);
    lap(debug ? &debug->msGenerator : nullptr);
    return pcm;
}

// ------------------------------------------------------------------ KokoroVoice

KokoroVoice::KokoroVoice(const std::string& path) {
    TensorStore ts;
    ts.addFile(path);
    const auto& all = ts.all();
    if (all.size() != 1) fail("not a Kokoro voice pack: " + path);
    const std::string name = all.begin()->first;
    const auto& shape = all.begin()->second.shape;
    if (shape.empty() || shape.back() != KokoroModel::kStyleDim) fail("not a Kokoro voice pack: " + path);
    data_ = ts.toF32(name);
    rows_ = int(data_.size() / KokoroModel::kStyleDim);
}

const float* KokoroVoice::style(int phonemes) const {
    const int r = std::max(0, std::min(rows_ - 1, phonemes - 1));
    return data_.data() + size_t(r) * KokoroModel::kStyleDim;
}

}  // namespace neko
