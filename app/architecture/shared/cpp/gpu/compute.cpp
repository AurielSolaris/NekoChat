#include "compute.h"

#include <cmath>
#include <random>
#include <vector>

#include "../common.h"
#include "../cpu_ops.h"
#include "../quant.h"

namespace neko::gpu {

namespace {

bool close(const std::vector<float>& a, const std::vector<float>& b, float tol, std::string& error, const char* what) {
    for (size_t i = 0; i < a.size(); i++) {
        float d = std::fabs(a[i] - b[i]);
        if (!(d <= tol * (1.0f + std::fabs(b[i])))) {
            error = std::string("self-test ") + what + " mismatch at " + std::to_string(i) + ": gpu " +
                    std::to_string(a[i]) + " cpu " + std::to_string(b[i]);
            return false;
        }
    }
    return true;
}

}  // namespace

bool selfTest(ComputeBackend& be, std::string& error) {
    try {
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        auto rnd = [&](size_t n) {
            std::vector<float> v(n);
            for (auto& x : v) x = u(rng);
            return v;
        };
        auto halves = [](std::vector<float>& v) {  // rounds v to f16 in place and returns the bits
            std::vector<uint16_t> h(v.size());
            for (size_t i = 0; i < v.size(); i++) {
                h[i] = f32_to_f16(v[i]);
                v[i] = f16_to_f32(h[i]);
            }
            return h;
        };
        auto run = [&](Kernel k, std::initializer_list<Buffer*> b, std::initializer_list<int32_t> p, uint32_t gx,
                       uint32_t gy) {
            Buffer* binds[4] = {};
            int32_t params[8] = {};
            std::copy(b.begin(), b.end(), binds);
            std::copy(p.begin(), p.end(), params);
            be.dispatch(k, binds, params, gx, gy);
        };
        auto read = [&](Buffer* b, size_t n) {
            std::vector<float> v(n);
            be.download(b, 0, v.data(), n * 4);
            return v;
        };

        // matmul with bias + gelu, T = 5 (exercises the partial 4-token tile).
        {
            const int K = 64, N = 37, T = 5;
            auto x = rnd(size_t(T * K)), bias = rnd(size_t(N)), wf = rnd(size_t(N * K));
            auto w = halves(wf);
            std::vector<float> ref(size_t(T * N));
            for (int t = 0; t < T; t++)
                for (int n = 0; n < N; n++) {
                    float s = bias[size_t(n)];
                    for (int k = 0; k < K; k++) s += x[size_t(t * K + k)] * wf[size_t(n * K + k)];
                    ref[size_t(t * N + n)] = cpu::gelu(s);
                }
            Buffer* by = be.create(ref.size() * 4, nullptr, true);
            be.begin();
            run(Kernel::Matmul, {be.create(x.size() * 4, x.data()), be.create(w.size() * 2, w.data()),
                                 be.create(bias.size() * 4, bias.data()), by},
                {K, N, T, 4 | 1}, uint32_t(N), uint32_t((T + 3) / 4));
            be.submitAndWait();
            if (!close(read(by, ref.size()), ref, 2e-3f, error, "matmul")) return false;
        }

        // K = 768 (loop runs past one pass per thread), row offset, accumulate, output column offset.
        {
            const int BK = 768, BN = 96, BT = 6, row0 = 1, nOff = 5, NS = BN + 8;
            auto bx = rnd(size_t((BT + row0) * BK)), bb = rnd(size_t(BN)), bwf = rnd(size_t(BN * BK));
            auto y0 = rnd(size_t(BT * NS));
            auto bw = halves(bwf);
            std::vector<float> ref = y0;
            for (int t = 0; t < BT; t++)
                for (int n = 0; n < BN; n++) {
                    float s = bb[size_t(n)];
                    for (int k = 0; k < BK; k++) s += bx[size_t((t + row0) * BK + k)] * bwf[size_t(n * BK + k)];
                    ref[size_t(t * NS + nOff + n)] += s;
                }
            Buffer* by = be.create(y0.size() * 4, y0.data(), true);
            be.begin();
            run(Kernel::Matmul, {be.create(bx.size() * 4, bx.data()), be.create(bw.size() * 2, bw.data()),
                                 be.create(bb.size() * 4, bb.data()), by},
                {BK, NS, BT, 4 | 2, row0, nOff}, uint32_t(BN), uint32_t((BT + 3) / 4));
            be.submitAndWait();
            if (!close(read(by, ref.size()), ref, 2e-3f, error, "matmul-768")) return false;
        }

        // Embedding lookup: token + position table, and a chunked table without positions.
        {
            const int EV = 10, EC = 64, EP = 4, ET = 3, epos = 1, split = 6;
            auto wte = rnd(size_t(EV * EC)), wpe = rnd(size_t(EP * EC));
            auto hte = halves(wte), hpe = halves(wpe);
            int32_t toks[ET] = {3, 7, 9};
            std::vector<float> ref(size_t(ET * EC)), refNoPos(size_t(ET * EC));
            for (int t = 0; t < ET; t++)
                for (int c = 0; c < EC; c++) {
                    refNoPos[size_t(t * EC + c)] = wte[size_t(toks[t] * EC + c)];
                    ref[size_t(t * EC + c)] = wte[size_t(toks[t] * EC + c)] + wpe[size_t((epos + t) * EC + c)];
                }
            Buffer* bt = be.create(sizeof toks, toks);
            Buffer* out = be.create(ref.size() * 4, nullptr, true);
            be.begin();
            run(Kernel::Embed, {bt, be.create(hte.size() * 2, hte.data()), be.create(hpe.size() * 2, hpe.data()), out},
                {ET, EC, epos, 1, 0, EV}, 1, ET);
            be.submitAndWait();
            if (!close(read(out, ref.size()), ref, 1e-5f, error, "embed")) return false;
            Buffer* c0 = be.create(size_t(split * EC) * 2, hte.data());
            Buffer* c1 = be.create(size_t((EV - split) * EC) * 2, hte.data() + split * EC);
            be.begin();
            run(Kernel::Embed, {bt, c0, nullptr, out}, {ET, EC, 0, 0, 0, split}, 1, ET);
            run(Kernel::Embed, {bt, c1, nullptr, out}, {ET, EC, 0, 0, split, EV - split}, 1, ET);
            be.submitAndWait();
            if (!close(read(out, ref.size()), refNoPos, 1e-5f, error, "embed-chunked")) return false;
        }

        // layernorm and rmsnorm over 3 rows.
        const int C = 96, R = 3;
        auto lx = rnd(size_t(C * R)), g = rnd(size_t(C)), b = rnd(size_t(C));
        for (auto& v : lx) v *= 20.0f;
        std::vector<float> lref(lx.size()), rref(lx.size());
        cpu::layerNorm(lx.data(), lref.data(), g.data(), b.data(), R, C, 1e-5f);
        cpu::rmsNorm(lx.data(), rref.data(), g.data(), R, C, 1e-6f);
        Buffer* blx = be.create(lx.size() * 4, lx.data());
        Buffer* bg = be.create(g.size() * 4, g.data());
        Buffer* bbb = be.create(b.size() * 4, b.data());
        {
            Buffer* bly = be.create(lx.size() * 4, nullptr, true);
            be.begin();
            run(Kernel::LayerNorm, {blx, bly, bg, bbb}, {C, floatBits(1e-5f), 0, 0}, R, 1);
            be.submitAndWait();
            if (!close(read(bly, lx.size()), lref, 1e-3f, error, "layernorm")) return false;
            be.begin();
            run(Kernel::LayerNorm, {blx, bly, bg, nullptr}, {C, floatBits(1e-6f), 0, 1}, R, 1);
            be.submitAndWait();
            if (!close(read(bly, lx.size()), rref, 1e-3f, error, "rmsnorm")) return false;
        }

        // Chained dispatches in one submission (layernorm -> wide matmul reading its output).
        // Catches drivers that do not honour storage barriers between dispatches.
        {
            const int CN = 3072;
            auto cw = rnd(size_t(CN * C));
            auto chw = halves(cw);
            std::vector<float> cref(size_t(R * CN));
            for (int t = 0; t < R; t++)
                for (int n = 0; n < CN; n++) {
                    float s = 0;
                    for (int k = 0; k < C; k++) s += lref[size_t(t * C + k)] * cw[size_t(n * C + k)];
                    cref[size_t(t * CN + n)] = s;
                }
            Buffer* bcy = be.create(cref.size() * 4, nullptr, true);
            Buffer* bln = be.create(lx.size() * 4);
            be.begin();
            run(Kernel::LayerNorm, {blx, bln, bg, bbb}, {C, floatBits(1e-5f), 0, 0}, R, 1);
            run(Kernel::Matmul, {bln, be.create(chw.size() * 2, chw.data()), nullptr, bcy}, {C, CN, R, 0}, uint32_t(CN), 1);
            be.submitAndWait();
            if (!close(read(bcy, cref.size()), cref, 5e-3f, error, "chained dispatch")) return false;
        }

        // Qwen3 q/k norm + rotary embedding: 2 query heads, 1 key head (+ 1 value head left untouched).
        {
            const int H = 2, KV = 1, hd = 64, T = 2, pos0 = 3, stride = (H + 2 * KV) * hd;
            auto qkv = rnd(size_t(T * stride)), wn = rnd(size_t(2 * hd));
            std::vector<float> cs(size_t((pos0 + T) * hd));
            for (int p = 0; p < pos0 + T; p++)
                for (int i = 0; i < hd / 2; i++) {
                    double a = p * std::pow(10000.0, -2.0 * i / hd);
                    cs[size_t(p * hd + 2 * i)] = float(std::cos(a));
                    cs[size_t(p * hd + 2 * i + 1)] = float(std::sin(a));
                }
            std::vector<float> ref = qkv;
            for (int t = 0; t < T; t++) {
                float* r = ref.data() + t * stride;
                cpu::rmsNorm(r, r, wn.data(), H, hd, 1e-6f);
                cpu::rmsNorm(r + H * hd, r + H * hd, wn.data() + hd, KV, hd, 1e-6f);
                cpu::rope(r, H + KV, hd, cs.data() + (pos0 + t) * hd);
            }
            Buffer* bq = be.create(qkv.size() * 4, qkv.data(), true);
            be.begin();
            run(Kernel::Rope, {bq, be.create(wn.size() * 4, wn.data()), be.create(cs.size() * 4, cs.data())},
                {stride, H, hd, pos0, floatBits(1e-6f), 1}, H + KV, T);
            be.submitAndWait();
            if (!close(read(bq, ref.size()), ref, 1e-3f, error, "rope")) return false;
        }

        // SwiGLU.
        {
            const int T = 2, I = 100;
            auto h = rnd(size_t(T * 2 * I));
            for (auto& v : h) v *= 6.0f;
            std::vector<float> ref(size_t(T * I));
            cpu::siluMul(h.data(), ref.data(), T, I);
            Buffer* bo = be.create(ref.size() * 4, nullptr, true);
            be.begin();
            run(Kernel::SiluMul, {be.create(h.size() * 4, h.data()), bo}, {T, I}, uint32_t((I + 63) / 64), T);
            be.submitAndWait();
            if (!close(read(bo, ref.size()), ref, 1e-4f, error, "silu_mul")) return false;
        }

        // FP8 / FP4 block weights (quant.h) against a dequantized reference: bias + accumulate + output offset,
        // K = 96 (odd block count, idle threads) and K = 1024 (several passes per thread), T = 5.
        for (WeightFormat f : {WeightFormat::FP8, WeightFormat::FP4}) {
            for (int K : {96, 1024}) {
                const int N = 37, T = 5, nOff = 3, NS = N + 5;
                auto x = rnd(size_t(T * K)), bias = rnd(size_t(N)), wf = rnd(size_t(N * K)), y0 = rnd(size_t(T * NS));
                QMatrix w = quantizeMatrix(f, N, K, [&](int r, float* out) {
                    std::copy(wf.begin() + long(r) * K, wf.begin() + long(r + 1) * K, out);
                }, nullptr);
                std::vector<float> ref = y0, row(static_cast<size_t>(K));
                for (int n = 0; n < N; n++) {
                    w.dequantizeRow(n, row.data());
                    for (int t = 0; t < T; t++) {
                        float s = bias[size_t(n)];
                        for (int k = 0; k < K; k++) s += x[size_t(t * K + k)] * row[size_t(k)];
                        ref[size_t(t * NS + nOff + n)] += s;
                    }
                }
                Buffer* by = be.create(y0.size() * 4, y0.data(), true);
                be.begin();
                run(f == WeightFormat::FP8 ? Kernel::MatmulQ8 : Kernel::MatmulQ4,
                    {be.create(x.size() * 4, x.data()), be.create(w.bytes(), w.data.data()),
                     be.create(bias.size() * 4, bias.data()), by},
                    {K, NS, T, 4 | 2, 0, nOff}, uint32_t(N), uint32_t((T + 3) / 4));
                be.submitAndWait();
                if (!close(read(by, ref.size()), ref, 2e-3f, error, weightFormatName(f))) return false;
            }
            // Embedding gather from a quantized table.
            const int EV = 10, EC = 64, ET = 3;
            auto table = rnd(size_t(EV * EC));
            QMatrix q = quantizeMatrix(f, EV, EC, [&](int r, float* out) {
                std::copy(table.begin() + long(r) * EC, table.begin() + long(r + 1) * EC, out);
            }, nullptr);
            int32_t toks[ET] = {3, 7, 9};
            std::vector<float> ref(size_t(ET * EC));
            for (int t = 0; t < ET; t++) q.dequantizeRow(toks[t], ref.data() + t * EC);
            Buffer* out = be.create(ref.size() * 4, nullptr, true);
            be.begin();
            run(Kernel::Embed, {be.create(sizeof toks, toks), be.create(q.bytes(), q.data.data()), nullptr, out},
                {ET, EC, 0, f == WeightFormat::FP8 ? 2 : 4, 0, EV, int(q.rowBytes / 4)}, 1, ET);
            be.submitAndWait();
            if (!close(read(out, ref.size()), ref, 1e-5f, error, "embed-quantized")) return false;
        }

        // kv_store + grouped-query attention over f16 caches. pos0 > 1024 exercises the tiled softmax.
        {
            const int H = 4, KV = 2, hd = 64, T = 3, pos0 = 1030, ctx = pos0 + T;
            const int stride = (H + 2 * KV) * hd, kvW = KV * hd, kOff = H * hd, vOff = (H + KV) * hd;
            auto qkv = rnd(size_t(T * stride));
            auto kf = rnd(size_t(ctx * kvW)), vf = rnd(size_t(ctx * kvW));
            auto kc = halves(kf), vc = halves(vf);
            std::vector<uint16_t> kcRef = kc, vcRef = vc;
            std::vector<float> ref(size_t(T * H * hd));
            ThreadPool pool(1);
            cpu::storeKv(qkv.data() + kOff, qkv.data() + vOff, stride, kcRef.data(), vcRef.data(), kvW, T, pos0);
            cpu::attention(pool, qkv.data(), stride, kcRef.data(), vcRef.data(), ref.data(), T, H, KV, hd, pos0);
            Buffer* bq = be.create(qkv.size() * 4, qkv.data());
            Buffer* bk = be.create(kc.size() * 2, kc.data());
            Buffer* bv = be.create(vc.size() * 2, vc.data());
            Buffer* bo = be.create(ref.size() * 4, nullptr, true);
            be.begin();
            run(Kernel::KvStore, {bq, bk, bv}, {T, kvW, pos0, 0, stride, kOff, vOff}, uint32_t((kvW / 2 + 63) / 64), T);
            run(Kernel::Attention, {bq, bk, bv, bo}, {T, stride, H, pos0, hd, floatBits(1.0f / std::sqrt(float(hd))), kvW},
                H, T);
            be.submitAndWait();
            if (!close(read(bo, ref.size()), ref, 2e-3f, error, "attention")) return false;
        }
        return true;
    } catch (const std::exception& e) {
        error = std::string("self-test failed: ") + e.what();
        return false;
    }
}

}  // namespace neko::gpu
