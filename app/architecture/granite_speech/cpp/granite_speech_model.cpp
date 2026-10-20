// SPDX-FileCopyrightText: 2026 AurielSolaris
// SPDX-License-Identifier: GPL-3.0-or-later

#include "granite_speech_model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <fstream>
#include <unordered_map>

#include "common.h"
#include "cpu_ops.h"
#include "json.h"
#include "tensor_store.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace neko {

namespace {

constexpr double kPi = 3.14159265358979323846;

// Activations are time-major: [T][C], one row of C channels per frame.
using Vec = std::vector<float>;

struct Linear {
    QMatrix w;  // [out][in]
    Vec b;      // empty when the layer has no bias
};

struct Norm {
    Vec g, b;
};

struct Block {
    Norm ff1Norm, attNorm, convNorm, ff2Norm, outNorm;
    Linear ff1In, ff1Out, ff2In, ff2Out;  // ff*Out carry the conformer's 0.5 residual factor
    Linear qkv;                           // q (pre-scaled by head_dim^-0.5) | k | v, no bias
    Linear o;
    Vec rel;  // [2 * context + 1][headDim]: rel_pos_emb rows for distances -context..context
    Linear pw1, pw2;
    Vec dw, dwBias;  // [inner][kernel] depthwise taps with the BatchNorm folded in, [inner]
    int stride = 1;
};

double nowMs() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

inline float dot(const float* a, const float* b, int n) {
#if defined(__aarch64__)
    float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
        s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    float s = vaddvq_f32(vaddq_f32(s0, s1));
#else
    float s = 0;
    int i = 0;
#endif
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

inline float silu(float v) { return v / (1.0f + std::exp(-v)); }

// In-place iterative radix-2 FFT of n = 2^k complex values.
void fft(std::vector<std::complex<double>>& a) {
    const size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; i++) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const double ang = -2.0 * kPi / double(len);
        const std::complex<double> wl(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            std::complex<double> w(1, 0);
            for (size_t k = 0; k < len / 2; k++) {
                std::complex<double> u = a[i + k], v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wl;
            }
        }
    }
}

const std::string kReplacement = "\xEF\xBF\xBD";  // U+FFFD

// Bytes of the well-formed UTF-8 sequence at p (Unicode table 3-7), or 0; `prefix` gets how many leading bytes
// could still begin one (at least 1), which is what a lossy decoder replaces with a single U+FFFD.
size_t utf8Sequence(const unsigned char* p, size_t avail, size_t& prefix) {
    prefix = 1;
    const unsigned char c = p[0];
    if (c < 0x80) return 1;
    size_t n;
    unsigned char lo = 0x80, hi = 0xBF;  // allowed range of the second byte
    if (c >= 0xC2 && c <= 0xDF) n = 2;
    else if (c >= 0xE0 && c <= 0xEF) {
        n = 3;
        if (c == 0xE0) lo = 0xA0;
        if (c == 0xED) hi = 0x9F;
    } else if (c >= 0xF0 && c <= 0xF4) {
        n = 4;
        if (c == 0xF0) lo = 0x90;
        if (c == 0xF4) hi = 0x8F;
    } else {
        return 0;
    }
    for (size_t k = 1; k < n; k++) {
        if (k >= avail) return 0;
        const unsigned char b = p[k];
        if (k == 1 ? (b < lo || b > hi) : (b < 0x80 || b > 0xBF)) return 0;
        prefix = k + 1;
    }
    return n;
}

// Rust's String::from_utf8_lossy (used by the tokenizers library): each maximal invalid subsequence -> U+FFFD.
std::string lossyUtf8(const std::string& s) {
    std::string out;
    const auto* p = reinterpret_cast<const unsigned char*>(s.data());
    for (size_t i = 0; i < s.size();) {
        size_t prefix = 1;
        const size_t n = utf8Sequence(p + i, s.size() - i, prefix);
        if (n) {
            out.append(s, i, n);
            i += n;
        } else {
            out += kReplacement;
            i += prefix;
        }
    }
    return out;
}

// Code points of a UTF-8 string (invalid bytes are passed through one by one).
std::u32string codePoints(const std::string& s) {
    std::u32string out;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char)s[i];
        const int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
        if (n == 1 || i + size_t(n) > s.size()) {
            out += char32_t(c);
            i++;
            continue;
        }
        char32_t cp = c & (0xFF >> (n + 1));
        for (int k = 1; k < n; k++) cp = (cp << 6) | ((unsigned char)s[i + size_t(k)] & 0x3F);
        out += cp;
        i += size_t(n);
    }
    return out;
}

}  // namespace

// The two Granite Speech 5 checkpoints ship different vocabularies: SentencePiece-style pieces with U+2581 for
// spaces and <0xNN> byte fallback (-nc), and GPT-2 byte-level BPE (Apache), where every character of a piece
// stands for one byte (bytes_to_unicode). Byte-level pieces are converted to raw bytes here.
void GraniteSpeechTokenizer::load(const std::string& dir, int vocab) {
    Json t = Json::parseFile(dir + "/tokenizer.json");
    byteLevel = t["decoder"].getStr("type", "") == "ByteLevel";
    std::unordered_map<char32_t, char> toByte;
    if (byteLevel) {
        int extra = 0;
        for (int b = 0; b < 256; b++) {
            const bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
            toByte[printable ? char32_t(b) : char32_t(256 + extra++)] = char(b);
        }
    }
    pieces.assign(size_t(vocab), std::string());
    control.assign(size_t(vocab), false);
    for (auto& kv : t["model"]["vocab"].members()) {
        const int id = int(kv.second.i64());
        if (id < 0 || id >= vocab) continue;
        if (!byteLevel) {
            pieces[size_t(id)] = kv.first;
            continue;
        }
        std::string raw;
        for (char32_t c : codePoints(kv.first)) {
            auto it = toByte.find(c);
            if (it != toByte.end()) raw += it->second;
        }
        pieces[size_t(id)] = raw;
    }
    for (auto& a : t["added_tokens"].items()) {
        const int id = int(a["id"].i64());
        if (id < 0 || id >= vocab) continue;
        pieces[size_t(id)] = a["content"].str();
        control[size_t(id)] = true;
    }
}

struct GraniteSpeechModel::Impl {
    std::unique_ptr<ThreadPool> pool;
    WeightFormat fmt = WeightFormat::F16;
    unsigned usedFormats = 0;
    size_t bytes = 0;

    // config.json encoder_config
    int C = 1024, layers = 16, heads = 8, headDim = 128, ff = 4096, kernel = 7, inner = 2048, context = 128;
    int maxPos = 512, mels = 80, vocab = 16384;
    float eps = 1e-5f;

    // front end (preprocessor_config.json)
    int nFft = 512, winLength = 400, hop = 160;
    float floorDb = 8.0f;
    Vec window;    // [nFft]: periodic Hann of winLength, centred
    Vec melBank;   // [mels][nFft / 2 + 1]

    Linear input, out, outMid;
    std::vector<Block> blocks;
    GraniteSpeechTokenizer tokenizer;
    double msMatmul = 0, msAttention = 0, msDepthwise = 0;  // profile of the last transcription

    // ---------------------------------------------------------------- loading

    QMatrix matrix(TensorStore& ts, const std::string& name, float scale = 1.0f) {
        RowSource src = ts.rows(name);
        const WeightFormat f = src.cols % kQuantBlock == 0 ? fmt : WeightFormat::F16;
        usedFormats |= 1u << int(f);
        auto read = src.read;
        const int cols = src.cols;
        QMatrix m = quantizeMatrix(f, src.rows, cols, [read, cols, scale](int r, float* o) {
            read(r, o);
            if (scale != 1.0f)
                for (int c = 0; c < cols; c++) o[c] *= scale;
        }, pool.get());
        bytes += m.bytes();
        return m;
    }

    Vec f32(TensorStore& ts, const std::string& name, float scale = 1.0f) {
        Vec v = ts.toF32(name);
        for (float& x : v) x *= scale;
        bytes += v.size() * 4;
        return v;
    }

    Linear linear(TensorStore& ts, const std::string& p, bool bias, float scale = 1.0f) {
        Linear l;
        l.w = matrix(ts, p + ".weight", scale);
        if (bias) l.b = f32(ts, p + ".bias", scale);
        return l;
    }

    Norm norm(TensorStore& ts, const std::string& p) { return {f32(ts, p + ".weight"), f32(ts, p + ".bias")}; }

    void loadConfig(const std::string& dir) {
        Json c = Json::parseFile(dir + "/config.json");
        const Json& e = c["encoder_config"];
        if (e.isNull()) fail("config.json has no encoder_config (not a Granite Speech 5 CTC model)");
        C = int(e.getInt("hidden_size", C));
        layers = int(e.getInt("num_hidden_layers", layers));
        heads = int(e.getInt("num_attention_heads", heads));
        headDim = int(e.getInt("head_dim", C / heads));
        if (int(e.getInt("num_key_value_heads", heads)) != heads) fail("grouped-query speech attention is not supported");
        ff = int(e.getInt("intermediate_size", ff));
        kernel = int(e.getInt("conv_kernel_size", kernel));
        inner = C * int(e.getInt("conv_expansion_factor", 2));
        context = int(e.getInt("context_size", context));
        maxPos = int(e.getInt("max_position_embeddings", maxPos));
        mels = int(e.getInt("num_mel_bins", mels));
        vocab = int(c.getInt("vocab_size", e.getInt("vocab_size", vocab)));
        if (e.getStr("hidden_act", "silu") != "silu") fail("only SiLU speech encoders are supported");
        if (kernel % 2 == 0) fail("conv_kernel_size must be odd");
        if (context > maxPos) fail("context_size larger than max_position_embeddings");
        std::vector<int> sub;
        for (auto& s : e["subsample_layers"].items()) sub.push_back(int(s.i64()));
        blocks.resize(size_t(layers));
        for (int s : sub)
            if (s >= 0 && s < layers) blocks[size_t(s)].stride = 2;

        const std::string pre = dir + "/preprocessor_config.json";
        if (std::ifstream(pre).good()) {
            Json p = Json::parseFile(pre);
            if (p.getInt("sample_rate", kSampleRate) != kSampleRate) fail("speech model expects another sample rate");
            nFft = int(p.getInt("n_fft", nFft));
            winLength = int(p.getInt("win_length", winLength));
            hop = int(p.getInt("hop_length", hop));
            floorDb = float(p.getNum("logmel_floor_db", floorDb));
            if (int(p.getInt("n_mels", mels)) != mels) fail("preprocessor n_mels differs from the encoder");
        }
        if (nFft & (nFft - 1)) fail("n_fft must be a power of two");
    }

    // torch.hann_window(winLength) (periodic), zero-padded to nFft and centred, as torch.stft does.
    // torchaudio.functional.melscale_fbanks(n_freqs, 0, sr / 2, mels, sr, norm=None, mel_scale="htk").
    void buildFrontEnd() {
        window.assign(size_t(nFft), 0.0f);
        const int left = (nFft - winLength) / 2;
        for (int i = 0; i < winLength; i++)
            window[size_t(left + i)] = float(0.5 - 0.5 * std::cos(2.0 * kPi * i / winLength));
        const int nFreq = nFft / 2 + 1;
        auto hzToMel = [](double f) { return 2595.0 * std::log10(1.0 + f / 700.0); };
        auto melToHz = [](double m) { return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0); };
        const double mMin = hzToMel(0.0), mMax = hzToMel(kSampleRate / 2.0);
        std::vector<double> fPts(size_t(mels) + 2);
        for (int i = 0; i < mels + 2; i++) fPts[size_t(i)] = melToHz(mMin + (mMax - mMin) * i / (mels + 1));
        melBank.assign(size_t(mels) * size_t(nFreq), 0.0f);
        for (int k = 0; k < nFreq; k++) {
            const double f = (kSampleRate / 2.0) * k / (nFreq - 1);
            for (int m = 0; m < mels; m++) {
                const double down = (f - fPts[size_t(m)]) / (fPts[size_t(m) + 1] - fPts[size_t(m)]);
                const double up = (fPts[size_t(m) + 2] - f) / (fPts[size_t(m) + 2] - fPts[size_t(m) + 1]);
                melBank[size_t(m) * size_t(nFreq) + size_t(k)] = float(std::max(0.0, std::min(down, up)));
            }
        }
    }


    void load(TensorStore& ts, const ProgressFn& progress) {
        const std::string E = "encoder.";
        if (!ts.has(E + "input_linear.weight")) fail("this doesn't look like a Granite Speech 5 checkpoint");
        input = linear(ts, E + "input_linear", true);
        if (input.w.cols != mels * 4) fail("input_linear doesn't match 2 stacked log-mel + delta frames");
        out = linear(ts, E + "out", true);
        outMid = linear(ts, E + "out_mid", true);
        if (out.w.rows != vocab) fail("CTC head size differs from vocab_size");
        const float qScale = 1.0f / std::sqrt(float(headDim));
        for (int l = 0; l < layers; l++) {
            progress(0.1f + 0.85f * float(l) / float(layers), "Loading speech layer " + std::to_string(l + 1));
            Block& b = blocks[size_t(l)];
            const std::string L = E + "layers." + std::to_string(l) + ".";
            b.ff1Norm = norm(ts, L + "norm_feed_forward1");
            b.attNorm = norm(ts, L + "norm_self_att");
            b.convNorm = norm(ts, L + "norm_conv");
            b.ff2Norm = norm(ts, L + "norm_feed_forward2");
            b.outNorm = norm(ts, L + "norm_out");
            b.ff1In = linear(ts, L + "feed_forward1.linear1", true);
            b.ff1Out = linear(ts, L + "feed_forward1.linear2", true, 0.5f);
            b.ff2In = linear(ts, L + "feed_forward2.linear1", true);
            b.ff2Out = linear(ts, L + "feed_forward2.linear2", true, 0.5f);
            b.qkv.w = matrix(ts, L + "self_attn.q_proj.weight", qScale);
            b.qkv.w.appendRows(matrix(ts, L + "self_attn.k_proj.weight"));
            b.qkv.w.appendRows(matrix(ts, L + "self_attn.v_proj.weight"));
            if (b.qkv.w.rows != 3 * heads * headDim) fail("speech attention projections have an unexpected shape");
            b.o = linear(ts, L + "self_attn.o_proj", true);
            Vec rel = ts.toF32(L + "self_attn.rel_pos_emb.weight");  // [2 * maxPos + 1][headDim]
            const size_t first = size_t(maxPos - context);
            b.rel.assign(rel.begin() + long(first * size_t(headDim)),
                         rel.begin() + long((first + 2 * size_t(context) + 1) * size_t(headDim)));
            bytes += b.rel.size() * 4;
            b.pw1 = linear(ts, L + "conv.pointwise_lin1", true);
            b.pw2 = linear(ts, L + "conv.pointwise_lin2", true);
            Vec dw = ts.toF32(L + "conv.depthwise_conv.weight");  // [inner][1][kernel]
            Vec g = ts.toF32(L + "conv.norm.weight"), beta = ts.toF32(L + "conv.norm.bias");
            Vec mean = ts.toF32(L + "conv.norm.running_mean"), var = ts.toF32(L + "conv.norm.running_var");
            if (dw.size() != size_t(inner) * size_t(kernel)) fail("depthwise conv has an unexpected shape");
            b.dw.resize(dw.size());
            b.dwBias.resize(size_t(inner));
            for (int c = 0; c < inner; c++) {
                const float s = g[size_t(c)] / std::sqrt(var[size_t(c)] + eps);
                for (int k = 0; k < kernel; k++)
                    b.dw[size_t(c) * size_t(kernel) + size_t(k)] = dw[size_t(c) * size_t(kernel) + size_t(k)] * s;
                b.dwBias[size_t(c)] = beta[size_t(c)] - mean[size_t(c)] * s;
            }
            bytes += (b.dw.size() + b.dwBias.size()) * 4;
        }
    }

    // ---------------------------------------------------------------- front end

    // Mirrors GraniteSpeech5FeatureExtractor._extract_features for one clip: returns [frames / 2][4 * mels].
    Vec features(const float* pcm, size_t n, int& encFrames) {
        const size_t melFrames = n / size_t(hop);
        const size_t numFrames = 2 * ((melFrames + 1) / 2);
        encFrames = int(numFrames / 2);
        if (numFrames == 0) return {};
        // Right-pad so the trailing frame pair is filled, then reflect-pad nFft / 2 on each side (center=True).
        Vec audio(pcm, pcm + n);
        const size_t needed = (numFrames - 1) * size_t(hop) + 1;
        if (audio.size() < needed) audio.resize(needed, 0.0f);
        const long len = long(audio.size()), half = nFft / 2;
        if (len <= half) fail("audio too short");
        auto sample = [&](long i) {
            if (i < 0) i = -i;
            if (i >= len) i = 2 * (len - 1) - i;
            return audio[size_t(i)];
        };
        const int nFreq = nFft / 2 + 1;
        Vec logmel(numFrames * size_t(mels));
        pool->parallelFor(int(numFrames), [&](int f0, int f1, int) {
            std::vector<std::complex<double>> buf(static_cast<size_t>(nFft));
            std::vector<double> power(static_cast<size_t>(nFreq));
            for (int f = f0; f < f1; f++) {
                const long start = long(f) * hop - half;
                for (int i = 0; i < nFft; i++) buf[size_t(i)] = {double(sample(start + i)) * window[size_t(i)], 0.0};
                fft(buf);
                for (int k = 0; k < nFreq; k++) power[size_t(k)] = std::norm(buf[size_t(k)]);
                for (int m = 0; m < mels; m++) {
                    const float* w = melBank.data() + size_t(m) * size_t(nFreq);
                    double s = 0;
                    for (int k = 0; k < nFreq; k++) s += w[k] * power[size_t(k)];
                    logmel[size_t(f) * size_t(mels) + size_t(m)] = std::log10(std::max(float(s), 1e-10f));
                }
            }
        });
        const float mx = *std::max_element(logmel.begin(), logmel.end());
        for (float& v : logmel) v = std::max(v, mx - floorDb) / 4.0f + 1.0f;
        // Deltas (torchaudio compute_deltas, window 3, replicate edges), then [logmel | delta] per frame and two
        // frames per encoder row.
        const size_t width = size_t(4 * mels);
        Vec feat(size_t(encFrames) * width);
        for (size_t f = 0; f < numFrames; f++) {
            const float* prev = logmel.data() + (f == 0 ? 0 : f - 1) * size_t(mels);
            const float* cur = logmel.data() + f * size_t(mels);
            const float* next = logmel.data() + std::min(f + 1, numFrames - 1) * size_t(mels);
            float* row = feat.data() + (f / 2) * width + (f % 2) * size_t(2 * mels);
            for (int m = 0; m < mels; m++) {
                row[m] = cur[m];
                row[mels + m] = (next[m] - prev[m]) / 2.0f;
            }
        }
        return feat;
    }

    // ---------------------------------------------------------------- encoder

    void matmul(const float* x, int T, const Linear& l, float* y, int flags) {
        const double t0 = nowMs();
        struct Done {
            double t0;
            double& total;
            ~Done() { total += nowMs() - t0; }
        } done{t0, msMatmul};
        cpu::matmul(*pool, x, T, l.w.cols, l.w, l.b.empty() ? nullptr : l.b.data(), y, l.w.rows, flags);
    }

    // h += 0.5 * linear2(silu(linear1(LayerNorm(h)))): the 0.5 is folded into linear2.
    void feedForward(Vec& h, int T, const Norm& n, const Linear& in, const Linear& outL, Vec& xn, Vec& tmp) {
        cpu::layerNorm(h.data(), xn.data(), n.g.data(), n.b.data(), T, C, eps);
        matmul(xn.data(), T, in, tmp.data(), 0);
        for (size_t i = 0, e = size_t(T) * size_t(ff); i < e; i++) tmp[i] = silu(tmp[i]);
        matmul(tmp.data(), T, outL, h.data(), cpu::kAccumulate);
    }

    // Self-attention inside consecutive blocks of `context` frames (the last one shorter), with Shaw relative
    // position scores q . rel[i - j]. q is pre-scaled, so scores = q . (k_j + rel[i - j]).
    void attention(const Vec& qkv, int T, const Block& b, Vec& att) {
        const double t0 = nowMs();
        const int W = heads * headDim, stride = 3 * W;
        const int nBlocks = (T + context - 1) / context;
        pool->parallelFor(nBlocks * heads, [&](int i0, int i1, int) {
            std::vector<float> p(static_cast<size_t>(context));
            for (int it = i0; it < i1; it++) {
                const int blk = it / heads, h = it % heads;
                const int s = blk * context, n = std::min(context, T - s);
                for (int i = 0; i < n; i++) {
                    const float* q = qkv.data() + size_t(s + i) * size_t(stride) + size_t(h) * size_t(headDim);
                    float mx = -INFINITY;
                    for (int j = 0; j < n; j++) {
                        const float* k = qkv.data() + size_t(s + j) * size_t(stride) + size_t(W + h * headDim);
                        const float* r = b.rel.data() + size_t(i - j + context) * size_t(headDim);
                        const float v = dot(q, k, headDim) + dot(q, r, headDim);
                        p[size_t(j)] = v;
                        mx = std::max(mx, v);
                    }
                    float sum = 0;
                    for (int j = 0; j < n; j++) sum += (p[size_t(j)] = std::exp(p[size_t(j)] - mx));
                    const float inv = 1.0f / sum;
                    float* o = att.data() + size_t(s + i) * size_t(W) + size_t(h) * size_t(headDim);
                    std::fill(o, o + headDim, 0.0f);
                    for (int j = 0; j < n; j++) {
                        const float* v = qkv.data() + size_t(s + j) * size_t(stride) + size_t(2 * W + h * headDim);
                        const float w = p[size_t(j)] * inv;
                        for (int d = 0; d < headDim; d++) o[d] += w * v[d];
                    }
                }
            }
        });
        msAttention += nowMs() - t0;
    }

    // pointwise_lin1 -> GLU -> depthwise conv (BatchNorm folded) -> SiLU -> pointwise_lin2 into `out`
    // ([To][C], To = ceil(T / stride)).
    int convModule(const Vec& xn, int T, const Block& b, Vec& tmp, Vec& glu, Vec& conv, float* out, int flags) {
        matmul(xn.data(), T, b.pw1, tmp.data(), 0);  // [T][2 * inner]
        for (int t = 0; t < T; t++) {
            const float* a = tmp.data() + size_t(t) * size_t(2 * inner);
            float* g = glu.data() + size_t(t) * size_t(inner);
            for (int c = 0; c < inner; c++) g[c] = a[c] / (1.0f + std::exp(-a[inner + c]));
        }
        const int pad = (kernel - 1) / 2, To = (T + 2 * pad - kernel) / b.stride + 1;
        const double t0 = nowMs();
        pool->parallelFor(inner, [&](int c0, int c1, int) {
            for (int c = c0; c < c1; c++) {
                const float* w = b.dw.data() + size_t(c) * size_t(kernel);
                for (int o = 0; o < To; o++) {
                    float s = b.dwBias[size_t(c)];
                    for (int k = 0; k < kernel; k++) {
                        const int t = o * b.stride + k - pad;
                        if (t >= 0 && t < T) s += w[k] * glu[size_t(t) * size_t(inner) + size_t(c)];
                    }
                    conv[size_t(o) * size_t(inner) + size_t(c)] = silu(s);
                }
            }
        });
        msDepthwise += nowMs() - t0;
        matmul(conv.data(), To, b.pw2, out, flags);
        return To;
    }

    // features [T][4 * mels] -> argmax CTC id per output frame.
    std::vector<int> encode(const Vec& feat, int T, Debug* dbg) {
        const size_t maxT = size_t(T);
        Vec h(maxT * size_t(C)), xn(maxT * size_t(C)), tmp(maxT * size_t(std::max(ff, 2 * inner)));
        Vec qkv(maxT * size_t(3 * heads * headDim)), att(maxT * size_t(heads * headDim));
        Vec glu(maxT * size_t(inner)), conv(maxT * size_t(inner)), sub(maxT * size_t(C));
        matmul(feat.data(), T, input, h.data(), 0);
        for (int l = 0; l < layers; l++) {
            Block& b = blocks[size_t(l)];
            feedForward(h, T, b.ff1Norm, b.ff1In, b.ff1Out, xn, tmp);

            cpu::layerNorm(h.data(), xn.data(), b.attNorm.g.data(), b.attNorm.b.data(), T, C, eps);
            matmul(xn.data(), T, b.qkv, qkv.data(), 0);
            attention(qkv, T, b, att);
            matmul(att.data(), T, b.o, h.data(), cpu::kAccumulate);

            cpu::layerNorm(h.data(), xn.data(), b.convNorm.g.data(), b.convNorm.b.data(), T, C, eps);
            if (b.stride == 1) {
                convModule(xn, T, b, tmp, glu, conv, h.data(), cpu::kAccumulate);
            } else {
                convModule(xn, T, b, tmp, glu, conv, sub.data(), 0);
                // Residual pooled by 2 (an odd last frame is dropped) plus the strided conv output.
                const int half = T / 2;
                for (int t = 0; t < half; t++) {
                    const float* a = h.data() + size_t(2 * t) * size_t(C);
                    const float* c = sub.data() + size_t(t) * size_t(C);
                    float* o = h.data() + size_t(t) * size_t(C);  // row t <= 2t: safe in place
                    for (int i = 0; i < C; i++) o[i] = (a[i] + a[C + i]) * 0.5f + c[i];
                }
                T = half;
                if (T == 0) return {};
            }

            feedForward(h, T, b.ff2Norm, b.ff2In, b.ff2Out, xn, tmp);
            cpu::layerNorm(h.data(), h.data(), b.outNorm.g.data(), b.outNorm.b.data(), T, C, eps);
            // Validation: the block's own output, before any self-conditioning (as a forward hook sees it).
            if (dbg && dbg->debugLayers == l + 1) {
                dbg->hidden.assign(h.begin(), h.begin() + long(T) * C);
                return {};
            }

            if (l + 1 == layers / 2) {  // self-conditioned CTC
                Vec mid(size_t(T) * size_t(vocab));
                matmul(h.data(), T, out, mid.data(), 0);
                pool->parallelFor(T, [&](int t0, int t1, int) {
                    for (int t = t0; t < t1; t++) {
                        float* r = mid.data() + size_t(t) * size_t(vocab);
                        const float mx = *std::max_element(r, r + vocab);
                        float s = 0;
                        for (int v = 0; v < vocab; v++) s += (r[v] = std::exp(r[v] - mx));
                        const float inv = 1.0f / s;
                        for (int v = 0; v < vocab; v++) r[v] *= inv;
                    }
                });
                matmul(mid.data(), T, outMid, h.data(), cpu::kAccumulate);
            }
        }
        Vec logits(size_t(T) * size_t(vocab));
        matmul(h.data(), T, out, logits.data(), 0);
        std::vector<int> ids(static_cast<size_t>(T));
        for (int t = 0; t < T; t++) {
            const float* r = logits.data() + size_t(t) * size_t(vocab);
            ids[size_t(t)] = int(std::max_element(r, r + vocab) - r);
        }
        return ids;
    }
};

GraniteSpeechModel::GraniteSpeechModel(const std::string& dir, const LoadOptions& opt, const ProgressFn& progress)
    : impl_(std::make_unique<Impl>()) {
    Impl& m = *impl_;
    m.fmt = opt.weights;
    m.pool = std::make_unique<ThreadPool>(opt.threads);
    progress(0.0f, "Reading speech config");
    m.loadConfig(dir);
    m.buildFrontEnd();
    m.tokenizer.load(dir, m.vocab);
    TensorStore ts;
    progress(0.05f, "Mapping speech model");
    ts.addFile(dir + "/model.safetensors");
    m.load(ts, progress);
    progress(1.0f, "Ready");
    NEKO_LOGI("Granite Speech loaded: %d layers, %s weights (%zu MB)", m.layers, weightsLabel().c_str(), m.bytes >> 20);
}

GraniteSpeechModel::~GraniteSpeechModel() = default;

size_t GraniteSpeechModel::weightBytes() const { return impl_->bytes; }

std::string GraniteSpeechModel::weightsLabel() const {
    std::string s = weightFormatName(impl_->fmt);
    if (impl_->fmt != WeightFormat::F16 && (impl_->usedFormats & (1u << int(WeightFormat::F16)))) s += " + FP16";
    return s;
}

std::string GraniteSpeechModel::transcribe(const float* pcm, size_t n, Debug* debug) {
    Impl& m = *impl_;
    if (n < size_t(kSampleRate / 10)) return {};
    const double t0 = nowMs();
    int frames = 0;
    Vec feat = m.features(pcm, n, frames);
    const double t1 = nowMs();
    if (debug) {
        debug->features = feat;
        debug->msFrontEnd = t1 - t0;
    }
    if (frames == 0) return {};
    m.msMatmul = m.msAttention = m.msDepthwise = 0;
    std::vector<int> frameIds = m.encode(feat, frames, debug);
    if (debug) {
        debug->msEncoder = nowMs() - t1;
        debug->frameIds = frameIds;
        debug->msMatmul = m.msMatmul;
        debug->msAttention = m.msAttention;
        debug->msDepthwise = m.msDepthwise;
    }
    // Greedy CTC: collapse repeats, then drop the blank (id 0).
    std::vector<int> ids;
    int prev = -1;
    for (int id : frameIds) {
        if (id != prev && id != 0) ids.push_back(id);
        prev = id;
    }
    return m.tokenizer.decode(ids);
}

// tokenizer.json's decoder. SentencePiece pieces: "▁" -> " ", byte fallback (<0xNN>), fuse, strip one
// leading space. Byte-level pieces are already raw bytes. Added tokens (<unk>, <|blank|>, <|tokN|>) carry no text.
// The text is always valid UTF-8 (it goes straight to Java), with invalid bytes replaced as tokenizers does:
// a run of <0xNN> pieces that isn't valid UTF-8 becomes one U+FFFD per piece; byte-level text is decoded lossily.
std::string GraniteSpeechTokenizer::decode(const std::vector<int>& ids) const {
    static const std::string kSpace = "\xE2\x96\x81";
    std::string outText, bytes;  // bytes: the pending run of byte-fallback pieces
    int byteRun = 0;
    auto flushBytes = [&] {
        if (byteRun == 0) return;
        if (lossyUtf8(bytes) == bytes) outText += bytes;
        else for (int k = 0; k < byteRun; k++) outText += kReplacement;
        bytes.clear();
        byteRun = 0;
    };
    for (int id : ids) {
        if (id <= 0 || id >= int(pieces.size()) || control[size_t(id)]) continue;
        const std::string& p = pieces[size_t(id)];
        if (byteLevel) {
            outText += p;
            continue;
        }
        if (p.size() == 6 && starts_with(p, "<0x") && p[5] == '>') {
            bytes += char(std::stoi(p.substr(3, 2), nullptr, 16));
            byteRun++;
            continue;
        }
        flushBytes();
        for (size_t i = 0; i < p.size();) {
            if (p.compare(i, kSpace.size(), kSpace) == 0) {
                outText += ' ';
                i += kSpace.size();
            } else {
                outText += p[i++];
            }
        }
    }
    flushBytes();
    if (byteLevel) return lossyUtf8(outText);
    if (!outText.empty() && outText[0] == ' ') outText.erase(0, 1);
    return outText;
}

}  // namespace neko
