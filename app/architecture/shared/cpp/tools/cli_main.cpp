// On-device validation / benchmark tool (not shipped in the APK).
//   nekochat_cli <modelDir> [--backend auto|vulkan|gl|cpu] [--weights fp16|fp8|fp4] [--threads N] [--prompt TEXT]
//                [-n N] [--temp T] [--ref reference.json]
//   nekochat_cli <modelDir> --kl FILE [-n N] [--backend ...] [--weights ...]
//   nekochat_cli <modelDir> --check-quant
// --ref checks tokenizer ids and greedy continuations against HF transformers output.
// --kl runs the text in FILE through an FP16 CPU reference and the selected backend/weights token by token and
// reports perplexities, the mean KL divergence of the next-token distributions and how often the top token agrees.
// --check-quant compares the CPU FP8/FP4 kernels with a dequantized reference.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>

#include "common.h"
#include "cpu_ops.h"
#include "engine.h"
#include "json.h"

using namespace neko;

namespace {

double msSince(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// log-softmax of logits into out; returns nothing.
void logSoftmax(const std::vector<float>& logits, std::vector<double>& out) {
    float mx = *std::max_element(logits.begin(), logits.end());
    double sum = 0;
    for (float l : logits) sum += std::exp(double(l) - mx);
    double lse = mx + std::log(sum);
    out.resize(logits.size());
    for (size_t i = 0; i < logits.size(); i++) out[i] = double(logits[i]) - lse;
}

int checkQuant() {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 0.05f);
    ThreadPool pool(0);
    int failures = 0;
    for (WeightFormat f : {WeightFormat::FP8, WeightFormat::FP4}) {
        for (int T : {1, 3, 70}) {
            const int K = 1024, N = 53;
            std::vector<float> x(size_t(T) * K), w(size_t(N) * K), bias(static_cast<size_t>(N));
            for (auto& v : x) v = nd(rng) * 20;
            for (auto& v : w) v = nd(rng);
            for (auto& v : bias) v = nd(rng);
            w[5] = 1.5f;  // an outlier
            QMatrix q = quantizeMatrix(f, N, K, [&](int r, float* out) {
                std::copy(w.begin() + long(r) * K, w.begin() + long(r + 1) * K, out);
            }, &pool);
            std::vector<float> y(size_t(T) * N), ref(y.size()), row(K);
            double qerr = 0, qref = 0;
            for (int n = 0; n < N; n++) {
                q.dequantizeRow(n, row.data());
                for (int k = 0; k < K; k++) {
                    double d = row[size_t(k)] - w[size_t(n) * K + size_t(k)];
                    qerr += d * d;
                    qref += double(w[size_t(n) * K + size_t(k)]) * w[size_t(n) * K + size_t(k)];
                }
                for (int t = 0; t < T; t++) {
                    double s = bias[size_t(n)];
                    for (int k = 0; k < K; k++) s += double(x[size_t(t) * K + size_t(k)]) * row[size_t(k)];
                    ref[size_t(t) * N + size_t(n)] = float(s);
                }
            }
            cpu::matmul(pool, x.data(), T, K, q, bias.data(), y.data(), N, 0);
            double e2 = 0, r2 = 0;
            for (size_t i = 0; i < y.size(); i++) {
                e2 += (double(y[i]) - ref[i]) * (double(y[i]) - ref[i]);
                r2 += double(ref[i]) * ref[i];
            }
            // FP8 runs in float (exact up to rounding); FP4 quantizes the activations to int8 as well.
            const double rel = std::sqrt(e2 / r2), limit = f == WeightFormat::FP8 ? 1e-5 : 1e-2;
            bool ok = rel < limit;
            failures += !ok;
            std::printf("%s T=%-2d kernel vs dequantized: rel rms err %.2e %s; weight SNR %.1f dB\n",
                        weightFormatName(f), T, rel, ok ? "PASS" : "FAIL", 10 * std::log10(qref / qerr));
        }
    }
    return failures ? 1 : 0;
}

// Decode-shaped matmuls (T = 1) per weight format: time per call and weight bandwidth.
int benchMatmul(int threads) {
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.0f, 0.05f);
    ThreadPool pool(threads);
    struct Shape { const char* name; int N, K; };
    for (Shape s : {Shape{"gate|up 6144x1024", 6144, 1024}, Shape{"down 1024x3072", 1024, 3072},
                    Shape{"lm-head 151936x1024", 151936, 1024}}) {
        std::vector<float> x(static_cast<size_t>(s.K)), y(static_cast<size_t>(s.N));
        for (auto& v : x) v = nd(rng);
        for (WeightFormat f : {WeightFormat::F16, WeightFormat::FP8, WeightFormat::FP4}) {
            QMatrix q = quantizeMatrix(f, s.N, s.K, [&](int, float* out) {
                for (int k = 0; k < s.K; k++) out[k] = nd(rng);
            }, nullptr);
            const int iters = s.N > 100000 ? 10 : 100;
            cpu::matmul(pool, x.data(), 1, s.K, q, nullptr, y.data(), s.N, 0);
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; i++) cpu::matmul(pool, x.data(), 1, s.K, q, nullptr, y.data(), s.N, 0);
            double ms = msSince(t0) / iters;
            std::printf("%-22s %s: %7.3f ms  %5.2f GB/s\n", s.name, weightFormatName(f), ms, q.bytes() / ms / 1e6);
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <modelDir> [--backend auto|vulkan|gl|cpu] [--weights fp16|fp8|fp4] [--prompt TEXT]"
                     " [-n N] [--ref FILE] [--kl FILE] [--check-quant]\n",
                     argv[0]);
        return 2;
    }
    std::string dir = argv[1], prompt = "Hello, my name is", ref, klFile;
    LoadOptions opt;
    int n = 32;
    float temp = 0.0f;
    bool bisect = false, quantCheck = false, bench = false;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--backend") {
            std::string b = next();
            opt.backend = b == "vulkan" ? BackendPref::Vulkan : b == "gl" ? BackendPref::OpenGL
                        : b == "cpu" ? BackendPref::CPU : BackendPref::Auto;
        } else if (a == "--weights") {
            std::string w = next();
            opt.weights = w == "fp8" ? WeightFormat::FP8 : w == "fp4" ? WeightFormat::FP4 : WeightFormat::F16;
        } else if (a == "--prompt") prompt = next();
        else if (a == "-n") n = std::stoi(next());
        else if (a == "--threads") opt.threads = std::stoi(next());
        else if (a == "--temp") temp = std::stof(next());
        else if (a == "--ref") ref = next();
        else if (a == "--kl") klFile = next();
        else if (a == "--bisect") bisect = true;
        else if (a == "--check-quant") quantCheck = true;
        else if (a == "--bench") bench = true;
    }

    try {
        if (quantCheck) return checkQuant();
        if (bench) return benchMatmul(opt.threads);
        auto quiet = [](float, const std::string&) {};
        if (bisect) {
            // Runs the CPU and the selected GPU backend layer by layer and reports where they diverge.
            std::string type = readModelType(dir);
            BpeTokenizer tk;
            tk.load(dir, type == "qwen3" ? PreTokenizer::Qwen2 : PreTokenizer::Gpt2);
            auto ids = tk.encode(prompt);
            LoadOptions cpuOpt = opt;
            cpuOpt.backend = BackendPref::CPU;
            auto ref = loadModel(type, dir, cpuOpt, quiet);
            auto gpu = loadModel(type, dir, opt, quiet);
            std::printf("bisect: %s vs CPU, %zu tokens, %s weights\n", gpu->backendName().c_str(), ids.size(),
                        gpu->weightsLabel().c_str());
            const ModelShape& c = ref->shape();
            auto stages = ref->debugStages();
            size_t maxW = size_t(c.embd);
            for (auto& s : stages) maxW = std::max(maxW, size_t(s.width));
            std::vector<float> a(std::max(size_t(c.vocab), ids.size() * maxW)), b(a.size());
            auto diff = [&](size_t n) {
                double m = 0;
                for (size_t k = 0; k < n; k++) m = std::max(m, double(std::fabs(a[k] - b[k])));
                return m;
            };
            auto both = [&](int layers, int stage) {
                ref->setDebugLayers(layers, stage);
                gpu->setDebugLayers(layers, stage);
                ref->forward(ids.data(), int(ids.size()), 0, a.data());
                gpu->forward(ids.data(), int(ids.size()), 0, b.data());
            };
            for (size_t op = 0; op < stages.size(); op++) {
                both(0, int(op) + 1);
                std::printf("  layer 0 after %-13s: max |cpu-gpu| = %g\n", stages[op].name,
                            diff(ids.size() * size_t(stages[op].width)));
            }
            for (int L = 0; L <= c.layers; L++) {
                both(L, 0);
                std::printf("  after %2d layers: max |cpu-gpu| = %g\n", L, diff(ids.size() * size_t(c.embd)));
            }
            both(-1, 0);
            std::printf("  logits: max |cpu-gpu| = %g\n", diff(size_t(c.vocab)));
            return 0;
        }
        if (!klFile.empty()) {
            std::string type = readModelType(dir);
            BpeTokenizer tk;
            tk.load(dir, type == "qwen3" ? PreTokenizer::Qwen2 : PreTokenizer::Gpt2);
            std::ifstream f(klFile, std::ios::binary);
            std::stringstream ss;
            ss << f.rdbuf();
            auto ids = tk.encode(ss.str());
            if (int(ids.size()) > n + 1) ids.resize(size_t(n) + 1);
            LoadOptions refOpt;
            refOpt.backend = BackendPref::CPU;
            refOpt.threads = opt.threads;
            auto refModel = loadModel(type, dir, refOpt, quiet);
            auto t0 = std::chrono::steady_clock::now();
            auto model = loadModel(type, dir, opt, quiet);
            double loadMs = msSince(t0);
            const int V = model->shape().vocab;
            std::vector<float> la(static_cast<size_t>(V)), lb(static_cast<size_t>(V));
            std::vector<double> pa, pb;
            double nllA = 0, nllB = 0, kl = 0, fwdMs = 0;
            int agree = 0, count = 0;
            for (size_t i = 0; i + 1 < ids.size(); i++) {
                refModel->forward(&ids[i], 1, int(i), la.data());
                auto t1 = std::chrono::steady_clock::now();
                model->forward(&ids[i], 1, int(i), lb.data());
                fwdMs += msSince(t1);
                logSoftmax(la, pa);
                logSoftmax(lb, pb);
                nllA -= pa[size_t(ids[i + 1])];
                nllB -= pb[size_t(ids[i + 1])];
                double k = 0;
                for (int v = 0; v < V; v++) k += std::exp(pa[size_t(v)]) * (pa[size_t(v)] - pb[size_t(v)]);
                kl += k;
                agree += std::max_element(la.begin(), la.end()) - la.begin() ==
                         std::max_element(lb.begin(), lb.end()) - lb.begin();
                count++;
            }
            MemoryStats m = model->memory(count);
            std::printf("%s on %s, %s weights (%.0f MB, load %.0f ms), %d tokens\n", type.c_str(),
                        model->backendName().c_str(), model->weightsLabel().c_str(), double(m.weights) / 1048576.0,
                        loadMs, count);
            std::printf("  ppl: fp16-cpu %.3f  this %.3f\n", std::exp(nllA / count), std::exp(nllB / count));
            std::printf("  mean KL(fp16 || this) %.5f nats, top-1 agreement %.1f%%, %.1f ms/token\n", kl / count,
                        100.0 * agree / count, fwdMs / count);
            return 0;
        }
        auto t0 = std::chrono::steady_clock::now();
        Engine engine(dir, opt, [](float f, const std::string& s) {
            std::fprintf(stderr, "\r[%3d%%] %-40s", int(f * 100), s.c_str());
        });
        double loadMs = msSince(t0);
        std::fprintf(stderr, "\nloaded in %.0f ms: %s\n", loadMs, engine.infoJson().c_str());

        int failures = 0;
        if (!ref.empty()) {
            Json r = Json::parseFile(ref);
            for (auto& tc : r["tokenize"].items()) {
                auto ids = engine.tokenize(tc["text"].str());
                bool ok = ids.size() == tc["ids"].size();
                for (size_t k = 0; ok && k < ids.size(); k++) ok = ids[k] == int(tc["ids"][k].i64());
                std::printf("tokenize %-4s %s\n", ok ? "PASS" : "FAIL", tc["text"].str().c_str());
                if (!ok) {
                    failures++;
                    std::printf("   got:");
                    for (int id : ids) std::printf(" %d", id);
                    std::printf("\n");
                }
            }
            SamplingParams greedy;
            greedy.temperature = 0.0f;
            greedy.repeatPenalty = 1.0f;
            for (auto& g : r["generate"].items()) {
                std::vector<int> ids;
                for (auto& v : g["ids"].items()) ids.push_back(int(v.i64()));
                engine.resetCache();
                std::string text;
                // Capture token ids by re-tokenizing is lossy, so compare decoded text instead.
                auto st = engine.generate(ids, int(g["out"].size()), greedy, [&](const std::string& s) {
                    text += s;
                    return true;
                });
                std::vector<int> expectIds;
                for (auto& v : g["out"].items()) expectIds.push_back(int(v.i64()));
                std::string expect = engine.tokenizer().decode(expectIds);
                bool ok = text == expect;
                std::printf("generate %-4s \"%s\" -> \"%s\"\n", ok ? "PASS" : "FAIL", g["prompt"].str().c_str(),
                            text.c_str());
                if (!ok) {
                    failures++;
                    std::printf("   expected \"%s\"\n", expect.c_str());
                }
                std::printf("   prefill %d tok %.1f ms, decode %d tok %.1f ms (%.2f tok/s)\n", st.promptTokens,
                            st.prefillMs, st.generatedTokens, st.decodeMs,
                            st.generatedTokens > 1 ? 1000.0 * (st.generatedTokens - 1) / st.decodeMs : 0.0);
            }
        }

        SamplingParams sp;
        sp.temperature = temp;
        if (temp <= 0) sp.repeatPenalty = 1.0f;
        engine.resetCache();
        std::printf("\n%s", prompt.c_str());
        auto st = engine.generate(engine.tokenize(prompt), n, sp, [](const std::string& s) {
            std::printf("%s", s.c_str());
            std::fflush(stdout);
            return true;
        });
        std::printf("\n\nprefill: %d tokens in %.1f ms (%.1f tok/s)\n", st.promptTokens, st.prefillMs,
                    1000.0 * st.promptTokens / std::max(st.prefillMs, 1e-3));
        std::printf("decode: %d tokens in %.1f ms (%.2f tok/s)\n", st.generatedTokens, st.decodeMs,
                    1000.0 * st.generatedTokens / std::max(st.decodeMs, 1e-3));
        MemoryStats m = engine.memory();
        std::printf("memory: weights %.1f MB, KV %.1f of %.1f MB (resident %.1f MB), workspace %.1f MB\n",
                    m.weights / 1048576.0, m.kvUsed / 1048576.0, m.kvCapacity / 1048576.0, m.kvResident / 1048576.0,
                    m.workspace / 1048576.0);
        if (!ref.empty()) std::printf("reference checks: %s (%d failures)\n", failures ? "FAILED" : "ALL PASS", failures);
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "\nerror: %s\n", e.what());
        return 1;
    }
}
