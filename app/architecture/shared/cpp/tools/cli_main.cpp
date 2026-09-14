// On-device validation / benchmark tool (not shipped in the APK).
//   nekochat_cli <modelDir> [--backend auto|vulkan|gl|cpu] [--threads N] [--prompt TEXT] [-n N] [--temp T]
//                [--ref reference.json]
// With --ref it checks tokenizer ids and greedy continuations against HF transformers output.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "common.h"
#include "engine.h"
#include "json.h"

using namespace neko;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <modelDir> [--backend auto|vulkan|gl|cpu] [--prompt TEXT] [-n N] [--ref FILE]\n",
                     argv[0]);
        return 2;
    }
    std::string dir = argv[1], prompt = "Hello, my name is", ref;
    BackendPref pref = BackendPref::Auto;
    int n = 32, threads = 0;
    float temp = 0.0f;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--backend") {
            std::string b = next();
            pref = b == "vulkan" ? BackendPref::Vulkan : b == "gl" ? BackendPref::OpenGL
                 : b == "cpu" ? BackendPref::CPU : BackendPref::Auto;
        } else if (a == "--prompt") prompt = next();
        else if (a == "-n") n = std::stoi(next());
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--temp") temp = std::stof(next());
        else if (a == "--ref") ref = next();
    }

    bool bisect = false;
    for (int i = 2; i < argc; i++)
        if (std::string(argv[i]) == "--bisect") bisect = true;

    try {
        if (bisect) {
            // Runs the CPU and the selected GPU backend layer by layer and reports where they diverge.
            auto quiet = [](float, const std::string&) {};
            std::string type = readModelType(dir);
            BpeTokenizer tk;
            tk.load(dir, type == "qwen3" ? PreTokenizer::Qwen2 : PreTokenizer::Gpt2);
            auto ids = tk.encode(prompt);
            auto ref = loadModel(type, dir, BackendPref::CPU, threads, quiet);
            auto gpu = loadModel(type, dir, pref, threads, quiet);
            std::printf("bisect: %s vs CPU, %zu tokens\n", gpu->backendName().c_str(), ids.size());
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
        auto t0 = std::chrono::steady_clock::now();
        Engine engine(dir, pref, threads, [](float f, const std::string& s) {
            std::fprintf(stderr, "\r[%3d%%] %-40s", int(f * 100), s.c_str());
        });
        double loadMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
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
                std::vector<int> out;
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
        if (!ref.empty()) std::printf("reference checks: %s (%d failures)\n", failures ? "FAILED" : "ALL PASS", failures);
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "\nerror: %s\n", e.what());
        return 1;
    }
}
