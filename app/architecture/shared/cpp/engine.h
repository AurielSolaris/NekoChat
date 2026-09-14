// Generation loop: tokenizer + model + sampler, with KV-cache prefix reuse across chat turns.
// Not thread safe: use from the single inference thread (cancel() may be called from any thread).
#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "model.h"
#include "sampler.h"
#include "tokenizer.h"

namespace neko {

struct GenerateStats {
    int promptTokens = 0;
    int reusedTokens = 0;  // prompt tokens whose KV was already cached
    int generatedTokens = 0;
    double prefillMs = 0;
    double decodeMs = 0;
    int stopReason = 0;  // 0 = max tokens, 1 = end of text, 2 = cancelled, 3 = context full
};

class Engine {
public:
    Engine(const std::string& modelDir, BackendPref pref, int threads, const ProgressFn& progress);

    std::vector<int> tokenize(const std::string& text) const { return tokenizer_.encode(text); }
    const BpeTokenizer& tokenizer() const { return tokenizer_; }

    // onText receives complete UTF-8 fragments; return false to stop.
    GenerateStats generate(const std::vector<int>& prompt, int maxNewTokens, const SamplingParams& sp,
                           const std::function<bool(const std::string&)>& onText);

    void cancel() { cancel_.store(true); }
    void resetCache() { cached_.clear(); }

    int contextLength() const { return model_->contextLength(); }
    std::string infoJson() const;

private:
    bool isStop(int id) const;

    BpeTokenizer tokenizer_;
    std::unique_ptr<Model> model_;
    std::vector<int> stopIds_;  // end-of-text / end-of-turn tokens
    std::vector<int> cached_;   // tokens whose K/V are resident in the cache, in order
    std::vector<float> logits_;
    std::atomic<bool> cancel_{false};
};

}  // namespace neko
