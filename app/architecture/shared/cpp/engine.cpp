#include "engine.h"

#include <algorithm>
#include <chrono>

#include "common.h"

namespace neko {

namespace {

double nowMs() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// Length of the longest prefix of s that does not end inside a UTF-8 sequence.
size_t completeUtf8(const std::string& s) {
    size_t n = s.size();
    for (size_t back = 1; back <= 4 && back <= n; back++) {
        unsigned char c = (unsigned char)s[n - back];
        if ((c & 0xC0) == 0x80) continue;  // continuation byte
        size_t need = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
        return need > back ? n - back : n;
    }
    return n;
}

std::string jsonEscape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else if ((unsigned char)c < 0x20) o += ' ';
        else o += c;
    }
    return o;
}

}  // namespace

Engine::Engine(const std::string& dir, BackendPref pref, int threads, const ProgressFn& progress) {
    std::string type = readModelType(dir);
    tokenizer_.load(dir, type == "qwen3" ? PreTokenizer::Qwen2 : PreTokenizer::Gpt2);
    model_ = loadModel(type, dir, pref, threads, progress);
    const int V = model_->shape().vocab;
    if (tokenizer_.vocabSize() > V + 1) NEKO_LOGW("tokenizer vocab (%d) larger than model vocab (%d)", tokenizer_.vocabSize(), V);
    // Base models end with <|endoftext|>; chat models end a turn with <|im_end|>.
    for (const char* s : {"<|endoftext|>", "<|im_end|>"}) {
        int id = tokenizer_.tokenId(s);
        if (id >= 0 && id < V) stopIds_.push_back(id);
    }
    logits_.resize(size_t(V));
}

bool Engine::isStop(int id) const { return std::find(stopIds_.begin(), stopIds_.end(), id) != stopIds_.end(); }

std::string Engine::infoJson() const {
    const ModelShape& s = model_->shape();
    return std::string("{") + "\"architecture\":\"" + model_->architecture() + "\",\"backend\":\"" +
           jsonEscape(model_->backendName()) + "\",\"device\":\"" + jsonEscape(model_->deviceName()) +
           "\",\"note\":\"" + jsonEscape(model_->backendNote()) + "\",\"format\":\"" + model_->checkpointFormat() +
           "\",\"params\":" + std::to_string(model_->parameterCount()) + ",\"layers\":" + std::to_string(s.layers) +
           ",\"embd\":" + std::to_string(s.embd) + ",\"heads\":" + std::to_string(s.heads) +
           ",\"kvHeads\":" + std::to_string(s.kvHeads) + ",\"context\":" + std::to_string(s.context) +
           ",\"vocab\":" + std::to_string(s.vocab) + "}";
}

GenerateStats Engine::generate(const std::vector<int>& promptIn, int maxNew, const SamplingParams& sp,
                               const std::function<bool(const std::string&)>& onText) {
    cancel_.store(false);
    GenerateStats st;
    const int ctx = model_->contextLength();
    const int V = model_->shape().vocab;

    std::vector<int> prompt = promptIn;
    if (prompt.empty()) prompt.push_back(tokenizer_.eosId() >= 0 ? tokenizer_.eosId() : 0);
    if (int(prompt.size()) >= ctx) prompt.erase(prompt.begin(), prompt.end() - (ctx - 1));
    st.promptTokens = int(prompt.size());

    // Reuse the cached prefix; at least one token must be re-run to obtain fresh logits.
    size_t common = 0;
    while (common < cached_.size() && common < prompt.size() && cached_[common] == prompt[common]) common++;
    if (common == prompt.size()) common--;
    cached_.resize(common);
    st.reusedTokens = int(common);

    double t0 = nowMs();
    for (size_t i = common; i < prompt.size();) {
        int T = int(std::min<size_t>(Model::kMaxBatch, prompt.size() - i));
        model_->forward(prompt.data() + i, T, int(cached_.size()), logits_.data());
        cached_.insert(cached_.end(), prompt.begin() + long(i), prompt.begin() + long(i) + T);
        i += size_t(T);
        if (cancel_.load()) {
            st.stopReason = 2;
            st.prefillMs = nowMs() - t0;
            return st;
        }
    }
    double t1 = nowMs();
    st.prefillMs = t1 - t0;

    Sampler sampler(sp);
    std::vector<int> history = cached_;
    std::string pending;
    st.stopReason = 0;
    for (int n = 0; n < maxNew; n++) {
        int next = sampler.sample(logits_.data(), V, history);
        if (isStop(next)) { st.stopReason = 1; break; }
        history.push_back(next);
        st.generatedTokens++;
        pending += tokenizer_.tokenBytes(next);
        size_t ok = completeUtf8(pending);
        if (ok > 0) {
            std::string piece = pending.substr(0, ok);
            pending.erase(0, ok);
            if (!onText(piece)) { st.stopReason = 2; break; }
        }
        if (cancel_.load()) { st.stopReason = 2; break; }
        if (int(cached_.size()) >= ctx) { st.stopReason = 3; break; }
        if (n + 1 == maxNew) break;  // no need to run the model for a token we will not sample
        model_->forward(&next, 1, int(cached_.size()), logits_.data());
        cached_.push_back(next);
    }
    if (!pending.empty()) onText(pending);
    st.decodeMs = nowMs() - t1;
    return st;
}

}  // namespace neko
