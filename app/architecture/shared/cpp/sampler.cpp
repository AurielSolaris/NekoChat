#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace neko {

Sampler::Sampler(const SamplingParams& p) : p_(p), rng_(p.seed ? p.seed : std::random_device{}()) {}

int Sampler::sample(float* logits, int n, const std::vector<int>& history) {
    if (p_.repeatPenalty != 1.0f && p_.repeatLastN > 0) {
        std::unordered_set<int> seen;
        size_t start = history.size() > size_t(p_.repeatLastN) ? history.size() - size_t(p_.repeatLastN) : 0;
        for (size_t i = start; i < history.size(); i++) {
            int id = history[i];
            if (id < 0 || id >= n || !seen.insert(id).second) continue;
            logits[id] = logits[id] > 0 ? logits[id] / p_.repeatPenalty : logits[id] * p_.repeatPenalty;
        }
    }
    if (p_.temperature <= 0.0f) return int(std::max_element(logits, logits + n) - logits);

    int k = p_.topK > 0 ? std::min(p_.topK, n) : n;
    cand_.resize(size_t(n));
    for (int i = 0; i < n; i++) cand_[size_t(i)] = {logits[i], i};
    std::partial_sort(cand_.begin(), cand_.begin() + k, cand_.end(), [](auto& a, auto& b) { return a.first > b.first; });
    cand_.resize(size_t(k));

    float mx = cand_[0].first;
    double sum = 0;
    for (auto& c : cand_) {
        c.first = std::exp((c.first - mx) / p_.temperature);
        sum += c.first;
    }
    // Nucleus cut.
    double cum = 0;
    size_t keep = cand_.size();
    for (size_t i = 0; i < cand_.size(); i++) {
        cum += cand_[i].first / sum;
        if (cum >= p_.topP) { keep = i + 1; break; }
    }
    cand_.resize(keep);
    double total = 0;
    for (auto& c : cand_) total += c.first;
    std::uniform_real_distribution<double> u(0.0, total);
    double r = u(rng_);
    for (auto& c : cand_) {
        r -= c.first;
        if (r <= 0) return c.second;
    }
    return cand_.back().second;
}

}  // namespace neko
