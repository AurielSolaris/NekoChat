#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace neko {

struct SamplingParams {
    float temperature = 0.8f;  // <= 0 means greedy
    int topK = 40;
    float topP = 0.95f;
    float repeatPenalty = 1.1f;
    int repeatLastN = 64;
    uint64_t seed = 0;  // 0 = random
};

class Sampler {
public:
    explicit Sampler(const SamplingParams& p);
    // logits are modified in place.
    int sample(float* logits, int n, const std::vector<int>& history);

private:
    SamplingParams p_;
    std::mt19937_64 rng_;
    std::vector<std::pair<float, int>> cand_;
};

}  // namespace neko
