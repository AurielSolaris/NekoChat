// Persistent worker pool for CPU kernels. Workers are pinned to the big cores and spin
// briefly between kernels (decode issues ~50 small kernels per token).
#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace neko {

class ThreadPool {
public:
    // nThreads <= 0 selects the number of performance cores.
    explicit ThreadPool(int nThreads);
    ~ThreadPool();

    int size() const { return nThreads_; }

    // Splits [0, n) into contiguous ranges; fn(begin, end, threadIndex). Calling thread participates.
    void parallelFor(int n, const std::function<void(int, int, int)>& fn);

    static std::vector<int> performanceCores();

    static constexpr int kWorkerNice = 4;

private:
    void workerLoop(int index);

    int nThreads_;
    std::vector<std::thread> workers_;
    std::vector<int> cores_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::atomic<uint64_t> generation_{0};
    std::atomic<int> pending_{0};
    std::atomic<bool> stop_{false};
    const std::function<void(int, int, int)>* job_ = nullptr;
    int jobN_ = 0;
};

}  // namespace neko
