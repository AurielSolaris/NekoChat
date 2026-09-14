#include "thread_pool.h"

#include <algorithm>
#include <cstdio>
#include <fstream>

#if defined(__linux__) || defined(__ANDROID__)
#include <sched.h>
#include <unistd.h>
#endif

#include "common.h"

namespace neko {

std::vector<int> ThreadPool::performanceCores() {
    std::vector<std::pair<long, int>> freqs;
    unsigned hc = std::max(1u, std::thread::hardware_concurrency());
    for (int cpu = 0; cpu < int(hc); cpu++) {
        char path[128];
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpu);
        std::ifstream f(path);
        long khz = 0;
        if (f >> khz) freqs.emplace_back(khz, cpu);
    }
    std::vector<int> out;
    if (freqs.empty()) {
        for (int i = 0; i < int(hc); i++) out.push_back(i);
        return out;
    }
    long top = 0;
    for (auto& f : freqs) top = std::max(top, f.first);
    // Big + prime cores are within ~20% of the fastest cluster (e.g. 2.8 vs 3.2 GHz); little cores
    // are further below (Exynos 9611: A55 1.74 GHz vs A76 2.31 GHz = 75%).
    for (auto& f : freqs)
        if (f.first * 5 >= top * 4) out.push_back(f.second);
    return out;
}

ThreadPool::ThreadPool(int nThreads) {
    cores_ = performanceCores();
    nThreads_ = nThreads > 0 ? nThreads : int(cores_.size());
    nThreads_ = std::max(1, std::min(nThreads_, 16));
    for (int i = 1; i < nThreads_; i++) workers_.emplace_back([this, i] { workerLoop(i); });
    NEKO_LOGI("thread pool: %d threads (%zu performance cores)", nThreads_, cores_.size());
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
        generation_++;
    }
    cv_.notify_all();
    for (auto& t : workers_) t.join();
}

void ThreadPool::workerLoop(int index) {
#if defined(__ANDROID__)
    if (!cores_.empty()) {
        cpu_set_t set;
        CPU_ZERO(&set);
        for (int c : cores_) CPU_SET(c, &set);
        sched_setaffinity(0, sizeof set, &set);
    }
#endif
    uint64_t seen = 0;
    for (;;) {
        // Spin a little before sleeping: kernels arrive back to back during a forward pass.
        int spins = 0;
        while (generation_.load(std::memory_order_acquire) == seen && spins < 20000) {
            spins++;
#if defined(__aarch64__)
            asm volatile("yield");
#endif
        }
        if (generation_.load(std::memory_order_acquire) == seen) {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return generation_.load() != seen; });
        }
        seen = generation_.load(std::memory_order_acquire);
        if (stop_) return;
        int n = jobN_;
        int b = int(int64_t(n) * index / nThreads_);
        int e = int(int64_t(n) * (index + 1) / nThreads_);
        if (b < e) (*job_)(b, e, index);
        pending_.fetch_sub(1, std::memory_order_acq_rel);
    }
}

void ThreadPool::parallelFor(int n, const std::function<void(int, int, int)>& fn) {
    if (n <= 0) return;
    if (nThreads_ == 1 || n == 1) {
        fn(0, n, 0);
        return;
    }
    {
        std::lock_guard<std::mutex> lk(mu_);
        job_ = &fn;
        jobN_ = n;
        pending_.store(nThreads_ - 1, std::memory_order_release);
        generation_.fetch_add(1, std::memory_order_acq_rel);
    }
    cv_.notify_all();
    int e = int(int64_t(n) / nThreads_);
    if (e > 0) fn(0, e, 0);
    while (pending_.load(std::memory_order_acquire) != 0) {
#if defined(__aarch64__)
        asm volatile("yield");
#else
        std::this_thread::yield();
#endif
    }
}

}  // namespace neko
