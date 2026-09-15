#include "thread_pool.h"

#include <algorithm>
#include <cstdio>
#include <fstream>

#if defined(__linux__) || defined(__ANDROID__)
#include <sched.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

#include "common.h"

namespace neko {

namespace {

long readSysLong(const char* fmt, int cpu) {
    char path[128];
    std::snprintf(path, sizeof path, fmt, cpu);
    std::ifstream f(path);
    long v = 0;
    return (f >> v) ? v : 0;
}

}  // namespace

std::vector<int> ThreadPool::performanceCores() {
    unsigned hc = std::max(1u, std::thread::hardware_concurrency());
    // The scheduler's per-core capacity (1024 = fastest) tells big from little cores even where the little
    // cluster clocks close to the big one (Snapdragon 695: A55 1.8 GHz vs A78 2.2 GHz). Older kernels only
    // expose frequencies.
    std::vector<std::pair<long, int>> capacity, freq;
    for (int cpu = 0; cpu < int(hc); cpu++) {
        if (long c = readSysLong("/sys/devices/system/cpu/cpu%d/cpu_capacity", cpu)) capacity.emplace_back(c, cpu);
        if (long f = readSysLong("/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpu)) freq.emplace_back(f, cpu);
    }
    const bool useCapacity = capacity.size() == hc;
    const auto& score = useCapacity ? capacity : freq;
    std::vector<int> out;
    if (score.empty()) {
        for (int i = 0; i < int(hc); i++) out.push_back(i);
        return out;
    }
    long top = 0;
    for (auto& s : score) top = std::max(top, s.first);
    // Big and prime cores: capacity within 30% of the fastest, or clock within 20% (Exynos 9611: A53 1.74 GHz
    // vs A73 2.31 GHz = 75%, little).
    for (auto& s : score)
        if (useCapacity ? s.first * 10 >= top * 7 : s.first * 5 >= top * 4) out.push_back(s.second);
    return out;
}

ThreadPool::ThreadPool(int nThreads) {
    cores_ = performanceCores();
    const int hc = int(std::max(1u, std::thread::hardware_concurrency()));
    // Leave at least two cores to the UI thread, RenderThread and the keyboard: with every core busy the app
    // and the IME stop responding while a model generates.
    int autoThreads = int(cores_.size());
    if (hc >= 4) autoThreads = std::min(autoThreads, hc - 2);
    nThreads_ = nThreads > 0 ? nThreads : autoThreads;
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
    // Same niceness as the inference thread (see InferenceEngine): below the UI and the keyboard, but still in
    // the foreground cgroup (Android moves threads at nice >= 10 to the little cores).
    setpriority(PRIO_PROCESS, 0, kWorkerNice);
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
