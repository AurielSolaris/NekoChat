package com.nekochat.engine

import android.os.Process
import android.system.Os
import android.system.OsConstants
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.Closeable
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference

enum class ComputeBackend(val id: Int, val label: String) {
    Auto(0, "Auto"),
    Vulkan(1, "Vulkan"),
    OpenGL(2, "OpenGL"),
    CPU(3, "CPU"),
}

/** How weights are stored in memory. ids match neko::WeightFormat. */
enum class WeightPrecision(val id: Int, val label: String, val detail: String) {
    FP16(0, "FP16", "Full quality and the most memory: the precision the model ships in."),
    FP8(1, "FP8", "About half the memory with nearly the same answers. A good pick for bigger models."),
    FP4(2, "FP4", "About a quarter of the memory and the fastest on the CPU. Answers change noticeably, " +
        "most of all on small models."),
}

/** Memory held by the loaded model, plus the whole app's resident memory for context. */
data class MemoryUsage(
    val weights: Long,
    val kvUsed: Long,
    val kvCapacity: Long,
    val kvResident: Long,
    val workspace: Long,
    val appResident: Long,
) {
    /** Weights, KV cache and activations: what the model itself occupies right now. */
    val model: Long get() = weights + kvResident + workspace
}

data class EngineInfo(
    val architecture: String,
    val backend: String,
    val device: String,
    val note: String,
    val format: String,
    /** Precision the weights are stored in ("FP8", or "FP8 + FP16" when some matrices couldn't be quantized). */
    val weights: String,
    val parameters: Long,
    val layers: Int,
    val contextLength: Int,
) {
    companion object {
        fun parse(json: String): EngineInfo {
            val o = JSONObject(json)
            return EngineInfo(
                architecture = o.optString("architecture"),
                backend = o.optString("backend"),
                device = o.optString("device"),
                note = o.optString("note").trim(),
                format = o.optString("format"),
                weights = o.optString("weights"),
                parameters = o.optLong("params"),
                layers = o.optInt("layers"),
                contextLength = o.optInt("context"),
            )
        }
    }
}

data class GenerationParams(
    val maxNewTokens: Int = 160,
    val temperature: Float = 0.8f,
    val topK: Int = 40,
    val topP: Float = 0.95f,
    val repeatPenalty: Float = 1.15f,
    val repeatLastN: Int = 64,
    val seed: Long = 0L,
)

data class GenerationStats(
    val promptTokens: Int,
    val reusedTokens: Int,
    val generatedTokens: Int,
    val prefillMs: Double,
    val decodeMs: Double,
    val stopReason: Int,
) {
    val tokensPerSecond: Double
        get() = if (decodeMs > 0) generatedTokens * 1000.0 / decodeMs else 0.0
}

/**
 * Latest streamed text, written by the inference thread and read by the UI once per frame.
 * The UI never recomposes more often than the display refreshes, whatever the token rate.
 */
class StreamBuffer {
    private val text = AtomicReference("")
    private val versionCounter = AtomicLong(0)
    val version: Long get() = versionCounter.get()
    val current: String get() = text.get()

    fun publish(value: String) {
        text.set(value)
        versionCounter.incrementAndGet()
    }

    fun reset() = publish("")
}

/**
 * Owns the dedicated inference thread. The UI thread (and Android's RenderThread) never run
 * model code: every native call is marshalled onto "neko-inference", which also owns the
 * Vulkan / EGL context. Native CPU kernels fan out to a worker pool pinned to the big cores.
 *
 * The inference thread and the native workers run at nice [INFERENCE_NICE]: below the UI, its
 * RenderThread and the keyboard, so typing stays smooth while a model generates, but under 10,
 * where Android would move them to the background cgroup (little cores only).
 */
class InferenceEngine : Closeable {
    private val executor = Executors.newSingleThreadExecutor { r ->
        Thread(null, {
            Process.setThreadPriority(INFERENCE_NICE)
            r.run()
        }, "neko-inference", 8L shl 20)
    }
    private val dispatcher = executor.asCoroutineDispatcher()
    private val handleLock = Any()

    @Volatile private var handle = 0L
    @Volatile private var cancelRequested = false

    suspend fun load(
        modelDir: File,
        backend: ComputeBackend,
        precision: WeightPrecision,
        onProgress: (Float, String) -> Unit,
    ): EngineInfo = withContext(dispatcher) {
        releaseOnThread()
        val h = NativeBridge.nativeLoad(modelDir.absolutePath, backend.id, 0, precision.id) { f, s -> onProgress(f, s) }
        synchronized(handleLock) { handle = h }
        EngineInfo.parse(NativeBridge.nativeInfo(h))
    }

    /** Current memory use, or null without a model. Safe from any thread, including during generation. */
    fun memory(): MemoryUsage? {
        val m = synchronized(handleLock) { if (handle != 0L) NativeBridge.nativeMemory(handle) else null } ?: return null
        return MemoryUsage(m[0], m[1], m[2], m[3], m[4], appResidentBytes())
    }

    private fun appResidentBytes(): Long = runCatching {
        // statm: size resident shared ... (in pages)
        val pages = File("/proc/self/statm").readText().trim().split(' ')[1].toLong()
        pages * Os.sysconf(OsConstants._SC_PAGESIZE)
    }.getOrDefault(0L)

    suspend fun contextLength(): Int = withContext(dispatcher) { NativeBridge.nativeContextLength(requireHandle()) }

    suspend fun tokenize(text: String): IntArray = withContext(dispatcher) {
        NativeBridge.nativeTokenize(requireHandle(), text.encodeToByteArray())
    }

    /** Blocks the inference thread until generation ends; [onText] runs on that thread. */
    suspend fun generate(
        prompt: IntArray,
        params: GenerationParams,
        onText: (String) -> Boolean,
    ): GenerationStats = withContext(dispatcher) {
        cancelRequested = false
        val r = NativeBridge.nativeGenerate(
            requireHandle(), prompt, params.maxNewTokens, params.temperature, params.topK, params.topP,
            params.repeatPenalty, params.repeatLastN, params.seed,
        ) { bytes -> !cancelRequested && onText(bytes.decodeToString()) }
        GenerationStats(r[0].toInt(), r[1].toInt(), r[2].toInt(), r[3], r[4], r[5].toInt())
    }

    /** Safe from any thread. */
    fun cancel() {
        cancelRequested = true
        synchronized(handleLock) {
            if (handle != 0L) NativeBridge.nativeCancel(handle)
        }
    }

    suspend fun resetCache() = withContext(dispatcher) {
        if (handle != 0L) NativeBridge.nativeResetCache(handle)
    }

    suspend fun release() = withContext(dispatcher) { releaseOnThread() }

    override fun close() {
        cancel()
        executor.execute { releaseOnThread() }
        executor.shutdown()
    }

    private fun requireHandle(): Long = handle.takeIf { it != 0L } ?: error("No model loaded")

    companion object {
        /** Matches kWorkerNice in thread_pool.h. */
        const val INFERENCE_NICE = 4
    }

    private fun releaseOnThread() {
        val h = synchronized(handleLock) { handle.also { handle = 0L } }
        if (h != 0L) NativeBridge.nativeRelease(h)
    }
}
