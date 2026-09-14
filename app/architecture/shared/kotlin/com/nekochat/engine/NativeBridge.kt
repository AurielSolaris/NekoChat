package com.nekochat.engine

/** Invoked on the inference thread while a model loads. */
fun interface LoadProgress {
    fun onProgress(fraction: Float, stage: String)
}

/** Receives complete UTF-8 fragments on the inference thread; return false to stop generating. */
fun interface TokenSink {
    fun onText(utf8: ByteArray): Boolean
}

/**
 * Raw JNI surface of libnekochat. Everything except [nativeCancel] must be called from the
 * inference thread owned by [InferenceEngine] (the GPU contexts are bound to that thread).
 */
internal object NativeBridge {
    init {
        System.loadLibrary("nekochat")
    }

    @JvmStatic external fun nativeLoad(dir: String, backend: Int, threads: Int, progress: LoadProgress?): Long
    @JvmStatic external fun nativeRelease(handle: Long)
    @JvmStatic external fun nativeInfo(handle: Long): String
    @JvmStatic external fun nativeContextLength(handle: Long): Int
    @JvmStatic external fun nativeTokenize(handle: Long, utf8: ByteArray): IntArray
    @JvmStatic external fun nativeCancel(handle: Long)
    @JvmStatic external fun nativeResetCache(handle: Long)

    /** Returns [promptTokens, reusedTokens, generatedTokens, prefillMs, decodeMs, stopReason]. */
    @JvmStatic external fun nativeGenerate(
        handle: Long,
        prompt: IntArray,
        maxNew: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        repeatPenalty: Float,
        repeatLastN: Int,
        seed: Long,
        sink: TokenSink,
    ): DoubleArray
}
