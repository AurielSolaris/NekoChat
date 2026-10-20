// SPDX-FileCopyrightText: 2026 AurielSolaris
// SPDX-License-Identifier: GPL-3.0-or-later

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

    @JvmStatic external fun nativeLoad(dir: String, backend: Int, threads: Int, weights: Int, progress: LoadProgress?): Long
    @JvmStatic external fun nativeRelease(handle: Long)
    @JvmStatic external fun nativeInfo(handle: Long): String

    /** [weights, kvUsed, kvCapacity, kvResident, workspace] in bytes. Safe from any thread. */
    @JvmStatic external fun nativeMemory(handle: Long): LongArray

    // Text-to-speech (Kokoro). Called from the speech thread owned by KokoroSpeech.
    @JvmStatic external fun nativeTtsLoad(modelDir: String, lexiconDir: String, weights: Int, threads: Int): Long
    @JvmStatic external fun nativeTtsRelease(handle: Long)
    @JvmStatic external fun nativeTtsWeightBytes(handle: Long): Long

    /** English text -> Kokoro phonemes (US or UK lexicon). */
    @JvmStatic external fun nativeTtsPhonemize(handle: Long, utf8: ByteArray, british: Boolean): String

    /** Phonemes (at most 510) -> 24 kHz mono samples in [-1, 1]. */
    @JvmStatic external fun nativeTtsSynthesize(
        handle: Long,
        phonemes: ByteArray,
        voicePath: String,
        speed: Float,
        seed: Long,
    ): FloatArray
    // Speech to text (Granite Speech). Called from the dictation thread owned by VoiceInput.
    @JvmStatic external fun nativeAsrLoad(modelDir: String, weights: Int, threads: Int): Long
    @JvmStatic external fun nativeAsrRelease(handle: Long)
    @JvmStatic external fun nativeAsrWeightBytes(handle: Long): Long

    /** 16 kHz mono samples in -1..1 -> UTF-8 text. */
    @JvmStatic external fun nativeAsrTranscribe(handle: Long, pcm: FloatArray): ByteArray

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
