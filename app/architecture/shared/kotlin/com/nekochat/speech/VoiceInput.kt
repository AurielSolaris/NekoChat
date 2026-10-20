// SPDX-FileCopyrightText: 2026 AurielSolaris
// SPDX-License-Identifier: GPL-3.0-or-later

package com.nekochat.speech

import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Process
import android.util.Log
import com.nekochat.engine.NativeBridge
import com.nekochat.engine.WeightPrecision
import java.io.Closeable
import java.io.File
import java.util.concurrent.Executors
import kotlin.concurrent.thread

/**
 * The speech-to-text models voice input can use: IBM Granite Speech 5.0 TurboCTC (470M, English). Same
 * architecture and size; they differ in training data and license.
 */
enum class DictationModel(val repo: String, val folder: String, val label: String, val license: String, val detail: String) {
    /** The default: trained on more audio (~75k hours), for personal, non-commercial use only. */
    GraniteNc(
        "ibm-granite/granite-speech-5.0-470m-turboctc-nc", "granite-speech-5.0-nc", "Granite Speech NC",
        "CC BY-NC-SA 4.0",
        "Most accurate. Licensed for personal, non-commercial use only (CC BY-NC-SA 4.0).",
    ),
    GraniteApache(
        "ibm-granite/granite-speech-5.0-470m-turboctc", "granite-speech-5.0", "Granite Speech",
        "Apache 2.0",
        "Free for any use (Apache 2.0). Trained on a little less audio.",
    );

    /** Size of the download, for the UI. */
    val downloadLabel: String get() = "950 MB"

    companion object {
        fun byRepo(repo: String): DictationModel? = entries.firstOrNull { it.repo == repo }
        fun byName(name: String?): DictationModel = entries.firstOrNull { it.name == name } ?: GraniteNc
    }
}

object DictationFiles {
    /** The files a speech-to-text download fetches. */
    val FILES = listOf("config.json", "preprocessor_config.json", "tokenizer.json", "model.safetensors")

    fun dir(context: Context, model: DictationModel): File =
        File(context.getExternalFilesDir("speech") ?: File(context.filesDir, "speech"), model.folder)

    fun installed(context: Context, model: DictationModel): Boolean {
        val d = dir(context, model)
        return listOf("config.json", "tokenizer.json", "model.safetensors").all { File(d, it).isFile }
    }
}

/**
 * Voice input: records 16 kHz mono from the microphone, then transcribes on the phone's CPU with Granite Speech.
 * The model is loaded on first use on its own thread (below the UI's priority) and released by [release] or
 * [close]. Recording needs the RECORD_AUDIO permission, which the caller requests first.
 */
class VoiceInput(private val context: Context) : Closeable {
    private val worker = Executors.newSingleThreadExecutor { r ->
        Thread(null, {
            Process.setThreadPriority(Process.THREAD_PRIORITY_DEFAULT + 4)
            r.run()
        }, "neko-dictation", 8L shl 20)
    }
    private var handle = 0L  // worker thread only
    private var loaded: Pair<DictationModel, WeightPrecision>? = null  // worker thread only

    @Volatile private var recording = false
    private var recorder: Thread? = null
    private val pcm = FloatArrayBuilder()

    /** Input level of the last 50 ms, 0..1, for the recording indicator. */
    @Volatile var level = 0f
        private set

    /** Bytes of model weights in memory (0 until loaded). */
    @Volatile var weightBytes = 0L
        private set

    val isRecording: Boolean get() = recording

    /** Starts recording; false when the microphone couldn't be opened. */
    @SuppressLint("MissingPermission")  // the caller holds RECORD_AUDIO
    fun start(): Boolean {
        if (recording) return true
        val min = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        if (min <= 0) return false
        val rec = try {
            AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION, SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT, maxOf(min, SAMPLE_RATE / 5 * 2))
        } catch (e: Exception) {
            Log.w("NekoChat", "Microphone unavailable", e)
            return false
        }
        if (rec.state != AudioRecord.STATE_INITIALIZED) {
            rec.release()
            return false
        }
        pcm.clear()
        recording = true
        rec.startRecording()
        recorder = thread(name = "neko-mic", isDaemon = true) {
            Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO)
            val buf = ShortArray(SAMPLE_RATE / 20)  // 50 ms
            try {
                while (recording && pcm.size < MAX_SECONDS * SAMPLE_RATE) {
                    val n = rec.read(buf, 0, buf.size)
                    if (n <= 0) break
                    level = pcm.addPcm16(buf, n)
                }
            } finally {
                runCatching { rec.stop() }
                rec.release()
                level = 0f
                recording = false
            }
        }
        return true
    }

    /** Stops recording and returns the audio (16 kHz, -1..1). */
    fun stop(): FloatArray {
        recording = false
        recorder?.join(1000)
        recorder = null
        return pcm.toArray()
    }

    /** Discards a recording in progress. */
    fun cancel() {
        stop()
        pcm.clear()
    }

    /** Transcribes [audio] on the worker thread; [onDone] receives the text or the failure. */
    fun transcribe(audio: FloatArray, model: DictationModel, precision: WeightPrecision, onDone: (Result<String>) -> Unit) {
        worker.execute {
            onDone(runCatching {
                ensureLoaded(model, precision)
                val t0 = System.nanoTime()
                val text = NativeBridge.nativeAsrTranscribe(handle, audio).toString(Charsets.UTF_8).trim()
                val peak = audio.maxOfOrNull { kotlin.math.abs(it) } ?: 0f
                Log.i("NekoChat", "Dictation: %.1f s of audio (peak %.2f), transcribed in %d ms, %d characters".format(
                    audio.size / SAMPLE_RATE.toFloat(), peak, (System.nanoTime() - t0) / 1_000_000, text.length))
                text
            })
        }
    }

    /** Loads the model ahead of the first transcription. */
    fun preload(model: DictationModel, precision: WeightPrecision) {
        worker.execute { runCatching { ensureLoaded(model, precision) }.onFailure { Log.w("NekoChat", "Speech model load failed", it) } }
    }

    /** Frees the model's memory; the next transcription loads it again. */
    fun release() {
        worker.execute { unload() }
    }

    override fun close() {
        cancel()
        worker.execute { unload() }
        worker.shutdown()
    }

    private fun ensureLoaded(model: DictationModel, precision: WeightPrecision) {
        if (handle != 0L && loaded == model to precision) return
        unload()
        handle = NativeBridge.nativeAsrLoad(DictationFiles.dir(context, model).path, precision.id, 0)
        loaded = model to precision
        weightBytes = NativeBridge.nativeAsrWeightBytes(handle)
    }

    private fun unload() {
        if (handle != 0L) NativeBridge.nativeAsrRelease(handle)
        handle = 0L
        loaded = null
        weightBytes = 0L
    }

    /** Growable float buffer shared between the microphone thread and [stop]. */
    private class FloatArrayBuilder {
        private var data = FloatArray(SAMPLE_RATE * 10)
        @Volatile var size = 0
            private set

        /** Appends [n] 16-bit samples as -1..1 floats; returns their peak level. */
        @Synchronized fun addPcm16(src: ShortArray, n: Int): Float {
            while (size + n > data.size) data = data.copyOf(data.size * 2)
            var peak = 0
            for (i in 0 until n) {
                data[size + i] = src[i] / 32768f
                peak = maxOf(peak, kotlin.math.abs(src[i].toInt()))
            }
            size += n
            return peak / 32768f
        }

        @Synchronized fun clear() {
            size = 0
        }

        @Synchronized fun toArray(): FloatArray = data.copyOf(size)
    }

    companion object {
        const val SAMPLE_RATE = 16000
        /** A dictation is cut after this long. */
        const val MAX_SECONDS = 120
    }
}
