// SPDX-FileCopyrightText: 2026 AurielSolaris
// SPDX-License-Identifier: GPL-3.0-or-later

package com.nekochat.speech

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.os.Build
import android.os.Process
import android.speech.tts.TextToSpeech
import android.util.Log
import com.nekochat.engine.NativeBridge
import com.nekochat.engine.WeightPrecision
import java.io.Closeable
import java.io.File
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.atomic.AtomicInteger
import kotlin.concurrent.thread

/** Something that can read text aloud. */
interface SpeechEngine : Closeable {
    /** Speaks [text]; interrupt = drop whatever is queued or playing first. */
    fun speak(text: String, interrupt: Boolean)

    fun stop()
}

/** Which engine reads aloud. */
enum class SpeechChoice(val label: String) {
    /** Kokoro when it's installed, otherwise the phone's own text-to-speech. */
    Auto("Auto"),
    Kokoro("Kokoro (on-device)"),
    System("Phone's voice"),
}

/** A Kokoro English voice pack: voices/<id>.pt in hexgrad/Kokoro-82M. */
data class KokoroVoiceInfo(val id: String, val name: String) {
    val british: Boolean get() = id.startsWith("b")
    val female: Boolean get() = id[1] == 'f'
    val accent: String get() = if (british) "UK" else "US"
}

object KokoroVoices {
    const val REPO = "hexgrad/Kokoro-82M"
    const val DEFAULT = "af_heart"

    /** English voices only: other languages need phonemizers NekoChat doesn't ship yet. */
    val all: List<KokoroVoiceInfo> = listOf(
        "af_heart" to "Heart", "af_bella" to "Bella", "af_nicole" to "Nicole", "af_sarah" to "Sarah",
        "af_sky" to "Sky", "af_nova" to "Nova", "af_river" to "River", "af_alloy" to "Alloy",
        "af_aoede" to "Aoede", "af_jessica" to "Jessica", "af_kore" to "Kore",
        "am_michael" to "Michael", "am_fenrir" to "Fenrir", "am_puck" to "Puck", "am_echo" to "Echo",
        "am_eric" to "Eric", "am_liam" to "Liam", "am_onyx" to "Onyx", "am_adam" to "Adam", "am_santa" to "Santa",
        "bf_emma" to "Emma", "bf_isabella" to "Isabella", "bf_alice" to "Alice", "bf_lily" to "Lily",
        "bm_george" to "George", "bm_fable" to "Fable", "bm_lewis" to "Lewis", "bm_daniel" to "Daniel",
    ).map { (id, name) -> KokoroVoiceInfo(id, name) }

    fun byId(id: String?): KokoroVoiceInfo = all.firstOrNull { it.id == id } ?: all.first()
}

/** Where Kokoro lives once downloaded, and the misaki lexicons it phonemizes with (bundled in the APK). */
object SpeechFiles {
    fun kokoroDir(context: Context): File =
        File(context.getExternalFilesDir("speech") ?: File(context.filesDir, "speech"), "kokoro-82m")

    fun installed(context: Context): Boolean {
        val d = kokoroDir(context)
        return File(d, "config.json").isFile && File(d, "kokoro-v1_0.pth").isFile &&
            File(d, "voices/${KokoroVoices.DEFAULT}.pt").isFile
    }

    /** Copies the misaki lexicons out of the APK once (the native side reads them by path). */
    fun lexiconDir(context: Context): File {
        val dir = File(context.filesDir, "misaki").apply { mkdirs() }
        for (name in listOf("us_gold.json", "us_silver.json", "gb_gold.json", "gb_silver.json")) {
            val f = File(dir, name)
            if (f.isFile && f.length() > 0) continue
            context.assets.open("misaki/$name").use { input -> f.outputStream().use { input.copyTo(it) } }
        }
        return dir
    }
}

/** Android's TextToSpeech. Missing on many de-Googled phones: [available] says whether an engine answered. */
class SystemSpeech(context: Context) : SpeechEngine {
    @Volatile var available = false
        private set
    private val tts: TextToSpeech = TextToSpeech(context.applicationContext) { status ->
        available = status == TextToSpeech.SUCCESS
    }

    override fun speak(text: String, interrupt: Boolean) {
        if (!available || text.isBlank()) return
        val mode = if (interrupt) TextToSpeech.QUEUE_FLUSH else TextToSpeech.QUEUE_ADD
        if (Build.VERSION.SDK_INT >= 21) tts.speak(text, mode, null, "neko-${ids.incrementAndGet()}")
    }

    override fun stop() {
        tts.stop()
    }

    override fun close() {
        tts.shutdown()
    }

    private val ids = AtomicInteger()

    companion object {
        /** True when the phone has at least one text-to-speech engine installed. */
        fun hasEngine(context: Context): Boolean = runCatching {
            context.packageManager.queryIntentServices(
                android.content.Intent(TextToSpeech.Engine.INTENT_ACTION_TTS_SERVICE), 0,
            ).isNotEmpty()
        }.getOrDefault(false)
    }
}

/**
 * Kokoro on the phone's CPU. Text is split into sentences; a synthesis thread turns each into audio while a
 * player thread streams the previous one to AudioTrack, so speech starts after the first sentence. Both run
 * below the UI's priority. [stop] bumps a generation counter that both threads check.
 */
class KokoroSpeech(
    private val context: Context,
    private val precision: WeightPrecision,
    @Volatile var voice: KokoroVoiceInfo,
    @Volatile var speed: Float,
) : SpeechEngine {
    private val synth = Executors.newSingleThreadExecutor { r ->
        Thread(null, {
            Process.setThreadPriority(Process.THREAD_PRIORITY_DEFAULT + 4)
            r.run()
        }, "neko-speech", 8L shl 20)
    }
    private val generation = AtomicInteger()
    private val chunks = LinkedBlockingQueue<Pair<Int, FloatArray>>()
    private var handle = 0L  // speech thread only
    @Volatile private var closed = false

    private val track: AudioTrack = run {
        val min = AudioTrack.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT)
        val size = maxOf(min, SAMPLE_RATE / 2 * 4)
        if (Build.VERSION.SDK_INT >= 23) {
            AudioTrack.Builder()
                .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_ASSISTANT)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
                .setAudioFormat(AudioFormat.Builder().setSampleRate(SAMPLE_RATE).setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
                .setBufferSizeInBytes(size)
                .setTransferMode(AudioTrack.MODE_STREAM)
                .build()
        } else {
            @Suppress("DEPRECATION")
            AudioTrack(AudioManager.STREAM_MUSIC, SAMPLE_RATE, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT,
                size, AudioTrack.MODE_STREAM)
        }
    }

    private val player = thread(name = "neko-speech-audio", isDaemon = true) {
        Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO)
        while (!closed) {
            val (gen, pcm) = try { chunks.take() } catch (_: InterruptedException) { break }
            if (gen != generation.get() || pcm.isEmpty()) continue
            if (track.playState != AudioTrack.PLAYSTATE_PLAYING) track.play()
            var off = 0
            while (off < pcm.size && gen == generation.get() && !closed) {
                val n = track.write(pcm, off, minOf(4096, pcm.size - off), AudioTrack.WRITE_BLOCKING)
                if (n <= 0) break
                off += n
            }
        }
    }

    /** Loads the model now (on the speech thread) so the first reply isn't delayed; [onDone] gets any error. */
    fun preload(onDone: (Throwable?) -> Unit = {}) {
        synth.execute { onDone(runCatching { ensureLoaded() }.exceptionOrNull()) }
    }

    override fun speak(text: String, interrupt: Boolean) {
        if (interrupt) stop()
        val gen = generation.get()
        val parts = sentences(text)
        synth.execute {
            try {
                ensureLoaded()
                val voiceFile = File(SpeechFiles.kokoroDir(context), "voices/${voice.id}.pt")
                for (s in parts) {
                    if (gen != generation.get() || closed) return@execute
                    val ps = NativeBridge.nativeTtsPhonemize(handle, s.encodeToByteArray(), voice.british)
                    for (piece in splitPhonemes(ps)) {
                        if (gen != generation.get() || closed) return@execute
                        val pcm = NativeBridge.nativeTtsSynthesize(handle, piece.encodeToByteArray(), voiceFile.path,
                            speed, System.nanoTime())
                        chunks.put(gen to pcm)
                    }
                }
            } catch (e: Throwable) {
                Log.w("NekoChat", "Kokoro speech failed", e)
            }
        }
    }

    override fun stop() {
        generation.incrementAndGet()
        chunks.clear()
        runCatching {
            track.pause()
            track.flush()
        }
    }

    /** Bytes of Kokoro weights in memory (0 until loaded). */
    @Volatile var weightBytes = 0L
        private set

    override fun close() {
        closed = true
        stop()
        player.interrupt()
        synth.execute {
            if (handle != 0L) NativeBridge.nativeTtsRelease(handle)
            handle = 0L
        }
        synth.shutdown()
        runCatching { track.release() }
    }

    private fun ensureLoaded() {
        if (handle != 0L) return
        val lexicons = SpeechFiles.lexiconDir(context)
        handle = NativeBridge.nativeTtsLoad(SpeechFiles.kokoroDir(context).path, lexicons.path, precision.id, 0)
        weightBytes = NativeBridge.nativeTtsWeightBytes(handle)
    }

    companion object {
        const val SAMPLE_RATE = 24000
        private const val MAX_PHONEMES = 510

        /** Sentences (with their punctuation), long ones cut at commas or spaces; empty ones dropped. */
        fun sentences(text: String): List<String> {
            val out = mutableListOf<String>()
            val cleaned = text.replace(Regex("[*_#`>|~]+"), " ")  // markdown marks aren't spoken
            for (m in Regex("[^.!?…\\n]+[.!?…]*").findAll(cleaned)) {
                var s = m.value.trim()
                while (s.length > 300) {
                    val cut = s.lastIndexOf(',', 300).takeIf { it > 80 } ?: s.lastIndexOf(' ', 300).takeIf { it > 0 } ?: 300
                    out += s.substring(0, cut + 1).trim()
                    s = s.substring(cut + 1).trim()
                }
                if (s.any { it.isLetterOrDigit() }) out += s
            }
            return out
        }

        /** Keeps each synthesis within Kokoro's 510 phonemes, cutting at spaces. */
        fun splitPhonemes(ps: String): List<String> {
            val cps = ps.codePoints().toArray()
            if (cps.size <= MAX_PHONEMES) return listOf(ps)
            val out = mutableListOf<String>()
            var start = 0
            while (start < cps.size) {
                var end = minOf(cps.size, start + MAX_PHONEMES)
                if (end < cps.size) {
                    var sp = end
                    while (sp > start + 50 && cps[sp - 1] != ' '.code) sp--
                    if (sp > start + 50) end = sp
                }
                out += String(cps, start, end - start).trim()
                start = end
            }
            return out
        }
    }
}

/** Speech for the whole app: picks the engine, reads replies and narrates actions when switched on. */
class SpeechManager(private val context: Context) : Closeable {
    private var kokoro: KokoroSpeech? = null
    private var system: SystemSpeech? = null

    var choice: SpeechChoice = SpeechChoice.Auto
    var precision: WeightPrecision = WeightPrecision.FP8
        set(value) {
            if (field != value) {
                field = value
                kokoro?.close()
                kokoro = null
            }
        }
    var voice: KokoroVoiceInfo = KokoroVoices.byId(null)
        set(value) {
            field = value
            kokoro?.voice = value
        }
    var speed: Float = 1f
        set(value) {
            field = value
            kokoro?.speed = value
        }

    val kokoroInstalled: Boolean get() = SpeechFiles.installed(context)
    val systemAvailable: Boolean get() = SystemSpeech.hasEngine(context)

    /** The engine that would speak right now, or null when none is usable. */
    fun activeChoice(): SpeechChoice? = when (choice) {
        SpeechChoice.Kokoro -> if (kokoroInstalled) SpeechChoice.Kokoro else null
        SpeechChoice.System -> if (systemAvailable) SpeechChoice.System else null
        SpeechChoice.Auto -> when {
            kokoroInstalled -> SpeechChoice.Kokoro
            systemAvailable -> SpeechChoice.System
            else -> null
        }
    }

    fun speak(text: String, interrupt: Boolean = true) {
        engine()?.speak(text, interrupt)
    }

    fun stop() {
        kokoro?.stop()
        system?.stop()
    }

    /** Memory held by the Kokoro weights, 0 when not loaded. */
    val weightBytes: Long get() = kokoro?.weightBytes ?: 0L

    fun preload() {
        if (activeChoice() == SpeechChoice.Kokoro) (engine() as? KokoroSpeech)?.preload()
    }

    private fun engine(): SpeechEngine? = when (activeChoice()) {
        SpeechChoice.Kokoro -> kokoro ?: KokoroSpeech(context, precision, voice, speed).also { kokoro = it }
        SpeechChoice.System -> system ?: SystemSpeech(context).also { system = it }
        else -> null
    }

    override fun close() {
        kokoro?.close()
        system?.close()
    }
}
