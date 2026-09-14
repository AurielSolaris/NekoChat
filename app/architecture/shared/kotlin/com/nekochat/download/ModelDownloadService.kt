package com.nekochat.download

import android.content.Context
import kotlinx.coroutines.flow.StateFlow

enum class DownloadStatus { Preparing, Downloading, Paused, Completed, Failed }

data class ModelDownload(
    /** Hugging Face repository, e.g. "Qwen/Qwen3-0.6B". */
    val repoId: String,
    val status: DownloadStatus,
    val bytesDone: Long = 0,
    val bytesTotal: Long = 0,
    val bytesPerSecond: Long = 0,
    val error: String? = null,
    /** The model folder, once completed. */
    val folder: String? = null,
    /** Android is blocking NekoChat's network access (per-app setting); the UI offers a way to fix it. */
    val networkBlocked: Boolean = false,
    /** Engine moving the bytes; fixed for the life of a download. */
    val engine: DownloadEngine = DownloadEngine.Aria2Multi,
) {
    val progress: Float get() = if (bytesTotal > 0) (bytesDone.toFloat() / bytesTotal).coerceIn(0f, 1f) else 0f
    val active: Boolean get() = status == DownloadStatus.Preparing || status == DownloadStatus.Downloading
    val name: String get() = repoId.substringAfterLast('/')
}

class DownloadException(message: String, val networkBlocked: Boolean = false) : Exception(message)

/** What moves the bytes. A download keeps the engine it was started with. */
enum class DownloadEngine(val label: String, val detail: String, val connections: Int) {
    Aria2Multi("aria2 (multi-thread)", "8 connections per file. Fastest.", 8),
    Aria2Single("aria2 (single-thread)", "aria2 with 1 connection per file, for servers or networks that dislike parallel connections.", 1),
    Fetch("Fetch (single-thread)", "Android's own HTTP client, 1 connection per file. No native downloader involved.", 1),
}

/** How download hosts are resolved. */
enum class DnsMode(val label: String) {
    /** Android's resolver: honours Private DNS, VPNs and the network's DNS. */
    System("System DNS"),

    /** The downloader's own resolver with public servers (Cloudflare, Google, Quad9). */
    Builtin("Built-in DNS"),
}

/**
 * Fetches models from the Hugging Face Hub into NekoChat's models folder, where the model
 * scanner picks them up. The UI depends only on this interface; [HubDownloadService] is the
 * current implementation (with aria2 and Fetch transfer engines) and can be replaced without
 * touching the model screens.
 */
interface ModelDownloadService {
    val downloads: StateFlow<List<ModelDownload>>

    fun start(repoId: String)
    fun pause(repoId: String)
    fun resume(repoId: String)

    /** Stops the download and deletes its partial files. */
    fun cancel(repoId: String)

    /** Removes a finished entry from the list; the model stays installed. */
    fun dismiss(repoId: String)

    /** Resolver for aria2 transfers. Applies to downloads started or resumed afterwards. */
    fun setDnsMode(mode: DnsMode)

    /** Engine for downloads started from now on; existing downloads keep theirs. */
    fun setEngine(engine: DownloadEngine)
}

/** App-wide instance: downloads outlive screens and the view model. */
object Downloads {
    @Volatile private var instance: ModelDownloadService? = null

    fun get(context: Context): ModelDownloadService = instance ?: synchronized(this) {
        instance ?: HubDownloadService(context.applicationContext).also { instance = it }
    }
}
