package com.nekochat.download

import java.io.File

internal enum class FileState { Active, Paused, Complete, Error }

internal data class FileStatus(
    val bytesDone: Long,
    val bytesPerSecond: Long,
    val state: FileState,
    val error: String? = null,
)

/**
 * Moves single files from a URL to disk for [HubDownloadService]. Engines continue from what is
 * already on disk, but each in its own format (aria2 writes segments out of order and keeps a
 * .aria2 control file; Fetch appends), so a download must always be resumed by the engine that
 * started it.
 */
internal interface TransferEngine {
    /** Starts, or continues from partial data, fetching [url] into dir/name. Returns a handle. */
    fun add(url: String, dir: File, name: String, size: Long, connections: Int): String

    /** null when the handle is unknown (e.g. the engine restarted): add the file again to continue. */
    fun status(handle: String): FileStatus?

    fun pause(handle: String)

    /** false when the handle is unknown: add the file again to continue. */
    fun resume(handle: String): Boolean

    fun remove(handle: String)

    fun close()
}
