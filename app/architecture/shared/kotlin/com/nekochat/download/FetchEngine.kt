package com.nekochat.download

import android.os.SystemClock
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong
import kotlin.concurrent.thread

/**
 * Plain HTTPS download with Android's HttpURLConnection: one connection per file, resumed with
 * an HTTP Range request from the current file length. Slower than aria2 but has no native parts.
 */
internal class FetchEngine : TransferEngine {
    private class Job(val url: String, val file: File, val size: Long) {
        @Volatile var done = if (file.exists()) file.length() else 0L
        @Volatile var speed = 0L
        @Volatile var state = FileState.Active
        @Volatile var error: String? = null
        @Volatile var stop = false
        var thread: Thread? = null
    }

    private val jobs = ConcurrentHashMap<String, Job>()
    private val ids = AtomicLong()

    override fun add(url: String, dir: File, name: String, size: Long, connections: Int): String {
        val job = Job(url, File(dir, name), size)
        val handle = "fetch:${ids.incrementAndGet()}"
        jobs[handle] = job
        run(job)
        return handle
    }

    override fun status(handle: String): FileStatus? =
        jobs[handle]?.let { FileStatus(it.done, it.speed, it.state, it.error) }

    override fun pause(handle: String) {
        jobs[handle]?.let {
            it.stop = true
            it.state = FileState.Paused
            it.speed = 0
        }
    }

    override fun resume(handle: String): Boolean {
        val job = jobs[handle] ?: return false
        if (job.state != FileState.Active && job.state != FileState.Complete) run(job)
        return true
    }

    override fun remove(handle: String) {
        jobs.remove(handle)?.let {
            it.stop = true
            it.thread?.join(3000)  // let it close the file before the caller deletes it
        }
    }

    override fun close() {
        jobs.values.forEach { it.stop = true }
    }

    private fun run(job: Job) {
        job.thread?.join(3000)  // never two writers on one file
        job.stop = false
        job.state = FileState.Active
        job.error = null
        job.thread = thread(isDaemon = true, name = "fetch-${job.file.name}") { transfer(job) }
    }

    private fun transfer(job: Job) {
        var attempt = 0
        while (!job.stop) {
            try {
                if (copy(job)) {
                    job.state = FileState.Complete
                    job.speed = 0
                    return
                }
            } catch (e: DownloadException) {
                job.state = FileState.Error
                job.error = e.message
                job.speed = 0
                return
            } catch (e: IOException) {
                if (job.stop) break
                if (++attempt >= MAX_TRIES) {
                    job.state = FileState.Error
                    job.error = "Network error (${e.message ?: e.javaClass.simpleName})."
                    job.speed = 0
                    return
                }
                job.speed = 0
                Thread.sleep(3000L * attempt)
            }
        }
        job.speed = 0
        if (job.state == FileState.Active) job.state = FileState.Paused
    }

    /** true when the file is complete, false when stopped; throws to retry. */
    private fun copy(job: Job): Boolean {
        var have = if (job.file.exists()) job.file.length() else 0L
        if (job.size > 0 && have > job.size) {
            job.file.delete()
            have = 0
        }
        if (job.size > 0 && have == job.size) {
            job.done = have
            return true
        }
        val c = (URL(job.url).openConnection() as HttpURLConnection).apply {
            connectTimeout = 20_000
            readTimeout = 30_000
            setRequestProperty("User-Agent", USER_AGENT)
            if (have > 0) setRequestProperty("Range", "bytes=$have-")
        }
        try {
            val append = when (val code = c.responseCode) {
                206 -> true
                200 -> false.also { have = 0 }  // server ignored the range: start over
                416 -> {
                    job.file.delete()
                    throw IOException("range not satisfiable, restarting")
                }
                401, 403 -> throw DownloadException("Access denied (HTTP $code).")
                404 -> throw DownloadException("File not found on the server (HTTP 404).")
                else -> throw IOException("HTTP $code")
            }
            job.done = have
            c.inputStream.use { input ->
                FileOutputStream(job.file, append).use { out ->
                    val buf = ByteArray(256 * 1024)
                    var windowStart = SystemClock.elapsedRealtime()
                    var windowBytes = 0L
                    while (true) {
                        if (job.stop) return false
                        val n = input.read(buf)
                        if (n < 0) break
                        out.write(buf, 0, n)
                        job.done += n
                        windowBytes += n
                        val now = SystemClock.elapsedRealtime()
                        if (now - windowStart >= 1000) {
                            job.speed = windowBytes * 1000 / (now - windowStart)
                            windowStart = now
                            windowBytes = 0
                        }
                    }
                }
            }
            if (job.size > 0 && job.done < job.size) throw IOException("connection closed early")
            return true
        } finally {
            c.disconnect()
        }
    }

    private companion object {
        const val MAX_TRIES = 5
    }
}

internal const val USER_AGENT = "NekoChat"
