package com.nekochat.download

import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.ServerSocket
import java.net.URL
import java.util.UUID
import java.util.concurrent.atomic.AtomicLong
import kotlin.concurrent.thread

/**
 * aria2c (shipped as libaria2c.so) as a [TransferEngine]: several connections per file and
 * resumable transfers via .aria2 control files. One aria2 process serves every download; the
 * connection count is set per file. Not thread safe: called from the download service's thread.
 */
internal class Aria2Engine(private val context: Context, private val defaultDir: File) : TransferEngine {
    /** Resolver used the next time aria2 starts (see [restart]). */
    var dnsMode = DnsMode.System

    private var daemon: Aria2Daemon? = null
    private var generation = 0  // handles are "generation:gid"; a new process invalidates old gids
    private val lastDone = HashMap<String, Long>()

    override fun add(url: String, dir: File, name: String, size: Long, connections: Int): String {
        val d = daemon()
        val options = JSONObject()
            .put("dir", dir.path)
            .put("out", name)
            .put("split", connections.toString())
            .put("max-connection-per-server", connections.toString())
        val gid = d.rpc.call("aria2.addUri", JSONArray().put(url), options) as String
        return "$generation:$gid"
    }

    override fun status(handle: String): FileStatus? {
        val d = daemon ?: return null
        val gid = gidOf(handle) ?: return null
        if (!d.alive) {
            Log.w("NekoChat", "aria2 exited:\n${d.logTail()}")  // details for logcat, not for the UI
            return FileStatus(lastDone[handle] ?: 0, 0, FileState.Error, "The aria2 downloader stopped unexpectedly.")
        }
        val s = runCatching {
            d.rpc.call("aria2.tellStatus", gid, JSONArray(listOf("status", "completedLength", "downloadSpeed", "errorMessage"))) as JSONObject
        }.getOrNull() ?: return FileStatus(lastDone[handle] ?: 0, 0, FileState.Active)  // busy: one missed reply is fine
        val done = s.optString("completedLength").toLongOrNull() ?: 0L
        lastDone[handle] = done
        val state = when (s.optString("status")) {
            "complete" -> FileState.Complete
            "error", "removed" -> FileState.Error
            "paused" -> FileState.Paused
            else -> FileState.Active
        }
        return FileStatus(done, s.optString("downloadSpeed").toLongOrNull() ?: 0L, state,
            if (state == FileState.Error) s.optString("errorMessage").ifBlank { "Download failed." } else null)
    }

    override fun pause(handle: String) {
        gidOf(handle)?.let { gid -> runCatching { daemon?.rpc?.call("aria2.forcePause", gid) } }
    }

    override fun resume(handle: String): Boolean {
        val gid = gidOf(handle) ?: return false
        if (daemon?.alive != true) return false
        return runCatching { daemon?.rpc?.call("aria2.unpause", gid) }.isSuccess
    }

    override fun remove(handle: String) {
        lastDone.remove(handle)
        val gid = gidOf(handle) ?: return
        runCatching { daemon?.rpc?.call("aria2.forceRemove", gid) }
        runCatching { daemon?.rpc?.call("aria2.removeDownloadResult", gid) }
    }

    /** Stops aria2 so the next download starts a fresh process (e.g. with a new resolver). */
    fun restart() {
        daemon?.close()
        daemon = null
    }

    override fun close() = restart()

    private fun gidOf(handle: String): String? {
        val parts = handle.split(':', limit = 2)
        return if (parts.size == 2 && parts[0].toIntOrNull() == generation && daemon != null) parts[1] else null
    }

    private fun daemon(): Aria2Daemon {
        daemon?.takeIf { it.alive }?.let { return it }
        daemon?.close()
        generation++
        return Aria2Daemon.start(context, defaultDir, dnsMode).also { daemon = it }
    }
}

/** A running aria2c process and its JSON-RPC endpoint (127.0.0.1 only, random port and secret). */
internal class Aria2Daemon private constructor(private val process: Process, val rpc: Rpc, private val log: ArrayDeque<String>) {

    val alive: Boolean
        get() = try {
            process.exitValue()
            false
        } catch (_: IllegalThreadStateException) {
            true
        }

    /** Last lines aria2 printed, for error messages. */
    fun logTail(): String = synchronized(log) { log.joinToString("\n") }

    fun close() {
        runCatching { rpc.call("aria2.shutdown") }
        process.destroy()
    }

    class Rpc(private val port: Int, private val secret: String) {
        private val ids = AtomicLong()

        fun call(method: String, vararg params: Any): Any? {
            val args = JSONArray().put("token:$secret")
            params.forEach { args.put(it) }
            val body = JSONObject().put("jsonrpc", "2.0").put("id", ids.incrementAndGet().toString())
                .put("method", method).put("params", args).toString()
            val c = URL("http://127.0.0.1:$port/jsonrpc").openConnection() as HttpURLConnection
            try {
                c.requestMethod = "POST"
                c.connectTimeout = 3000
                c.readTimeout = 10_000
                c.doOutput = true
                c.setRequestProperty("Content-Type", "application/json")
                c.outputStream.use { it.write(body.toByteArray()) }
                val stream = if (c.responseCode < 400) c.inputStream else c.errorStream
                val o = JSONObject(stream.bufferedReader().use { it.readText() })
                o.optJSONObject("error")?.let { throw DownloadException(it.optString("message", "aria2 error")) }
                return o.opt("result")
            } finally {
                c.disconnect()
            }
        }
    }

    companion object {
        fun start(context: Context, defaultDir: File, dns: DnsMode): Aria2Daemon {
            val bin = File(context.applicationInfo.nativeLibraryDir, "libaria2c.so")
            if (!bin.exists()) throw DownloadException("The aria2 downloader isn't available on this device. Try Fetch.")
            val home = File(context.filesDir, "aria2").apply { mkdirs() }
            val port = ServerSocket(0).use { it.localPort }
            val secret = UUID.randomUUID().toString()
            val cmd = listOf(
                bin.path,
                "--no-conf=true",
                "--enable-rpc=true",
                "--rpc-listen-all=false",
                "--rpc-listen-port=$port",
                "--rpc-secret=$secret",
                // No --stop-with-process: the app sandbox hides our pid from aria2, which then shuts itself
                // down within a second. Android kills the app's whole process group (aria2 included) anyway.
                "--dir=${defaultDir.path}",
                "--continue=true",
                "--allow-overwrite=true",
                "--auto-file-renaming=false",
                "--file-allocation=none",
                "--max-concurrent-downloads=3",
                "--min-split-size=8M",
                "--max-tries=5",
                "--retry-wait=3",
                "--connect-timeout=20",
                "--ca-certificate=${caBundle(home).path}",
                "--check-certificate=true",
                "--user-agent=$USER_AGENT",
                "--console-log-level=warn",
                "--summary-interval=0",
            ) + when (dns) {
                // getaddrinfo: Android's resolver, including Private DNS and VPN DNS.
                DnsMode.System -> listOf("--async-dns=false")
                // c-ares with explicit servers (Android has no resolv.conf for it to read).
                DnsMode.Builtin -> listOf("--async-dns=true", "--async-dns-server=1.1.1.1,8.8.8.8,9.9.9.9")
            }
            val process = ProcessBuilder(cmd).directory(home).redirectErrorStream(true)
                .apply { environment()["HOME"] = home.path }
                .start()
            val log = ArrayDeque<String>()
            thread(isDaemon = true, name = "aria2-log") {
                runCatching {
                    process.inputStream.bufferedReader().forEachLine { line ->
                        synchronized(log) {
                            log.addLast(line)
                            if (log.size > 20) log.removeFirst()
                        }
                    }
                }
            }
            val d = Aria2Daemon(process, Rpc(port, secret), log)
            repeat(60) {
                if (!d.alive) {
                    Log.w("NekoChat", "aria2 failed to start:\n${d.logTail()}")
                    throw DownloadException("The aria2 downloader failed to start. Try Fetch in Settings → Downloads.")
                }
                if (runCatching { d.rpc.call("aria2.getVersion") }.isSuccess) return d
                Thread.sleep(100)
            }
            d.close()
            throw DownloadException("The aria2 downloader didn't respond.")
        }

        /** aria2 needs one PEM bundle; Android keeps its trusted roots as one file per certificate. */
        private fun caBundle(home: File): File {
            val out = File(home, "ca-bundle.pem")
            val dir = listOf("/apex/com.android.conscrypt/cacerts", "/system/etc/security/cacerts")
                .map(::File).firstOrNull { it.listFiles()?.isNotEmpty() == true }
                ?: throw DownloadException("No trusted certificates found on this device.")
            out.bufferedWriter().use { w ->
                dir.listFiles()!!.sortedBy { it.name }.forEach { f ->
                    runCatching { w.write(f.readText()); w.write("\n") }
                }
            }
            return out
        }
    }
}
