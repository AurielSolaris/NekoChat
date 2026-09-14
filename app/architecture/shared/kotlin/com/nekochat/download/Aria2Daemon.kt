package com.nekochat.download

import android.content.Context
import android.os.Process as AndroidProcess
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.ServerSocket
import java.net.URL
import java.util.UUID
import java.util.concurrent.atomic.AtomicLong
import kotlin.concurrent.thread

/** A running aria2c process and its JSON-RPC endpoint (127.0.0.1 only, random port and secret). */
internal class Aria2Daemon private constructor(private val process: Process, val rpc: Rpc, private val log: ArrayDeque<String>) {

    val alive: Boolean
        get() = try {
            process.exitValue()
            false
        } catch (_: IllegalThreadStateException) {
            true
        }

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
            if (!bin.exists()) throw DownloadException("The downloader isn't available on this device.")
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
                "--stop-with-process=${AndroidProcess.myPid()}",  // never outlive the app
                "--dir=${defaultDir.path}",
                "--continue=true",
                "--allow-overwrite=true",
                "--auto-file-renaming=false",
                "--file-allocation=none",
                "--max-concurrent-downloads=3",
                "--max-connection-per-server=8",
                "--split=8",
                "--min-split-size=8M",
                "--max-tries=5",
                "--retry-wait=3",
                "--connect-timeout=20",
                "--ca-certificate=${caBundle(home).path}",
                "--check-certificate=true",
                "--user-agent=NekoChat",
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
                    val tail = synchronized(log) { log.joinToString("\n") }
                    throw DownloadException("The downloader failed to start. $tail".trim())
                }
                if (runCatching { d.rpc.call("aria2.getVersion") }.isSuccess) return d
                Thread.sleep(100)
            }
            d.close()
            throw DownloadException("The downloader didn't respond.")
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
