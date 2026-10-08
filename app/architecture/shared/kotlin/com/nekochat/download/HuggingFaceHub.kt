package com.nekochat.download

import android.net.Uri
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.net.UnknownHostException

/** Resolves which files of a Hugging Face model repository NekoChat needs. Blocking; call off the main thread. */
object HuggingFaceHub {
    private const val BASE = "https://huggingface.co"
    private val SUPPORT_FILES = setOf(
        "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    )

    data class RepoFile(val path: String, val size: Long)

    fun resolveUrl(repo: String, path: String): String =
        "$BASE/$repo/resolve/main/" + path.split('/').joinToString("/") { Uri.encode(it) }

    /** Config + tokenizer files and the weights the native loader will use (mirrors its choice). */
    fun modelFiles(repo: String): List<RepoFile> {
        val arr = JSONArray(get("$BASE/api/models/$repo/tree/main"))
        val files = (0 until arr.length()).map { arr.getJSONObject(it) }
            .filter { it.optString("type") == "file" && '/' !in it.optString("path") }
            .map { RepoFile(it.getString("path"), it.optLong("size")) }
        val names = files.map { it.path }.toSet()
        if ("config.json" !in names) throw DownloadException("This repository has no config.json, so it isn't a model NekoChat can run.")
        if ("tokenizer.json" !in names && !("vocab.json" in names && "merges.txt" in names)) {
            throw DownloadException("This repository has no tokenizer files.")
        }
        val safetensors = files.filter { it.path.endsWith(".safetensors") }
        val weights = safetensors.ifEmpty {
            files.filter { f -> listOf(".pt", ".pth", ".bin").any { f.path.endsWith(it) } }.sortedBy { it.path }.take(1)
        }
        if (weights.isEmpty()) throw DownloadException("No .safetensors or .pt weights in this repository.")
        return files.filter { it.path in SUPPORT_FILES } + weights
    }

    /** Kokoro: config, weights and the English voice packs (voices/<id>.pt). */
    fun speechFiles(repo: String, voices: Collection<String>): List<RepoFile> {
        fun list(path: String): List<RepoFile> {
            val arr = JSONArray(get("$BASE/api/models/$repo/tree/main$path"))
            return (0 until arr.length()).map { arr.getJSONObject(it) }
                .filter { it.optString("type") == "file" }
                .map { RepoFile(it.getString("path"), it.optLong("size")) }
        }
        val root = list("").filter { it.path == "config.json" || it.path == "kokoro-v1_0.pth" }
        if (root.size != 2) throw DownloadException("This repository doesn't look like Kokoro-82M.")
        val packs = list("/voices").filter { f -> f.path.removePrefix("voices/").removeSuffix(".pt") in voices }
        if (packs.isEmpty()) throw DownloadException("No English voices found in this repository.")
        return root + packs
    }

    /** A speech-to-text model for voice input: exactly [files], which must all exist. */
    fun dictationFiles(repo: String, files: List<String>): List<RepoFile> {
        val arr = JSONArray(get("$BASE/api/models/$repo/tree/main"))
        val all = (0 until arr.length()).map { arr.getJSONObject(it) }
            .filter { it.optString("type") == "file" }
            .associate { it.getString("path") to it.optLong("size") }
        val missing = files.filter { it !in all }
        if (missing.isNotEmpty()) throw DownloadException("This repository doesn't look like Granite Speech (no ${missing.first()}).")
        return files.map { RepoFile(it, all.getValue(it)) }
    }

    /** The repository's config.json (empty when it has none or it isn't JSON). */
    fun config(repo: String): JSONObject =
        runCatching { JSONObject(get(resolveUrl(repo, "config.json"))) }
            .getOrElse { if (it is DownloadException) throw it else JSONObject() }

    private fun get(url: String): String {
        val c = try {
            (URL(url).openConnection() as HttpURLConnection).apply {
                connectTimeout = 15_000
                readTimeout = 30_000
                setRequestProperty("User-Agent", "NekoChat")
            }
        } catch (e: IOException) {
            throw DownloadException("Couldn't reach Hugging Face. Check your internet connection.")
        }
        try {
            when (val code = c.responseCode) {
                200 -> return c.inputStream.bufferedReader().use { it.readText() }
                401, 403 -> throw DownloadException("This model is gated or private. NekoChat can only download public models.")
                404 -> throw DownloadException("Model not found on Hugging Face. Check the name, e.g. Qwen/Qwen3-0.6B.")
                else -> throw DownloadException("Hugging Face returned HTTP $code. Try again later.")
            }
        } catch (e: UnknownHostException) {
            // Android prints no stack trace for this one, so log the message explicitly.
            Log.w("NekoChat", "Hugging Face DNS lookup failed: $url ($e)")
            throw DownloadException(
                "NekoChat couldn't look up huggingface.co. Check your connection, and that no VPN, firewall " +
                    "or data-saver setting blocks NekoChat.",
            )
        } catch (e: IOException) {
            Log.w("NekoChat", "Hugging Face request failed: $url ($e)", e)
            throw DownloadException("Couldn't reach Hugging Face. Check your internet connection.")
        } finally {
            c.disconnect()
        }
    }
}
