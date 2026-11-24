// SPDX-FileCopyrightText: 2026 LinuxKernel44
// SPDX-License-Identifier: GPL-3.0-or-later

package com.nekochat.download

import android.content.Context
import android.os.Build
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import androidx.annotation.RequiresApi
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/**
 * The user's Hugging Face access token, for gated and private models.
 *
 * It is encrypted with AES-256-GCM under a key that lives in the Android Keystore (the key never
 * leaves the secure hardware or keystore daemon and is not exported by backups), and only the
 * ciphertext is written to the app's private, backup-excluded storage. It survives app restarts
 * until the user removes or replaces it. Nothing here logs or returns the token other than
 * [current], which only the download code calls, and [HubHttp] only ever sends it to huggingface.co.
 *
 * Needs Android 6.0 (API 23) for AES keys in the Keystore; below that [supported] is false and no
 * token is ever stored, rather than falling back to weaker storage.
 */
object HubToken {
    private const val KEYSTORE = "AndroidKeyStore"
    private const val ALIAS = "nekochat_hf_token"
    private const val FILE = "hf_token.bin"
    private const val GCM_TAG_BITS = 128
    private val FORMAT = Regex("hf_[A-Za-z0-9]{10,200}")

    val supported: Boolean get() = Build.VERSION.SDK_INT >= Build.VERSION_CODES.M

    private val lock = Any()
    private var file: File? = null
    private var cached: String? = null
    private var loaded = false

    /** Idempotent; call before anything else (the app and the download service both do). */
    fun init(context: Context) = synchronized(lock) {
        if (file == null) file = File(context.applicationContext.noBackupFilesDir, FILE)
    }

    /** True when [raw] looks like a Hugging Face token (hf_ followed by letters and digits). */
    fun isWellFormed(raw: String): Boolean = FORMAT.matches(raw.trim())

    /** The stored token, or null when none is saved (or it can no longer be decrypted). */
    fun current(): String? = synchronized(lock) {
        if (!loaded) {
            cached = readLocked()
            loaded = true
        }
        cached
    }

    /** The token with all but its last four characters hidden, e.g. "hf_••••a1b2"; null when none. */
    fun masked(): String? = current()?.let { "hf_••••" + it.takeLast(4) }

    /** Encrypts and stores [raw], replacing any token already saved. */
    fun save(raw: String) = synchronized(lock) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.M) throw IllegalStateException("Needs Android 6.0")
        val token = raw.trim()
        require(isWellFormed(token)) { "not a Hugging Face token" }
        val target = file ?: throw IllegalStateException("HubToken.init was not called")
        val cipher = Cipher.getInstance("AES/GCM/NoPadding").apply { init(Cipher.ENCRYPT_MODE, key(create = true)) }
        val iv = cipher.iv
        val sealed = cipher.doFinal(token.toByteArray(Charsets.UTF_8))
        val tmp = File(target.parentFile, "$FILE.tmp")
        tmp.writeBytes(byteArrayOf(iv.size.toByte()) + iv + sealed)
        if (!tmp.renameTo(target)) {
            tmp.delete()
            throw IOException("Couldn't save the token.")
        }
        cached = token
        loaded = true
    }

    /** Forgets the token: deletes the ciphertext and the Keystore key. */
    fun clear() = synchronized(lock) {
        file?.delete()
        runCatching { KeyStore.getInstance(KEYSTORE).apply { load(null) }.deleteEntry(ALIAS) }
        cached = null
        loaded = true
    }

    /**
     * Asks Hugging Face who [raw] belongs to and returns the account name; throws a
     * [DownloadException] saying what's wrong when it is rejected or unreachable. Blocking.
     */
    fun verify(raw: String): String {
        val token = raw.trim()
        if (!isWellFormed(token)) throw DownloadException("That doesn't look like a Hugging Face token. They start with hf_.")
        val c = try {
            HubHttp.open("https://huggingface.co/api/whoami-v2", token) {
                connectTimeout = 15_000
                readTimeout = 30_000
            }
        } catch (e: IOException) {
            throw DownloadException("Couldn't reach Hugging Face. Check your internet connection.")
        }
        try {
            when (val code = c.responseCode) {
                200 -> return JSONObject(c.inputStream.bufferedReader().use { it.readText() }).optString("name").ifBlank { "your account" }
                401, 403 -> throw DownloadException("Hugging Face didn't accept this token. Check that you copied all of it and that it hasn't been revoked.")
                else -> throw DownloadException("Hugging Face returned HTTP $code. Try again later.")
            }
        } catch (e: IOException) {
            throw DownloadException("Couldn't reach Hugging Face. Check your internet connection.")
        } finally {
            c.disconnect()
        }
    }

    private fun readLocked(): String? {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.M) return null
        val source = file?.takeIf { it.exists() } ?: return null
        return try {
            decrypt(source.readBytes())
        } catch (e: Exception) {
            // Key gone (e.g. lock screen removed on some devices) or file damaged: unusable, so drop it.
            source.delete()
            null
        }
    }

    @RequiresApi(Build.VERSION_CODES.M)
    private fun decrypt(blob: ByteArray): String {
        val ivLen = blob[0].toInt()
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.DECRYPT_MODE, key(create = false), GCMParameterSpec(GCM_TAG_BITS, blob, 1, ivLen))
        return String(cipher.doFinal(blob, 1 + ivLen, blob.size - 1 - ivLen), Charsets.UTF_8)
    }

    @RequiresApi(Build.VERSION_CODES.M)
    private fun key(create: Boolean): SecretKey {
        val store = KeyStore.getInstance(KEYSTORE).apply { load(null) }
        (store.getKey(ALIAS, null) as? SecretKey)?.let { return it }
        check(create) { "no key" }
        val spec = KeyGenParameterSpec.Builder(ALIAS, KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
            .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
            .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
            .setKeySize(256)
            .build()
        return KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, KEYSTORE).run {
            init(spec)
            generateKey()
        }
    }
}
