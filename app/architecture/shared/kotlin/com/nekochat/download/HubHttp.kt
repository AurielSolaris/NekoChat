// SPDX-FileCopyrightText: 2026 LinuxKernel44
// SPDX-License-Identifier: GPL-3.0-or-later

package com.nekochat.download

import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

/**
 * HTTP for Hugging Face with the user's [HubToken], if any. Redirects are followed by hand so the
 * token can be attached to requests for huggingface.co and to nothing else: file downloads are
 * redirected to a CDN on another domain, and that hop (and anything after it) never sees the token.
 */
internal object HubHttp {
    private const val HUB_HOST = "huggingface.co"
    private const val MAX_REDIRECTS = 6
    private val REDIRECTS = setOf(301, 302, 303, 307, 308)

    private fun isHub(url: URL) = url.protocol == "https" && url.host.equals(HUB_HOST, ignoreCase = true)

    /** One request, redirects not followed. The token is added only when [url] is on huggingface.co. */
    private fun hop(url: URL, token: String?, method: String, configure: HttpURLConnection.() -> Unit): HttpURLConnection =
        (url.openConnection() as HttpURLConnection).apply {
            instanceFollowRedirects = false
            requestMethod = method
            setRequestProperty("User-Agent", USER_AGENT)
            if (token != null && isHub(url)) setRequestProperty("Authorization", "Bearer $token")
            configure()
        }

    private fun nextUrl(from: URL, c: HttpURLConnection): URL {
        val location = c.getHeaderField("Location")
        c.disconnect()
        val next = URL(from, location ?: throw IOException("Redirect without a Location"))
        if (next.protocol != "https") throw IOException("Refusing a redirect to ${next.protocol}")
        return next
    }

    /**
     * Opens [url] and follows redirects, returning the connection whose response is not a redirect.
     * [configure] (timeouts, Range, ...) is applied to every hop. The caller disconnects.
     */
    fun open(
        url: String,
        token: String? = HubToken.current(),
        method: String = "GET",
        configure: HttpURLConnection.() -> Unit = {},
    ): HttpURLConnection {
        var target = URL(url)
        repeat(MAX_REDIRECTS + 1) {
            val c = hop(target, token, method, configure)
            if (c.responseCode !in REDIRECTS) return c
            target = nextUrl(target, c)
        }
        throw IOException("Too many redirects")
    }

    /**
     * A URL a third-party downloader (aria2) can fetch without any credentials: for a file that
     * Hugging Face redirects to its CDN, the CDN's signed URL; the URL itself when no token is set.
     * null when huggingface.co serves the file itself, which needs the token, so the caller must
     * download it in-process. Throws [DownloadException] when Hugging Face denies access.
     */
    fun anonymousUrl(url: String): String? {
        val token = HubToken.current() ?: return url
        var target = URL(url)
        repeat(MAX_REDIRECTS + 1) {
            if (!isHub(target)) return target.toString()
            val c = hop(target, token, "HEAD") {
                connectTimeout = 20_000
                readTimeout = 30_000
            }
            val code = c.responseCode
            if (code in REDIRECTS) {
                target = nextUrl(target, c)
                return@repeat
            }
            c.disconnect()
            when (code) {
                200 -> return null
                401, 403 -> throw DownloadException(denied(hadToken = true))
                404 -> throw DownloadException("File not found on the server (HTTP 404).")
                else -> throw IOException("HTTP $code")
            }
        }
        throw IOException("Too many redirects")
    }

    /** What to tell the user after HTTP 401/403 from Hugging Face. */
    fun denied(hadToken: Boolean = HubToken.current() != null): String =
        if (hadToken) {
            "Hugging Face refused access. If this model is gated, accept its terms on huggingface.co with the " +
                "account your token belongs to. If your token has expired, replace or remove it in Settings → Hugging Face."
        } else {
            "This model is gated or private. Add a Hugging Face access token in Settings → Hugging Face to download it."
        }
}
