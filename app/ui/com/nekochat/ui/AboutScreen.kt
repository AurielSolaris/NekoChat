package com.nekochat.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.nekochat.ui.theme.Neko
import com.nekochat.ui.theme.NekoRadius

private data class License(val title: String, val asset: String)

@Composable
fun AboutScreen(vm: ChatViewModel) {
    val context = LocalContext.current
    val version = remember {
        runCatching { context.packageManager.getPackageInfo(context.packageName, 0).versionName }.getOrNull() ?: "?"
    }
    var shown by rememberSaveable { mutableStateOf<String?>(null) }
    val licenses = listOf(
        License("GNU General Public License v3", "licenses/GPL-3.0.txt"),
        License("aria2 (GPL-2.0-or-later)", "licenses/aria2-COPYING.txt"),
        License("OpenSSL License", "licenses/openssl-LICENSE.txt"),
    )

    Column(
        Modifier
            .fillMaxSize()
            .systemBarsPadding()
            .padding(horizontal = 12.dp, vertical = 8.dp),
    ) {
        GlassTopBar("About", null, onBack = vm::back, backLabel = "Back to settings")
        Column(
            Modifier
                .weight(1f)
                .verticalScroll(rememberScrollState())
                .padding(top = 16.dp, bottom = 16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            NekoMascot(112.dp)
            Wordmark()
            Text("Version $version", color = Neko.TextSecondary, fontSize = 14.sp)
            Text("Private AI that runs on your phone", color = Neko.TextMuted, fontSize = 13.sp)
            Spacer(Modifier.height(4.dp))

            SettingsGroup("NekoChat") {
                InfoRow("Models", "GPT-2 and Qwen3 · SafeTensors or PyTorch")
                InfoRow("Engine", "NekoChat's own C++ engine · NEON CPU, Vulkan, OpenGL ES")
                InfoRow("Privacy", "Chats and models stay on this phone. The internet is only used to download models.")
            }

            SettingsGroup("License") {
                Text(
                    "NekoChat is free software: you can redistribute it and/or modify it under the terms of the " +
                        "GNU General Public License, version 3.",
                    color = Neko.TextSecondary, fontSize = 13.sp, lineHeight = 19.sp,
                )
                NavRow(licenses[0].title, "Read the full license") { shown = licenses[0].asset }
            }

            SettingsGroup("Open-source components") {
                NavRow("aria2 1.37.0", "Model downloads · GPL-2.0-or-later") { shown = licenses[1].asset }
                NavRow("OpenSSL, zlib, expat, c-ares, libssh2", "Built into aria2 · OpenSSL License and others") {
                    shown = licenses[2].asset
                }
                InfoRow("Jetpack Compose, Room, Kotlin coroutines", "Apache License 2.0")
            }
        }
    }

    shown?.let { asset -> LicenseDialog(licenses.first { it.asset == asset }.title, asset) { shown = null } }
}

@Composable
private fun InfoRow(title: String, value: String) {
    Column(Modifier.fillMaxWidth().heightIn(min = 48.dp).padding(vertical = 6.dp)) {
        Text(title, color = Neko.Text, fontSize = 15.sp)
        Text(value, color = Neko.TextMuted, fontSize = 12.sp, lineHeight = 17.sp)
    }
}

@Composable
private fun LicenseDialog(title: String, asset: String, onDismiss: () -> Unit) {
    val context = LocalContext.current
    val text = remember(asset) {
        runCatching { context.assets.open(asset).bufferedReader().use { it.readText() } }.getOrDefault("License text unavailable.")
    }
    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor = Neko.SurfaceElevated,
        shape = RoundedCornerShape(NekoRadius.XLarge),
        title = { Text(title, color = Neko.Text, fontSize = 18.sp, fontWeight = FontWeight.SemiBold) },
        text = {
            Column(Modifier.heightIn(max = 460.dp).verticalScroll(rememberScrollState())) {
                Text(text, color = Neko.TextSecondary, fontSize = 11.sp, lineHeight = 15.sp, fontFamily = FontFamily.Monospace)
            }
        },
        confirmButton = {
            Row {
                TextButton(onClick = onDismiss, modifier = Modifier.heightIn(min = 48.dp)) {
                    Text("Close", color = Neko.Accent, fontWeight = FontWeight.SemiBold)
                }
            }
        },
    )
}
