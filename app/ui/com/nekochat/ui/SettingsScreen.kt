package com.nekochat.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import com.nekochat.download.DnsMode
import com.nekochat.download.DownloadEngine
import com.nekochat.engine.WeightPrecision
import com.nekochat.engine.formatBytes
import androidx.compose.ui.text.font.FontWeight
import com.nekochat.ui.theme.Neko
import com.nekochat.ui.theme.NekoRadius

/** App settings: everything Setup asked for, changeable any time, plus downloads and About. */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun SettingsScreen(vm: ChatViewModel) {
    val load by vm.load.collectAsState()
    Column(
        Modifier
            .fillMaxSize()
            .systemBarsPadding()
            .imePadding()
            .padding(horizontal = 12.dp, vertical = 8.dp),
    ) {
        GlassTopBar("Settings", null, onBack = vm::back, backLabel = "Back to chats")
        Column(
            Modifier
                .weight(1f)
                .verticalScroll(rememberScrollState())
                .padding(top = 12.dp, bottom = 16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            SettingsGroup("Profile") {
                OutlinedTextField(
                    value = vm.username,
                    onValueChange = vm::updateUsername,
                    label = { Text("Your name") },
                    singleLine = true,
                    isError = vm.username.isBlank(),
                    supportingText = {
                        Text(if (vm.username.isBlank()) "Neko needs a name to call you" else "Neko calls you this in chats",
                            fontSize = 12.sp)
                    },
                    shape = RoundedCornerShape(NekoRadius.Large),
                    colors = glassFieldColors(),
                    keyboardOptions = KeyboardOptions(capitalization = KeyboardCapitalization.Words, imeAction = ImeAction.Done),
                    modifier = Modifier.fillMaxWidth(),
                )
            }

            SettingsGroup("Model") {
                val status = loadSummary(load)
                if (status.isNotEmpty()) {
                    Text(status, color = Neko.TextSecondary, fontSize = 13.sp)
                    Spacer(Modifier.height(12.dp))
                }
                if (load is LoadState.Ready) {
                    MemorySection(vm, (load as LoadState.Ready).info.weights)
                    Spacer(Modifier.height(16.dp))
                }
                ModelsSection(vm)
            }

            SettingsGroup("Compute backend") {
                BackendSection(vm)
            }

            SettingsGroup("Weight precision") {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    WeightPrecision.entries.forEach { p ->
                        Chip(p.label, selected = vm.precision == p) { vm.selectPrecision(p) }
                    }
                }
                Text(
                    vm.precision.detail + " Changing it reloads the model.",
                    color = Neko.TextMuted,
                    fontSize = 12.sp,
                    lineHeight = 17.sp,
                    modifier = Modifier.padding(top = 8.dp),
                )
            }

            SettingsGroup("Appearance") {
                NavRow("Theme", vm.theme.label, onClick = vm::openThemes)
            }

            SettingsGroup("Downloads") {
                Text("Downloader", color = Neko.Text, fontSize = 15.sp)
                Spacer(Modifier.height(8.dp))
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    DownloadEngine.entries.forEach { e ->
                        Chip(e.label, selected = vm.downloadEngine == e) { vm.selectDownloadEngine(e) }
                    }
                }
                Text(
                    vm.downloadEngine.detail + " Applies to new downloads; running ones keep their downloader.",
                    color = Neko.TextMuted,
                    fontSize = 12.sp,
                    lineHeight = 17.sp,
                    modifier = Modifier.padding(top = 8.dp),
                )

                // DNS only matters for aria2; Fetch always uses the system resolver.
                if (vm.downloadEngine != DownloadEngine.Fetch) {
                    Spacer(Modifier.height(16.dp))
                    Text("DNS for aria2", color = Neko.Text, fontSize = 15.sp)
                    Spacer(Modifier.height(8.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        DnsMode.entries.forEach { m -> Chip(m.label, selected = vm.dnsMode == m) { vm.selectDnsMode(m) } }
                    }
                    Text(
                        when (vm.dnsMode) {
                            DnsMode.System -> "Uses your phone's DNS, including Private DNS and VPNs. Recommended."
                            DnsMode.Builtin -> "aria2 looks up hosts itself via Cloudflare, Google and Quad9. Try this if " +
                                "downloads fail to find huggingface.co on your network."
                        },
                        color = Neko.TextMuted,
                        fontSize = 12.sp,
                        lineHeight = 17.sp,
                        modifier = Modifier.padding(top = 8.dp),
                    )
                }
            }

            SettingsGroup("About") {
                NavRow("About NekoChat", "Version, license and open-source components", onClick = vm::openAbout)
            }
        }
    }
}

/** Live memory of the loaded model: total first, then what it is made of. */
@Composable
private fun MemorySection(vm: ChatViewModel, weightsLabel: String) {
    val m by vm.memory.collectAsState()
    val usage = m ?: return
    Column(Modifier.fillMaxWidth().innerCard().padding(14.dp)) {
        Text("Memory in use", color = Neko.TextSecondary, fontSize = 13.sp)
        Text(formatBytes(usage.model), color = Neko.Text, fontSize = 22.sp, fontWeight = FontWeight.SemiBold)
        Spacer(Modifier.height(8.dp))
        MemoryRow("Weights ($weightsLabel)", formatBytes(usage.weights))
        MemoryRow(
            "Chat memory (KV cache)",
            if (usage.kvResident > usage.kvUsed) "${formatBytes(usage.kvResident)} reserved"
            else "${formatBytes(usage.kvUsed)} of ${formatBytes(usage.kvCapacity)}",
        )
        MemoryRow("Working buffers", formatBytes(usage.workspace))
        if (usage.appResident > 0) MemoryRow("Whole app (RAM)", formatBytes(usage.appResident))
    }
}

@Composable
private fun MemoryRow(label: String, value: String) {
    Row(Modifier.fillMaxWidth().padding(vertical = 2.dp)) {
        Text(label, color = Neko.TextMuted, fontSize = 13.sp, modifier = Modifier.weight(1f))
        Text(value, color = Neko.TextSecondary, fontSize = 13.sp)
    }
}
