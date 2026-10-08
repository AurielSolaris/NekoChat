package com.nekochat.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.PlayArrow
import androidx.compose.material.icons.rounded.Warning
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.nekochat.download.DownloadStatus
import com.nekochat.engine.WeightPrecision
import com.nekochat.engine.formatBytes
import com.nekochat.speech.DictationModel
import com.nekochat.speech.KokoroVoices
import com.nekochat.speech.SpeechChoice
import com.nekochat.ui.theme.Neko
import com.nekochat.ui.theme.Spectral
import com.nekochat.ui.theme.NekoRadius

const val KOKORO_SIZE_LABEL = "340 MB"

/** Read aloud, narration and the voice: Kokoro on the phone, or the phone's own text-to-speech. */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun AccessibilityScreen(vm: ChatViewModel) {
    Column(
        Modifier
            .fillMaxSize()
            .systemBarsPadding()
            .padding(horizontal = 12.dp, vertical = 8.dp),
    ) {
        GlassTopBar("Accessibility", "Read aloud, narration, voice input", onBack = vm::back, backLabel = "Back")
        Column(
            Modifier
                .weight(1f)
                .verticalScroll(rememberScrollState())
                .padding(top = 12.dp, bottom = 16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            SettingsGroup("Read aloud") {
                ToggleRow("Read replies aloud", "Neko speaks each reply when it's finished.", vm.readReplies,
                    vm::updateReadReplies)
                ToggleRow(
                    "Narrate actions",
                    "Announces screens and actions, like “Message sent” or “Chat deleted”. TalkBack already " +
                        "reads the whole screen, so TalkBack users may not need this.",
                    vm.narrate,
                    vm::updateNarrate,
                )
            }

            SettingsGroup("Voice engine") {
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    SpeechChoice.entries.forEach { c ->
                        Chip(c.label, selected = vm.speechChoice == c) { vm.selectSpeechChoice(c) }
                    }
                }
                Text(
                    when (vm.speechChoice) {
                        SpeechChoice.Auto -> "Kokoro when it's downloaded, otherwise the phone's voice."
                        SpeechChoice.Kokoro -> "Kokoro runs on this phone's CPU: private, offline, natural sounding."
                        SpeechChoice.System -> "The text-to-speech engine installed on this phone."
                    },
                    color = Neko.TextMuted,
                    fontSize = 12.sp,
                    lineHeight = 17.sp,
                    modifier = Modifier.padding(top = 8.dp),
                )
                if (!vm.systemVoiceAvailable) {
                    Notice("This phone has no text-to-speech engine (common on de-Googled phones). Kokoro works without one.")
                }
            }

            SettingsGroup("Kokoro voice") { KokoroSection(vm) }

            SettingsGroup("Voice input") { DictationSection(vm) }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun KokoroSection(vm: ChatViewModel) {
    val download by vm.voiceDownload.collectAsState()
    if (!vm.kokoroInstalled) {
        Text(
            "Kokoro-82M reads replies in a natural voice, entirely on this phone. English (US and UK), " +
                "28 voices. One download of about $KOKORO_SIZE_LABEL; no internet needed afterwards.",
            color = Neko.TextSecondary,
            fontSize = 13.sp,
            lineHeight = 19.sp,
        )
        Spacer(Modifier.height(12.dp))
        VoiceDownloadStatus(vm, download)
        return
    }
    Text("Voice", color = Neko.Text, fontSize = 15.sp)
    for (accent in listOf("US", "UK")) {
        Text(accent, color = Neko.TextMuted, fontSize = 12.sp, modifier = Modifier.padding(top = 8.dp, bottom = 4.dp))
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            KokoroVoices.all.filter { it.accent == accent }.forEach { v ->
                Chip("${v.name} ${if (v.female) "♀" else "♂"}", selected = vm.speechVoice.id == v.id) { vm.selectVoice(v) }
            }
        }
    }
    Spacer(Modifier.height(14.dp))
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text("Speed", color = Neko.Text, fontSize = 15.sp, modifier = Modifier.weight(1f))
        Text("%.1f×".format(vm.speechSpeed), color = Neko.Accent, fontSize = 14.sp, fontWeight = FontWeight.SemiBold)
    }
    Slider(
        value = vm.speechSpeed,
        onValueChange = vm::updateSpeechSpeed,
        valueRange = 0.5f..2f,
        steps = 14,
        colors = SliderDefaults.colors(thumbColor = Neko.Accent, activeTrackColor = Neko.Accent,
            inactiveTrackColor = Neko.Control, activeTickColor = Color.Transparent, inactiveTickColor = Color.Transparent),
    )
    Spacer(Modifier.height(6.dp))
    Text("Precision", color = Neko.Text, fontSize = 15.sp)
    Spacer(Modifier.height(8.dp))
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        WeightPrecision.entries.forEach { p -> Chip(p.label, selected = vm.speechPrecision == p) { vm.selectSpeechPrecision(p) } }
    }
    Text(
        "A voice doesn't need full precision: FP8 (default) keeps it clear with less memory; FP4 uses the least. " +
            "Applies the next time Kokoro speaks." + vm.speech.weightBytes.takeIf { it > 0 }?.let { " Now using ${formatBytes(it)}." }.orEmpty(),
        color = Neko.TextMuted,
        fontSize = 12.sp,
        lineHeight = 17.sp,
        modifier = Modifier.padding(top = 8.dp),
    )
    Spacer(Modifier.height(12.dp))
    SecondaryButton("Play a sample", icon = Icons.Rounded.PlayArrow, modifier = Modifier.fillMaxWidth(), onClick = vm::playSample)
}

@Composable
fun VoiceDownloadStatus(vm: ChatViewModel, download: com.nekochat.download.ModelDownload?) {
    when (download?.status) {
        DownloadStatus.Preparing, DownloadStatus.Downloading, DownloadStatus.Paused -> {
            LinearProgressIndicator(
                progress = { download.progress },
                modifier = Modifier.fillMaxWidth().height(6.dp).clip(RoundedCornerShape(3.dp)),
                color = Neko.Accent,
                trackColor = Neko.Control,
                drawStopIndicator = {},
            )
            Text(
                if (download.status == DownloadStatus.Paused) "Paused · ${formatBytes(download.bytesDone)} of ${formatBytes(download.bytesTotal)}"
                else "Downloading Kokoro · ${formatBytes(download.bytesDone)} of ${formatBytes(download.bytesTotal)}",
                color = Neko.TextSecondary,
                fontSize = 13.sp,
                modifier = Modifier.padding(top = 6.dp),
            )
            if (download.status == DownloadStatus.Paused) {
                Spacer(Modifier.height(8.dp))
                Chip("Resume", selected = false) { vm.downloads.resume(download.repoId) }
            }
        }
        DownloadStatus.Failed -> {
            Notice(download.error ?: "The download failed.")
            Spacer(Modifier.height(8.dp))
            Chip("Retry", selected = false) { vm.downloads.resume(download.repoId) }
        }
        else -> PrimaryButton("Download Kokoro ($KOKORO_SIZE_LABEL)", enabled = true, modifier = Modifier.fillMaxWidth(),
            onClick = vm::downloadVoice)
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun DictationSection(vm: ChatViewModel) {
    val downloads by vm.dictationDownloads.collectAsState()
    Text(
        "Speak instead of typing: tap the mic in a chat, talk, tap again. Granite Speech turns it into text on " +
            "this phone; the recording never leaves it. English only.",
        color = Neko.TextSecondary,
        fontSize = 13.sp,
        lineHeight = 19.sp,
    )
    Spacer(Modifier.height(12.dp))
    Text("Model", color = Neko.Text, fontSize = 15.sp)
    Spacer(Modifier.height(8.dp))
    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        DictationModel.entries.forEach { m ->
            Chip(m.label, selected = vm.dictationModel == m) { vm.selectDictationModel(m) }
        }
    }
    Text(
        (if (vm.dictationReady) "Installed. " else "") + vm.dictationModel.detail,
        color = Neko.TextMuted,
        fontSize = 12.sp,
        lineHeight = 17.sp,
        modifier = Modifier.padding(top = 8.dp),
    )
    Spacer(Modifier.height(12.dp))
    if (vm.dictationReady) {
        Text("Precision", color = Neko.Text, fontSize = 15.sp)
        Spacer(Modifier.height(8.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            WeightPrecision.entries.forEach { p ->
                Chip(p.label, selected = vm.dictationPrecision == p) { vm.selectDictationPrecision(p) }
            }
        }
        Text(
            "FP8 (default) uses half the memory of FP16. The model loads when you tap the mic and leaves memory " +
                "two minutes after you stop." +
                vm.dictationWeightBytes.takeIf { it > 0 }?.let { " Now using ${formatBytes(it)}." }.orEmpty(),
            color = Neko.TextMuted,
            fontSize = 12.sp,
            lineHeight = 17.sp,
            modifier = Modifier.padding(top = 8.dp),
        )
        return
    }
    val download = downloads.lastOrNull { it.repoId == vm.dictationModel.repo }
    when (download?.status) {
        DownloadStatus.Preparing, DownloadStatus.Downloading, DownloadStatus.Paused -> {
            LinearProgressIndicator(
                progress = { download.progress },
                modifier = Modifier.fillMaxWidth().height(6.dp).clip(RoundedCornerShape(3.dp)),
                color = Neko.Accent,
                trackColor = Neko.Control,
                drawStopIndicator = {},
            )
            Text(
                (if (download.status == DownloadStatus.Paused) "Paused" else "Downloading ${vm.dictationModel.label}") +
                    " · ${formatBytes(download.bytesDone)} of ${formatBytes(download.bytesTotal)}",
                color = Neko.TextSecondary,
                fontSize = 13.sp,
                modifier = Modifier.padding(top = 6.dp),
            )
            if (download.status == DownloadStatus.Paused) {
                Spacer(Modifier.height(8.dp))
                Chip("Resume", selected = false) { vm.downloads.resume(download.repoId) }
            }
        }
        DownloadStatus.Failed -> {
            Notice(download.error ?: "The download failed.")
            Spacer(Modifier.height(8.dp))
            Chip("Retry", selected = false) { vm.downloads.resume(download.repoId) }
        }
        else -> PrimaryButton("Download ${vm.dictationModel.label} (${vm.dictationModel.downloadLabel})", enabled = true,
            modifier = Modifier.fillMaxWidth(), onClick = { vm.downloadDictation() })
    }
}

@Composable
fun ToggleRow(title: String, detail: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(
        Modifier
            .fillMaxWidth()
            .heightIn(min = 56.dp)
            .clip(RoundedCornerShape(NekoRadius.Medium))
            .clickable(role = Role.Switch) { onChange(!checked) }
            .padding(vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f).padding(end = 12.dp)) {
            Text(title, color = Neko.Text, fontSize = 15.sp)
            Text(detail, color = Neko.TextMuted, fontSize = 12.sp, lineHeight = 17.sp)
        }
        Switch(
            checked = checked,
            onCheckedChange = null,  // the whole row toggles (bigger target)
            colors = SwitchDefaults.colors(checkedThumbColor = Neko.OnAccent, checkedTrackColor = Neko.Accent,
                uncheckedThumbColor = Neko.TextSecondary, uncheckedTrackColor = Neko.SurfaceElevated,
                uncheckedBorderColor = Neko.Control),
        )
    }
}

@Composable
private fun Notice(text: String) {
    Row(verticalAlignment = Alignment.Top, modifier = Modifier.padding(top = 8.dp)) {
        Icon(Icons.Rounded.Warning, contentDescription = null, tint = Neko.Warning, modifier = Modifier.size(16.dp))
        Text(text, color = Neko.Warning, fontSize = 12.sp, lineHeight = 17.sp, modifier = Modifier.padding(start = 6.dp))
    }
}

/** Pop-up when reading aloud is switched on but nothing can speak yet. */
@Composable
fun SpeechOfferDialog(vm: ChatViewModel) {
    if (!vm.speechOffer) return
    AlertDialog(
        onDismissRequest = vm::dismissSpeechOffer,
        containerColor = Neko.SurfaceElevated,
        shape = RoundedCornerShape(NekoRadius.XLarge),
        title = { Text("Download a voice?", color = Neko.Text, fontSize = 19.sp, fontWeight = FontWeight.SemiBold, fontFamily = Spectral) },
        text = {
            Text(
                (if (vm.systemVoiceAvailable) "Your phone's voice can read replies now. " else "This phone has no text-to-speech engine. ") +
                    "Kokoro is a natural-sounding voice that runs entirely on this phone: one download of about " +
                    "$KOKORO_SIZE_LABEL, then it works offline.",
                color = Neko.TextSecondary,
                fontSize = 14.sp,
                lineHeight = 20.sp,
            )
        },
        dismissButton = {
            TextButton(onClick = vm::dismissSpeechOffer, modifier = Modifier.heightIn(min = 48.dp)) {
                Text(if (vm.systemVoiceAvailable) "Use phone's voice" else "Not now", color = Neko.TextSecondary, fontSize = 15.sp)
            }
        },
        confirmButton = {
            TextButton(onClick = vm::downloadVoice, modifier = Modifier.heightIn(min = 48.dp)) {
                Text("Download Kokoro", color = Neko.Accent, fontSize = 15.sp, fontWeight = FontWeight.SemiBold)
            }
        },
    )
}

/** Pop-up when the mic is tapped before a speech model is downloaded. */
@Composable
fun DictationOfferDialog(vm: ChatViewModel) {
    if (!vm.dictationOffer) return
    val m = vm.dictationModel
    AlertDialog(
        onDismissRequest = vm::dismissDictationOffer,
        containerColor = Neko.SurfaceElevated,
        shape = RoundedCornerShape(NekoRadius.XLarge),
        title = { Text("Download voice input?", color = Neko.Text, fontSize = 19.sp, fontWeight = FontWeight.SemiBold, fontFamily = Spectral) },
        text = {
            Text(
                "Voice input uses ${m.label}, a speech-to-text model that runs entirely on this phone: one download " +
                    "of about ${m.downloadLabel}, then it works offline. ${m.detail} You can switch models in " +
                    "Settings → Accessibility.",
                color = Neko.TextSecondary,
                fontSize = 14.sp,
                lineHeight = 20.sp,
            )
        },
        dismissButton = {
            TextButton(onClick = vm::dismissDictationOffer, modifier = Modifier.heightIn(min = 48.dp)) {
                Text("Not now", color = Neko.TextSecondary, fontSize = 15.sp)
            }
        },
        confirmButton = {
            TextButton(onClick = { vm.downloadDictation() }, modifier = Modifier.heightIn(min = 48.dp)) {
                Text("Download", color = Neko.Accent, fontSize = 15.sp, fontWeight = FontWeight.SemiBold)
            }
        },
    )
}
