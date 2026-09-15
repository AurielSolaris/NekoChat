package com.nekochat.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.provider.Settings
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material.icons.rounded.Warning
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.nekochat.download.DownloadStatus
import com.nekochat.download.ModelDownload
import com.nekochat.engine.formatBytes
import com.nekochat.ui.theme.Neko
import com.nekochat.ui.theme.NekoRadius

private data class Suggestion(val title: String, val repoId: String, val detail: String)

// A recommended default for each supported architecture keeps the choice small (Hick's law).
private val SUGGESTIONS = listOf(
    Suggestion("Qwen3 0.6B", "Qwen/Qwen3-0.6B", "Chat model · follows instructions · 1.4 GB"),
    Suggestion("GPT-2", "openai-community/gpt2", "Classic text model · fastest · 548 MB"),
)

/** Downloads GPT-2 / Qwen3 models from Hugging Face through [com.nekochat.download.ModelDownloadService]. */
@Composable
fun AddModelsScreen(vm: ChatViewModel) {
    val context = LocalContext.current
    val downloads by vm.downloads.downloads.collectAsState()
    var repo by rememberSaveable { mutableStateOf("") }
    var confirmCancel by remember { mutableStateOf<ModelDownload?>(null) }
    val notifications = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { }
    val start = { id: String ->
        // Progress shows in a notification while NekoChat is in the background; the download works either way.
        if (Build.VERSION.SDK_INT >= 33 &&
            ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) {
            notifications.launch(Manifest.permission.POST_NOTIFICATIONS)
        }
        vm.downloads.start(id)
    }

    Column(
        Modifier
            .fillMaxSize()
            .systemBarsPadding()
            .imePadding()
            .padding(horizontal = 12.dp, vertical = 8.dp),
    ) {
        GlassTopBar("Download models", "From Hugging Face", onBack = vm::back)
        Column(
            Modifier
                .weight(1f)
                .verticalScroll(rememberScrollState())
                .padding(top = 12.dp, bottom = 16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            if (downloads.isNotEmpty()) {
                SettingsGroup("Downloads") {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        downloads.asReversed().forEach { d ->
                            DownloadRow(
                                d,
                                onPause = { vm.downloads.pause(d.repoId) },
                                onResume = { vm.downloads.resume(d.repoId) },
                                onCancel = { confirmCancel = d },
                                onDismiss = { vm.downloads.dismiss(d.repoId) },
                                onUse = { d.folder?.let(vm::useDownloadedModel) },
                                onOpenNetworkSettings = {
                                    // App info is where "Mobile data & Wi-Fi → Allow network access" lives.
                                    context.startActivity(
                                        Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.fromParts("package", context.packageName, null)),
                                    )
                                },
                            )
                        }
                    }
                }
            }

            SettingsGroup("Recommended") {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    SUGGESTIONS.forEach { s ->
                        val existing = downloads.firstOrNull { it.repoId.equals(s.repoId, ignoreCase = true) }
                        Row(
                            Modifier.fillMaxWidth().innerCard().padding(start = 14.dp, top = 10.dp, bottom = 10.dp, end = 10.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Column(Modifier.weight(1f)) {
                                Text(s.title, color = Neko.Text, fontSize = 15.sp, fontWeight = FontWeight.Medium)
                                Text(s.detail, color = Neko.TextSecondary, fontSize = 12.sp)
                                Text(s.repoId, color = Neko.TextMuted, fontSize = 12.sp)
                            }
                            Spacer(Modifier.width(8.dp))
                            when (existing?.status) {
                                null -> Chip("Download", selected = false) { start(s.repoId) }
                                DownloadStatus.Completed -> Text("Installed", color = Neko.Success, fontSize = 13.sp,
                                    fontWeight = FontWeight.Medium)
                                else -> Text("In downloads", color = Neko.TextMuted, fontSize = 13.sp)
                            }
                        }
                    }
                }
            }

            SettingsGroup("Any GPT-2 or Qwen3 model") {
                OutlinedTextField(
                    value = repo,
                    onValueChange = { repo = it.trim().take(120) },
                    label = { Text("Hugging Face model") },
                    placeholder = { Text("owner/model, e.g. Qwen/Qwen3-1.7B", color = Neko.TextMuted) },
                    singleLine = true,
                    shape = RoundedCornerShape(NekoRadius.Large),
                    colors = glassFieldColors(),
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri, imeAction = ImeAction.Go),
                    keyboardActions = KeyboardActions(onGo = {
                        if (repo.isNotBlank()) {
                            start(repo)
                            repo = ""
                        }
                    }),
                    modifier = Modifier.fillMaxWidth(),
                )
                Spacer(Modifier.height(10.dp))
                PrimaryButton("Download", enabled = repo.isNotBlank(), modifier = Modifier.fillMaxWidth()) {
                    start(repo)
                    repo = ""
                }
                Text(
                    "NekoChat checks the model type first and only downloads GPT-2 and Qwen3 models " +
                        "(SafeTensors or PyTorch). Downloads use several connections and resume where they stopped.",
                    color = Neko.TextMuted,
                    fontSize = 12.sp,
                    lineHeight = 17.sp,
                    modifier = Modifier.padding(top = 8.dp),
                )
            }
        }
    }

    confirmCancel?.let { d ->
        ConfirmDialog(
            title = "Cancel download?",
            message = "The ${formatBytes(d.bytesDone)} downloaded so far for ${d.name} will be deleted. " +
                "To keep it and continue later, use Pause instead.",
            confirmLabel = "Cancel download",
            onConfirm = { vm.downloads.cancel(d.repoId) },
            onDismiss = { confirmCancel = null },
        )
    }
}

@Composable
private fun DownloadRow(
    d: ModelDownload,
    onPause: () -> Unit,
    onResume: () -> Unit,
    onCancel: () -> Unit,
    onDismiss: () -> Unit,
    onUse: () -> Unit,
    onOpenNetworkSettings: () -> Unit,
) {
    Column(Modifier.fillMaxWidth().innerCard().padding(14.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text(d.name, color = Neko.Text, fontSize = 15.sp, fontWeight = FontWeight.Medium, maxLines = 1,
                    overflow = TextOverflow.Ellipsis)
                Text("${d.repoId} · ${d.engine.label}", color = Neko.TextMuted, fontSize = 12.sp, maxLines = 1,
                    overflow = TextOverflow.Ellipsis)
            }
            if (d.status == DownloadStatus.Completed) {
                IconButton(onClick = onDismiss) {
                    Icon(Icons.Rounded.Close, contentDescription = "Remove from list", tint = Neko.TextSecondary)
                }
            }
        }
        when (d.status) {
            DownloadStatus.Preparing -> StatusText("Checking the model on Hugging Face…")
            DownloadStatus.Downloading, DownloadStatus.Paused -> {
                Spacer(Modifier.height(8.dp))
                LinearProgressIndicator(
                    progress = { d.progress },
                    modifier = Modifier.fillMaxWidth().height(6.dp).clip(RoundedCornerShape(3.dp)),
                    color = if (d.status == DownloadStatus.Paused) Neko.TextMuted else Neko.Accent,
                    trackColor = Neko.Border,
                    drawStopIndicator = {},
                )
                StatusText(
                    "${formatBytes(d.bytesDone)} of ${formatBytes(d.bytesTotal)} · " +
                        if (d.status == DownloadStatus.Paused) "Paused" else "${formatBytes(d.bytesPerSecond)}/s",
                )
            }
            DownloadStatus.Completed -> StatusText("Installed · ${formatBytes(d.bytesTotal)}", color = Neko.Success)
            DownloadStatus.Failed -> Row(verticalAlignment = Alignment.Top, modifier = Modifier.padding(top = 6.dp)) {
                Icon(Icons.Rounded.Warning, contentDescription = "Error", tint = Neko.Error, modifier = Modifier.size(16.dp))
                Text(d.error ?: "Download failed.", color = Neko.Error, fontSize = 13.sp, maxLines = 3,
                    overflow = TextOverflow.Ellipsis, modifier = Modifier.padding(start = 6.dp))
            }
        }
        val actions: List<@Composable () -> Unit> = when (d.status) {
            DownloadStatus.Preparing -> emptyList()
            DownloadStatus.Downloading -> listOf({ Chip("Pause", selected = false, onClick = onPause) },
                { Chip("Cancel", selected = false, destructive = true, onClick = onCancel) })
            DownloadStatus.Paused -> listOf({ Chip("Resume", selected = false, onClick = onResume) },
                { Chip("Cancel", selected = false, destructive = true, onClick = onCancel) })
            DownloadStatus.Failed -> listOfNotNull(
                if (d.networkBlocked) ({ Chip("Open network settings", selected = true, onClick = onOpenNetworkSettings) }) else null,
                { Chip("Retry", selected = false, onClick = onResume) },
                { Chip("Remove", selected = false, destructive = true, onClick = onCancel) },
            )
            DownloadStatus.Completed -> listOf({ Chip("Use this model", selected = true, onClick = onUse) })
        }
        if (actions.isNotEmpty()) {
            Spacer(Modifier.height(10.dp))
            // Routine actions first; the destructive one last and apart (Fitts: keep it away from Pause/Resume).
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                actions.first()()
                Spacer(Modifier.weight(1f))
                actions.drop(1).forEach { it() }
            }
        }
    }
}

@Composable
private fun StatusText(text: String, color: androidx.compose.ui.graphics.Color = Neko.TextSecondary) {
    Text(text, color = color, fontSize = 13.sp, modifier = Modifier.padding(top = 6.dp))
}
