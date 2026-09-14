package com.nekochat.ui

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Add
import androidx.compose.material.icons.rounded.Check
import androidx.compose.material.icons.rounded.Warning
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import app.auriel.realm.nekochat.R
import com.nekochat.engine.ComputeBackend
import com.nekochat.engine.LocalModel
import com.nekochat.engine.formatBytes
import com.nekochat.ui.theme.GlassLevel
import com.nekochat.ui.theme.Neko
import com.nekochat.ui.theme.NekoRadius
import com.nekochat.ui.theme.glass

/** Models folder, the models found in it, and the way to download more. Shared by Setup and Settings. */
@Composable
fun ModelsSection(vm: ChatViewModel) {
    val scanned by vm.models.collectAsState()
    val models = scanned.orEmpty()
    val folderPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri ->
        uri?.let(vm::setModelsFolder)
    }
    SectionLabel("Models folder")
    FolderRow(vm.modelsFolderName) { folderPicker.launch(null) }

    Spacer(Modifier.height(20.dp))
    SectionLabel("Model")
    when {
        scanned == null -> Text("Looking for models…", color = Neko.TextMuted, fontSize = 13.sp)
        models.isEmpty() -> EmptyModels(vm.modelsFolderName)
        else -> Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            models.forEach { m -> ModelRow(m, selected = m.id == vm.selectedModelPath) { vm.selectModel(m) } }
        }
    }
    Spacer(Modifier.height(10.dp))
    SecondaryButton("Download models", icon = Icons.Rounded.Add, modifier = Modifier.fillMaxWidth(), onClick = vm::openAddModels)
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun BackendSection(vm: ChatViewModel) {
    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        ComputeBackend.entries.forEach { b -> Chip(b.label, selected = vm.backend == b) { vm.selectBackend(b) } }
    }
    Text(
        "Auto tries Vulkan, then OpenGL ES, then CPU, and only uses a GPU that passes a self-test.",
        color = Neko.TextMuted,
        fontSize = 12.sp,
        modifier = Modifier.padding(top = 8.dp),
    )
}

@Composable
private fun ModelRow(model: LocalModel, selected: Boolean, onClick: () -> Unit) {
    Row(
        Modifier
            .fillMaxWidth()
            .innerCard(selected = selected)
            .selectable(selected = selected, enabled = model.supported, role = Role.RadioButton, onClick = onClick)
            .padding(horizontal = 12.dp, vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(
            Modifier
                .size(40.dp)
                .glass(RoundedCornerShape(NekoRadius.Medium), level = GlassLevel.Base),
            contentAlignment = Alignment.Center,
        ) {
            Icon(
                painterResource(R.drawable.ic_model),
                contentDescription = null,
                tint = if (selected) Neko.Pink else Neko.TextSecondary,
                modifier = Modifier.size(22.dp),
            )
        }
        Column(Modifier.weight(1f).padding(horizontal = 12.dp).alpha(if (model.supported) 1f else 0.75f)) {
            Text(model.name, color = Neko.Text, fontSize = 15.sp, fontWeight = FontWeight.Medium, maxLines = 1,
                overflow = TextOverflow.Ellipsis)
            Text(
                "${architectureLabel(model.architecture)} · ${if (model.format == "safetensors") "SafeTensors" else "PyTorch"} · " +
                    formatBytes(model.sizeBytes),
                color = Neko.TextSecondary,
                fontSize = 12.sp,
            )
            if (model.problem != null) {
                Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(top = 2.dp)) {
                    Icon(Icons.Rounded.Warning, contentDescription = null, tint = Neko.Warning, modifier = Modifier.size(14.dp))
                    Text(model.problem, color = Neko.Warning, fontSize = 12.sp, modifier = Modifier.padding(start = 4.dp))
                }
            }
        }
        if (selected) {
            Box(
                Modifier.size(24.dp).clip(CircleShape).background(Neko.Pink),
                contentAlignment = Alignment.Center,
            ) {
                Icon(Icons.Rounded.Check, contentDescription = "Selected", tint = Neko.OnPink, modifier = Modifier.size(16.dp))
            }
        }
    }
}

fun architectureLabel(arch: String): String = when (arch) {
    "gpt2" -> "GPT-2"
    "qwen3" -> "Qwen3"
    else -> arch.uppercase()
}

/** The folder the user picked through the system picker; models are its subfolders. */
@Composable
private fun FolderRow(folderName: String?, onChoose: () -> Unit) {
    if (folderName == null) {
        Text(
            "Choose the folder where you keep your models. NekoChat only reads that folder.",
            color = Neko.TextSecondary,
            fontSize = 13.sp,
            modifier = Modifier.padding(bottom = 10.dp),
        )
        PrimaryButton("Choose folder", enabled = true, modifier = Modifier.fillMaxWidth(), onClick = onChoose)
        return
    }
    Row(
        Modifier
            .fillMaxWidth()
            .innerCard()
            .padding(start = 12.dp, top = 8.dp, bottom = 8.dp, end = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(
            Modifier.size(40.dp).glass(RoundedCornerShape(NekoRadius.Medium), level = GlassLevel.Base),
            contentAlignment = Alignment.Center,
        ) {
            Icon(painterResource(R.drawable.ic_folder), contentDescription = null, tint = Neko.Pink, modifier = Modifier.size(22.dp))
        }
        Column(Modifier.weight(1f).padding(horizontal = 12.dp)) {
            Text(folderName, color = Neko.Text, fontSize = 15.sp, fontWeight = FontWeight.Medium, maxLines = 1,
                overflow = TextOverflow.Ellipsis)
            Text("New models appear automatically", color = Neko.TextMuted, fontSize = 12.sp)
        }
        Chip("Change", selected = false, onClick = onChoose)
    }
}

@Composable
private fun EmptyModels(folderName: String?) {
    Column(
        Modifier
            .fillMaxWidth()
            .innerCard()
            .padding(14.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        NekoMascot(64.dp)
        Text(
            if (folderName == null) "No models yet" else "No models in $folderName yet",
            color = Neko.Text,
            fontWeight = FontWeight.Medium,
            modifier = Modifier.padding(top = 6.dp),
        )
        Text(
            "Download one below, or put each model in its own subfolder with config.json, tokenizer files " +
                "and .safetensors or .pt weights.",
            color = Neko.TextSecondary,
            fontSize = 13.sp,
            modifier = Modifier.padding(top = 4.dp),
        )
    }
}
