package com.nekochat.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.KeyboardArrowDown
import androidx.compose.material.icons.rounded.KeyboardArrowUp
import androidx.compose.material3.Icon
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.nekochat.ui.theme.Neko
import com.nekochat.ui.theme.NekoRadius
import com.nekochat.ui.theme.glass

/** First run only: a name and a model (compute backend tucked under "Advanced"). Later changes happen in Settings. */
@Composable
fun SetupScreen(vm: ChatViewModel) {
    val scanned by vm.models.collectAsState()
    val models = scanned.orEmpty()
    val selected = models.firstOrNull { it.id == vm.selectedModelPath && it.supported }
    val canStart = vm.username.isNotBlank() && selected != null
    var advanced by rememberSaveable { mutableStateOf(false) }

    // Sensible default: with exactly one usable model, pre-select it.
    LaunchedEffect(scanned) {
        val usable = models.filter { it.supported }
        if (selected == null && usable.size == 1) vm.selectModel(usable.first())
    }

    Column(
        Modifier
            .fillMaxSize()
            .systemBarsPadding()
            .imePadding()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp, vertical = 16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Spacer(Modifier.height(16.dp))
        NekoMascot(112.dp)
        Spacer(Modifier.height(8.dp))
        Wordmark()
        Text("Private AI that runs on your phone", color = Neko.TextSecondary, fontSize = 14.sp)
        Spacer(Modifier.height(24.dp))

        Column(
            Modifier
                .widthIn(max = 520.dp)
                .fillMaxWidth()
                .glass()
                .padding(18.dp),
        ) {
            SectionLabel("Your name")
            OutlinedTextField(
                value = vm.username,
                onValueChange = vm::updateUsername,
                singleLine = true,
                placeholder = { Text("What should Neko call you?", color = Neko.TextMuted) },
                shape = RoundedCornerShape(NekoRadius.Large),
                colors = glassFieldColors(),
                keyboardOptions = KeyboardOptions(capitalization = KeyboardCapitalization.Words, imeAction = ImeAction.Done),
                modifier = Modifier.fillMaxWidth(),
            )

            Spacer(Modifier.height(20.dp))
            ModelsSection(vm)

            // Progressive disclosure: the default (Auto) is right for almost everyone.
            Spacer(Modifier.height(12.dp))
            Row(
                Modifier
                    .fillMaxWidth()
                    .heightIn(min = 48.dp)
                    .clip(RoundedCornerShape(NekoRadius.Medium))
                    .clickable(role = Role.Button) { advanced = !advanced },
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text("Advanced", color = Neko.TextSecondary, fontSize = 14.sp, modifier = Modifier.weight(1f))
                Text(vm.backend.label, color = Neko.TextMuted, fontSize = 13.sp)
                Icon(
                    if (advanced) Icons.Rounded.KeyboardArrowUp else Icons.Rounded.KeyboardArrowDown,
                    contentDescription = if (advanced) "Hide advanced" else "Show advanced",
                    tint = Neko.TextSecondary,
                )
            }
            AnimatedVisibility(advanced) {
                Column {
                    SectionLabel("Compute backend")
                    BackendSection(vm)
                }
            }
        }

        Spacer(Modifier.height(20.dp))
        PrimaryButton("Start chatting", enabled = canStart, modifier = Modifier.widthIn(max = 520.dp).fillMaxWidth()) {
            selected?.let(vm::startChatting)
        }
        if (!canStart) {
            Text(
                when {
                    vm.username.isBlank() && selected == null -> "Enter your name and pick a model"
                    vm.username.isBlank() -> "Enter your name to continue"
                    else -> "Pick a model to continue"
                },
                color = Neko.TextMuted,
                fontSize = 12.sp,
                modifier = Modifier.padding(top = 8.dp),
            )
        }
    }
}
