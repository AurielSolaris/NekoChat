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
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.nekochat.data.ChatSettings
import com.nekochat.engine.Persona
import com.nekochat.ui.theme.Neko
import com.nekochat.ui.theme.NekoRadius
import kotlin.math.roundToInt

/** Per-chat generation settings. Every change is applied and saved immediately. */
@Composable
fun ChatSettingsScreen(vm: ChatViewModel) {
    val chat = vm.active ?: return
    val s = chat.settings
    Column(
        Modifier
            .fillMaxSize()
            .systemBarsPadding()
            .imePadding()
            .padding(horizontal = 12.dp, vertical = 8.dp),
    ) {
        GlassTopBar("Chat settings", chat.title, onBack = vm::back, backLabel = "Back to chat")

        Column(
            Modifier
                .weight(1f)
                .verticalScroll(rememberScrollState())
                .padding(top = 12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            SettingsGroup("Chat") {
                OutlinedTextField(
                    value = chat.title,
                    onValueChange = vm::renameActive,
                    label = { Text("Name") },
                    singleLine = true,
                    shape = RoundedCornerShape(NekoRadius.Large),
                    colors = glassFieldColors(),
                    modifier = Modifier.fillMaxWidth(),
                )
                Spacer(Modifier.height(10.dp))
                OutlinedTextField(
                    value = s.persona,
                    onValueChange = { v -> vm.updateSettings { it.copy(persona = v.take(200)) } },
                    label = { Text("${Persona.NAME} is…") },
                    minLines = 2,
                    shape = RoundedCornerShape(NekoRadius.Large),
                    colors = glassFieldColors(),
                    modifier = Modifier.fillMaxWidth(),
                )
            }

            SettingsGroup("Sampling") {
                Setting("Temperature", "%.2f".format(s.temperature), "0 always picks the most likely word") {
                    NekoSlider(s.temperature, 0f..2f, steps = 39) { v -> vm.updateSettings { it.copy(temperature = v) } }
                }
                Setting("Top-K", if (s.topK == 0) "Off" else s.topK.toString(), "Choose among the K most likely tokens") {
                    NekoSlider(s.topK.toFloat(), 0f..200f) { v -> vm.updateSettings { it.copy(topK = v.roundToInt()) } }
                }
                Setting("Top-P", "%.2f".format(s.topP), "Keep tokens covering this share of probability") {
                    NekoSlider(s.topP, 0.05f..1f, steps = 18) { v -> vm.updateSettings { it.copy(topP = v) } }
                }
            }

            SettingsGroup("Repetition") {
                Setting("Penalty", "%.2f".format(s.repeatPenalty), "1.00 is off; higher discourages repeats") {
                    NekoSlider(s.repeatPenalty, 1f..2f, steps = 19) { v -> vm.updateSettings { it.copy(repeatPenalty = v) } }
                }
                Setting("Window", "${s.repeatLastN} tokens", "How far back the penalty looks") {
                    NekoSlider(s.repeatLastN.toFloat(), 0f..512f) { v ->
                        vm.updateSettings { it.copy(repeatLastN = (v / 16).roundToInt() * 16) }
                    }
                }
            }

            SettingsGroup("Output") {
                Setting("Max reply length", "${s.maxNewTokens} tokens", "Replies also stop when the model's context is full") {
                    NekoSlider(s.maxNewTokens.toFloat(), 16f..ChatSettings.MAX_REPLY_TOKENS.toFloat()) { v ->
                        vm.updateSettings { it.copy(maxNewTokens = ((v / 16).roundToInt() * 16).coerceIn(16, ChatSettings.MAX_REPLY_TOKENS)) }
                    }
                }
                OutlinedTextField(
                    value = if (s.seed == 0L) "" else s.seed.toString(),
                    onValueChange = { v -> vm.updateSettings { it.copy(seed = v.filter(Char::isDigit).take(18).toLongOrNull() ?: 0L) } },
                    label = { Text("Seed") },
                    placeholder = { Text("Random", color = Neko.TextMuted) },
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    shape = RoundedCornerShape(NekoRadius.Large),
                    colors = glassFieldColors(),
                    modifier = Modifier.fillMaxWidth(),
                )
            }

            Row(Modifier.fillMaxWidth().padding(bottom = 16.dp), horizontalArrangement = Arrangement.Center) {
                Chip("Reset to defaults", selected = false) { vm.updateSettings { ChatSettings() } }
            }
        }
    }
}

@Composable
private fun Setting(name: String, value: String, hint: String?, control: @Composable () -> Unit) {
    Column(Modifier.padding(vertical = 4.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(name, color = Neko.Text, fontSize = 15.sp, modifier = Modifier.weight(1f))
            Text(value, color = Neko.Pink, fontSize = 14.sp, fontWeight = FontWeight.SemiBold, fontFamily = FontFamily.Monospace)
        }
        if (hint != null) Text(hint, color = Neko.TextMuted, fontSize = 12.sp)
        control()
    }
}

@Composable
private fun NekoSlider(value: Float, range: ClosedFloatingPointRange<Float>, steps: Int = 0, onChange: (Float) -> Unit) {
    Slider(
        value = value.coerceIn(range),
        onValueChange = onChange,
        valueRange = range,
        steps = steps,
        colors = SliderDefaults.colors(
            thumbColor = Neko.Pink,
            activeTrackColor = Neko.Pink,
            inactiveTrackColor = Neko.Border,
            activeTickColor = Color.Transparent,
            inactiveTickColor = Color.Transparent,
        ),
    )
}
