package com.nekochat.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Check
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.nekochat.ui.theme.Neko
import com.nekochat.ui.theme.NekoRadius
import com.nekochat.ui.theme.NekoTheme
import com.nekochat.ui.theme.Palette

/** Theme picker. Each option previews its own colours, so the choice is made by recognition, not by name. */
@Composable
fun ThemesScreen(vm: ChatViewModel) {
    Column(
        Modifier
            .fillMaxSize()
            .systemBarsPadding()
            .padding(horizontal = 12.dp, vertical = 8.dp),
    ) {
        GlassTopBar("Theme", null, onBack = vm::back, backLabel = "Back to settings")
        Column(
            Modifier
                .weight(1f)
                .verticalScroll(rememberScrollState())
                .padding(top = 12.dp, bottom = 16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            NekoTheme.entries.forEach { t ->
                ThemeCard(t, selected = vm.theme == t) { vm.selectTheme(t) }
            }
        }
    }
}

@Composable
private fun ThemeCard(theme: NekoTheme, selected: Boolean, onClick: () -> Unit) {
    val shape = RoundedCornerShape(NekoRadius.XLarge)
    Column(
        Modifier
            .fillMaxWidth()
            .innerCard(selected = selected)
            .selectable(selected = selected, role = Role.RadioButton, onClick = onClick)
            .padding(14.dp),
    ) {
        ThemePreview(theme.palette)
        Spacer(Modifier.height(12.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("NekoChat ${theme.label}", color = Neko.Text, fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
                Text(theme.description, color = Neko.TextSecondary, fontSize = 13.sp)
            }
            if (selected) {
                Box(Modifier.size(24.dp).clip(CircleShape).background(Neko.Accent), contentAlignment = Alignment.Center) {
                    Icon(Icons.Rounded.Check, contentDescription = "Selected", tint = Neko.OnAccent, modifier = Modifier.size(16.dp))
                }
            }
        }
    }
}

/** A tiny chat drawn in the theme's own palette. */
@Composable
private fun ThemePreview(p: Palette) {
    val bubble = RoundedCornerShape(14.dp)
    Column(
        Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(NekoRadius.Large))
            .background(p.background)
            .border(1.dp, p.border, RoundedCornerShape(NekoRadius.Large))
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Box(
            Modifier
                .clip(bubble)
                .background(p.surface)
                .border(1.dp, p.border, bubble)
                .padding(horizontal = 12.dp, vertical = 8.dp),
        ) {
            Text("Hi! What should we talk about?", color = p.text, fontSize = 13.sp)
        }
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
            Box(
                Modifier
                    .clip(bubble)
                    .background(p.accent.copy(alpha = 0.22f))
                    .border(1.dp, p.accent.copy(alpha = 0.6f), bubble)
                    .padding(horizontal = 12.dp, vertical = 8.dp),
            ) {
                Text("Cats, obviously", color = p.text, fontSize = 13.sp)
            }
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            listOf(p.accent, p.accentLight, p.accentSoft, p.secondaryGlow, p.gold).forEach { c ->
                Box(Modifier.size(16.dp).clip(CircleShape).background(c))
                Spacer(Modifier.width(6.dp))
            }
            Spacer(Modifier.weight(1f))
            Box(
                Modifier.clip(RoundedCornerShape(50)).background(p.accent).padding(horizontal = 12.dp, vertical = 4.dp),
            ) {
                Text("Send", color = p.background, fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
            }
        }
    }
}
