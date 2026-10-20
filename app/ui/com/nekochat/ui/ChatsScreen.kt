// SPDX-FileCopyrightText: 2026 AurielSolaris
// SPDX-License-Identifier: GPL-3.0-or-later

package com.nekochat.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Add
import androidx.compose.material.icons.rounded.Delete
import androidx.compose.material.icons.rounded.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.nekochat.data.Conversation
import com.nekochat.ui.theme.Neko
import com.nekochat.ui.theme.NekoRadius
import com.nekochat.ui.theme.Spectral
import com.nekochat.ui.theme.glass
import java.text.DateFormat
import java.util.Calendar
import java.util.Date
import kotlinx.coroutines.delay

/** Home screen. No back button: setup is done; everything else lives behind Settings. */
@Composable
fun ChatsScreen(vm: ChatViewModel) {
    val load by vm.load.collectAsState()
    var pendingDelete by remember { mutableStateOf<Conversation?>(null) }
    Column(
        Modifier
            .fillMaxSize()
            .systemBarsPadding()
            .padding(horizontal = 12.dp, vertical = 8.dp),
    ) {
        GlassTopBar("Chats", loadSummary(load), onBack = null) {
            IconButton(onClick = vm::openAppSettings) {
                Icon(Icons.Rounded.Settings, contentDescription = "Settings", tint = Neko.Text)
            }
        }

        Greeting(vm.username, Modifier.padding(start = 8.dp, end = 8.dp, top = 16.dp, bottom = 4.dp))

        Box(Modifier.weight(1f).fillMaxWidth()) {
            if (vm.chats.isEmpty()) {
                Column(
                    Modifier.fillMaxSize(),
                    verticalArrangement = Arrangement.Center,
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    NekoMascot(96.dp)
                    Text("No chats yet", color = Neko.TextSecondary, fontSize = 15.sp, modifier = Modifier.padding(top = 8.dp))
                }
            } else {
                LazyColumn(
                    contentPadding = PaddingValues(vertical = 12.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    items(vm.chats, key = { it.id }) { c ->
                        ChatRow(c, onOpen = { vm.openChat(c.id) }, onDelete = { pendingDelete = c }, Modifier.animateItem())
                    }
                }
            }
        }

        PrimaryButton("New chat", enabled = true, modifier = Modifier.fillMaxWidth(), icon = Icons.Rounded.Add,
            onClick = vm::newChat)
        Spacer(Modifier.height(4.dp))
    }

    pendingDelete?.let { c ->
        ConfirmDialog(
            title = "Delete this chat?",
            message = "“${c.title}” and its ${c.messages.size} messages will be deleted. This can't be undone.",
            confirmLabel = "Delete chat",
            onConfirm = { vm.deleteChat(c.id) },
            onDismiss = { pendingDelete = null },
        )
    }
}

@Composable
private fun ChatRow(c: Conversation, onOpen: () -> Unit, onDelete: () -> Unit, modifier: Modifier) {
    val last = c.messages.lastOrNull()
    Row(
        modifier
            .fillMaxWidth()
            .glass(RoundedCornerShape(NekoRadius.XLarge))
            .clickable(onClick = onOpen)
            .padding(start = 16.dp, top = 12.dp, bottom = 12.dp, end = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f)) {
            Text(c.title, color = Neko.Text, fontSize = 15.sp, fontWeight = FontWeight.Medium, maxLines = 1,
                overflow = TextOverflow.Ellipsis)
            Text(last?.text ?: "No messages yet", color = Neko.TextSecondary, fontSize = 13.sp, maxLines = 1,
                overflow = TextOverflow.Ellipsis)
            Text(
                DateFormat.getDateTimeInstance(DateFormat.SHORT, DateFormat.SHORT).format(Date(c.updatedAt)),
                color = Neko.TextMuted, fontSize = 12.sp,
            )
        }
        IconButton(onClick = onDelete) {
            Icon(Icons.Rounded.Delete, contentDescription = "Delete chat ${c.title}", tint = Neko.Error)
        }
    }
}

/** Part of the day by local hour; each has a headline and a line from Neko. */
private enum class DayPart(val hello: String, val line: String) {
    Morning("Good morning", "Fresh coffee, fresh chats. What's on your mind?"),
    Afternoon("Good afternoon", "Taking a break? Neko's all ears."),
    Evening("Good evening", "Winding down? Let's chat."),
    Night("Up late", "The moon's out and so is Neko.");

    companion object {
        fun at(hour: Int) = when (hour) {
            in 5..11 -> Morning
            in 12..16 -> Afternoon
            in 17..21 -> Evening
            else -> Night
        }

        fun now() = at(Calendar.getInstance().get(Calendar.HOUR_OF_DAY))
    }
}

@Composable
private fun Greeting(username: String, modifier: Modifier = Modifier) {
    // Re-checked every minute so the greeting turns over while the screen stays open.
    val part by produceState(DayPart.now()) {
        while (true) {
            delay(60_000)
            value = DayPart.now()
        }
    }
    val name = username.trim()
    val headline = when {
        part == DayPart.Night -> if (name.isEmpty()) "Up late?" else "Up late, $name?"
        name.isEmpty() -> part.hello
        else -> "${part.hello}, $name"
    }
    Column(modifier.fillMaxWidth()) {
        Text(headline, color = Neko.Text, fontSize = 26.sp, fontFamily = Spectral, maxLines = 1,
            overflow = TextOverflow.Ellipsis)
        Text(part.line, color = Neko.TextSecondary, fontSize = 14.sp)
    }
}

fun loadSummary(load: LoadState): String = when (load) {
    is LoadState.Ready -> "${load.modelName} · ${load.info.backend} · ${load.info.device}"
    is LoadState.Loading -> "Loading model… ${(load.progress * 100).toInt()}%"
    is LoadState.Failed -> "Model failed to load"
    LoadState.Idle -> ""
}
