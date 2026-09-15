package com.nekochat.ui

import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
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
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.Send
import androidx.compose.material.icons.rounded.Delete
import androidx.compose.material.icons.rounded.Settings
import androidx.compose.material.icons.rounded.Warning
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.runtime.withFrameNanos
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.semantics.Role as SemanticsRole
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.nekochat.data.Conversation
import com.nekochat.data.Role
import com.nekochat.engine.Persona
import com.nekochat.engine.formatBytes
import com.nekochat.ui.theme.GlassLevel
import com.nekochat.ui.theme.Neko
import com.nekochat.ui.theme.NekoRadius
import com.nekochat.ui.theme.glass

@Composable
fun ChatScreen(vm: ChatViewModel) {
    val load by vm.load.collectAsState()
    val chat = vm.active ?: return
    Column(
        Modifier
            .fillMaxSize()
            .systemBarsPadding()
            .imePadding()
            .padding(horizontal = 12.dp, vertical = 8.dp),
    ) {
        val memory by vm.memory.collectAsState()
        val subtitle = when (val s = load) {
            is LoadState.Ready -> listOfNotNull(
                s.info.backend,
                s.info.weights.ifEmpty { null },
                vm.lastSpeed,
                memory?.let { formatBytes(it.model) },
            ).joinToString(" · ")
            else -> loadSummary(s)
        }
        var confirmClear by remember { mutableStateOf(false) }
        GlassTopBar(chat.title, subtitle, onBack = vm::back, backLabel = "Back to chats") {
            IconButton(onClick = { confirmClear = true }, enabled = chat.messages.isNotEmpty()) {
                Icon(Icons.Rounded.Delete, contentDescription = "Clear messages",
                    tint = if (chat.messages.isNotEmpty()) Neko.Error else Neko.TextMuted.copy(alpha = 0.5f))
            }
            IconButton(onClick = vm::openChatSettings) {
                Icon(Icons.Rounded.Settings, contentDescription = "Chat settings", tint = Neko.Text)
            }
        }
        Box(Modifier.weight(1f).fillMaxWidth()) {
            when (val s = load) {
                is LoadState.Loading -> LoadingCard(s)
                is LoadState.Failed -> FailedCard(s.message, onRetry = vm::retryLoad, onSettings = vm::openAppSettings)
                is LoadState.Ready -> MessageList(vm, chat)
                LoadState.Idle -> Unit
            }
        }
        InputBar(vm, enabled = load is LoadState.Ready)
        if (confirmClear) {
            ConfirmDialog(
                title = "Clear all messages?",
                message = "All ${chat.messages.size} messages in “${chat.title}” will be deleted. The chat and its " +
                    "settings stay. This can't be undone.",
                confirmLabel = "Clear messages",
                onConfirm = vm::clearActiveChat,
                onDismiss = { confirmClear = false },
            )
        }
    }
}

@Composable
private fun MessageList(vm: ChatViewModel, chat: Conversation) {
    val listState = rememberLazyListState()
    val messages = chat.messages
    val streaming = vm.generatingChatId == chat.id
    var streamText by remember { mutableStateOf("") }

    // Render/inference decoupling: tokens land in a StreamBuffer on the inference thread; the UI
    // samples it once per vsync, so at most one recomposition happens per frame.
    LaunchedEffect(streaming) {
        if (!streaming) return@LaunchedEffect
        streamText = ""
        var seen = -1L
        while (true) {
            withFrameNanos { }
            val v = vm.stream.version
            if (v != seen) {
                seen = v
                streamText = vm.stream.current
            }
        }
    }
    LaunchedEffect(messages.size) {
        if (listState.firstVisibleItemIndex <= 1) listState.scrollToItem(0)
    }

    if (messages.isEmpty() && !streaming) {
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                NekoMascot(96.dp)
                Text(
                    "Say hi to ${Persona.NAME}, ${vm.username}",
                    color = Neko.TextSecondary,
                    fontSize = 15.sp,
                    modifier = Modifier.padding(top = 8.dp),
                )
            }
        }
        return
    }

    // reverseLayout keeps the newest message (and the growing streamed reply) pinned to the bottom.
    LazyColumn(
        state = listState,
        reverseLayout = true,
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        if (streaming) {
            item(key = "stream") { Message(Role.Assistant, streamText, typing = streamText.isEmpty()) }
        }
        items(messages.asReversed(), key = { it.id }) { m ->
            Message(
                m.role,
                m.text,
                modifier = Modifier.animateItem(fadeInSpec = tween(180), placementSpec = tween(180), fadeOutSpec = null),
            )
        }
    }
}

@Composable
private fun Message(role: Role, text: String, modifier: Modifier = Modifier, typing: Boolean = false) {
    val mine = role == Role.User
    Column(
        modifier.fillMaxWidth(),
        horizontalAlignment = if (mine) Alignment.End else Alignment.Start,
    ) {
        if (role == Role.Assistant) {
            Text(Persona.NAME, color = Neko.Accent, fontSize = 12.sp, fontWeight = FontWeight.SemiBold,
                modifier = Modifier.padding(start = 6.dp, bottom = 4.dp))
        }
        val shape = RoundedCornerShape(NekoRadius.XLarge)
        Row(
            Modifier
                .widthIn(max = 300.dp)
                .then(
                    when (role) {
                        Role.User -> Modifier.glass(shape, level = GlassLevel.Elevated, tint = Neko.Accent)
                            .background(Neko.Accent.copy(alpha = 0.14f), shape)
                        Role.Assistant -> Modifier.glass(shape)
                        Role.Error -> Modifier.glass(shape, tint = Neko.Error)
                    },
                )
                .padding(horizontal = 16.dp, vertical = 12.dp),
            verticalAlignment = Alignment.Top,
        ) {
            if (role == Role.Error) {
                Icon(Icons.Rounded.Warning, contentDescription = "Error", tint = Neko.Error,
                    modifier = Modifier.padding(end = 8.dp).size(18.dp))
            }
            if (typing) TypingDots() else Text(text, color = Neko.Text, fontSize = 15.sp, lineHeight = 22.sp)
        }
    }
}

@Composable
private fun TypingDots() {
    val t = rememberInfiniteTransition(label = "typing")
    Row(Modifier.padding(vertical = 6.dp), horizontalArrangement = Arrangement.spacedBy(5.dp)) {
        repeat(3) { i ->
            val a by t.animateFloat(
                initialValue = 0.3f,
                targetValue = 1f,
                animationSpec = infiniteRepeatable(tween(480, delayMillis = i * 150), RepeatMode.Reverse),
                label = "dot$i",
            )
            Box(Modifier.size(7.dp).alpha(a).clip(CircleShape).background(Neko.AccentLight))
        }
    }
}

@Composable
private fun InputBar(vm: ChatViewModel, enabled: Boolean) {
    var text by rememberSaveable { mutableStateOf("") }
    var focused by remember { mutableStateOf(false) }
    val generating = vm.generating
    val canSend = enabled && text.isNotBlank() && !generating
    val send = {
        if (canSend) {
            vm.send(text)
            text = ""
        }
    }
    Row(
        Modifier
            .fillMaxWidth()
            .glass(NekoRadius.Pill, selected = focused)
            .padding(start = 20.dp, end = 6.dp, top = 6.dp, bottom = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(Modifier.weight(1f).padding(vertical = 8.dp)) {
            if (text.isEmpty()) {
                Text("Message ${Persona.NAME}…", color = Neko.TextMuted, fontSize = 16.sp)
            }
            BasicTextField(
                value = text,
                onValueChange = { text = it },
                enabled = enabled,
                maxLines = 5,
                textStyle = TextStyle(color = Neko.Text, fontSize = 16.sp, lineHeight = 22.sp),
                cursorBrush = SolidColor(Neko.Accent),
                keyboardOptions = KeyboardOptions(capitalization = KeyboardCapitalization.Sentences, imeAction = ImeAction.Send),
                keyboardActions = KeyboardActions(onSend = { send() }),
                modifier = Modifier.fillMaxWidth().onFocusChanged { focused = it.isFocused },
            )
        }
        Spacer(Modifier.width(8.dp))
        val active = generating || canSend
        Box(
            Modifier
                .size(48.dp)
                .clip(CircleShape)
                .background(if (active) Neko.Accent else Neko.SurfaceElevated, CircleShape)
                .clickable(enabled = active, role = SemanticsRole.Button) { if (generating) vm.stop() else send() },
            contentAlignment = Alignment.Center,
        ) {
            if (generating) {
                Box(Modifier.size(14.dp).clip(RoundedCornerShape(3.dp)).background(Neko.OnAccent))
            } else {
                Icon(Icons.AutoMirrored.Rounded.Send, contentDescription = "Send",
                    tint = if (active) Neko.OnAccent else Neko.TextMuted, modifier = Modifier.size(22.dp))
            }
        }
    }
}

@Composable
private fun LoadingCard(s: LoadState.Loading) {
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(
            Modifier
                .widthIn(max = 360.dp)
                .fillMaxWidth(0.85f)
                .glass()
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            NekoMascot(88.dp)
            Text("Waking Neko up", color = Neko.Text, fontSize = 17.sp, fontWeight = FontWeight.SemiBold,
                modifier = Modifier.padding(top = 8.dp))
            Spacer(Modifier.height(14.dp))
            LinearProgressIndicator(
                progress = { s.progress },
                modifier = Modifier.fillMaxWidth().height(6.dp).clip(RoundedCornerShape(3.dp)),
                color = Neko.Accent,
                trackColor = Neko.Border,
                drawStopIndicator = {},
            )
            Spacer(Modifier.height(10.dp))
            Text("${s.stage} · ${(s.progress * 100).toInt()}%", color = Neko.TextSecondary, fontSize = 13.sp)
        }
    }
}

@Composable
private fun FailedCard(message: String, onRetry: () -> Unit, onSettings: () -> Unit) {
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(
            Modifier
                .widthIn(max = 380.dp)
                .fillMaxWidth(0.9f)
                .glass(tint = Neko.Error)
                .padding(22.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Rounded.Warning, contentDescription = null, tint = Neko.Error)
                Text("Couldn't load the model", color = Neko.Text, fontSize = 17.sp, fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.padding(start = 8.dp))
            }
            Text(message, color = Neko.TextSecondary, fontSize = 13.sp, modifier = Modifier.padding(top = 8.dp))
            Spacer(Modifier.height(18.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                PrimaryButton("Retry", enabled = true, onClick = onRetry)
                Chip("Settings", selected = false, onClick = onSettings)
            }
        }
    }
}
