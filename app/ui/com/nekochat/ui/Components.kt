package com.nekochat.ui

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.ArrowBack
import androidx.compose.material.icons.automirrored.rounded.KeyboardArrowRight
import androidx.compose.material.icons.rounded.Check
import androidx.compose.material.icons.rounded.Delete
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import app.auriel.realm.nekochat.R
import com.nekochat.ui.theme.GlassLevel
import com.nekochat.ui.theme.Neko
import com.nekochat.ui.theme.NekoRadius
import com.nekochat.ui.theme.glass

/** Maneki-neko mascot (res/drawable/neko_mascot.xml, derived from app/branding/mascot/neko-mascot.svg). */
@Composable
fun NekoMascot(size: Dp, modifier: Modifier = Modifier) {
    Image(painterResource(R.drawable.neko_mascot), contentDescription = null, modifier = modifier.size(size))
}

@Composable
fun Wordmark(fontSize: TextUnit = 30.sp) {
    Text(
        buildAnnotatedString {
            withStyle(SpanStyle(color = Neko.Pink)) { append("Neko") }
            withStyle(SpanStyle(color = Neko.Text)) { append("Chat") }
        },
        fontSize = fontSize,
        fontWeight = FontWeight.Bold,
        letterSpacing = (-0.5).sp,
    )
}

/** Glass top bar shared by every screen after setup. Root screens pass onBack = null. */
@Composable
fun GlassTopBar(
    title: String,
    subtitle: String?,
    onBack: (() -> Unit)?,
    backLabel: String = "Back",
    actions: @Composable RowScope.() -> Unit = {},
) {
    Row(
        Modifier
            .fillMaxWidth()
            .heightIn(min = 56.dp)
            .glass(RoundedCornerShape(NekoRadius.XLarge))
            .padding(4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (onBack != null) {
            IconButton(onClick = onBack) {
                Icon(Icons.AutoMirrored.Rounded.ArrowBack, contentDescription = backLabel, tint = Neko.Text)
            }
        } else {
            Spacer(Modifier.width(14.dp))
        }
        Column(Modifier.weight(1f).padding(end = 4.dp)) {
            Text(title, color = Neko.Text, fontSize = 16.sp, fontWeight = FontWeight.SemiBold, maxLines = 1,
                overflow = TextOverflow.Ellipsis)
            if (!subtitle.isNullOrEmpty()) {
                Text(subtitle, color = Neko.TextSecondary, fontSize = 12.sp, maxLines = 1, overflow = TextOverflow.Ellipsis)
            }
        }
        actions()
    }
}

@Composable
fun SectionLabel(text: String, modifier: Modifier = Modifier) {
    Text(
        text,
        color = Neko.TextSecondary,
        fontSize = 13.sp,
        fontWeight = FontWeight.Medium,
        modifier = modifier.padding(bottom = 8.dp),
    )
}

/** Primary action: pastel pink with One Dark content. Disabled state is glass with muted text. */
@Composable
fun PrimaryButton(
    text: String,
    enabled: Boolean,
    modifier: Modifier = Modifier,
    icon: ImageVector? = null,
    onClick: () -> Unit,
) {
    val shape = RoundedCornerShape(NekoRadius.Large)
    Row(
        modifier
            .height(52.dp)
            .clip(shape)
            .then(if (enabled) Modifier.background(Neko.Pink, shape) else Modifier.glass(shape))
            .clickable(enabled = enabled, role = Role.Button, onClick = onClick)
            .padding(horizontal = 20.dp),
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        val content = if (enabled) Neko.OnPink else Neko.TextMuted
        if (icon != null) {
            Icon(icon, contentDescription = null, tint = content, modifier = Modifier.size(20.dp))
            Spacer(Modifier.width(8.dp))
        }
        Text(text, color = content, fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
    }
}

/**
 * Selectable pill. Selection is shown by fill, contrast and a check mark (never colour alone).
 * destructive chips use the error colour for their label and border (and always say what they do).
 */
@Composable
fun Chip(label: String, selected: Boolean, destructive: Boolean = false, onClick: () -> Unit) {
    val shape = NekoRadius.Pill
    Row(
        Modifier
            .heightIn(min = 44.dp)
            .clip(shape)
            .then(
                when {
                    selected -> Modifier.background(Neko.Pink, shape)
                    destructive -> Modifier.glass(shape, tint = Neko.Error).border(1.dp, Neko.Error.copy(alpha = 0.6f), shape)
                    else -> Modifier.glass(shape)
                },
            )
            .clickable(role = if (destructive) Role.Button else Role.RadioButton, onClick = onClick)
            .padding(horizontal = 14.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (selected) {
            Icon(Icons.Rounded.Check, contentDescription = null, tint = Neko.OnPink, modifier = Modifier.size(16.dp))
            Spacer(Modifier.width(4.dp))
        }
        Text(
            label,
            color = when {
                selected -> Neko.OnPink
                destructive -> Neko.Error
                else -> Neko.TextSecondary
            },
            fontSize = 14.sp,
            fontWeight = if (selected || destructive) FontWeight.SemiBold else FontWeight.Medium,
        )
    }
}

/** Full-width secondary action: glass with pink content, below the primary button in emphasis. */
@Composable
fun SecondaryButton(text: String, icon: ImageVector?, modifier: Modifier = Modifier, onClick: () -> Unit) {
    val shape = RoundedCornerShape(NekoRadius.Large)
    Row(
        modifier
            .heightIn(min = 48.dp)
            .clip(shape)
            .glass(shape, level = GlassLevel.Elevated)
            .clickable(role = Role.Button, onClick = onClick)
            .padding(horizontal = 16.dp),
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (icon != null) {
            Icon(icon, contentDescription = null, tint = Neko.Pink, modifier = Modifier.size(20.dp))
            Spacer(Modifier.width(8.dp))
        }
        Text(text, color = Neko.Pink, fontSize = 15.sp, fontWeight = FontWeight.SemiBold)
    }
}

/**
 * Confirmation for destructive actions. The confirm label names the action ("Delete chat"),
 * is red, and sits on the right, away from Cancel; tapping outside cancels.
 */
@Composable
fun ConfirmDialog(
    title: String,
    message: String,
    confirmLabel: String,
    onConfirm: () -> Unit,
    onDismiss: () -> Unit,
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor = Neko.SurfaceElevated,
        shape = RoundedCornerShape(NekoRadius.XLarge),
        icon = { Icon(Icons.Rounded.Delete, contentDescription = null, tint = Neko.Error) },
        title = { Text(title, color = Neko.Text, fontSize = 19.sp, fontWeight = FontWeight.SemiBold) },
        text = { Text(message, color = Neko.TextSecondary, fontSize = 14.sp, lineHeight = 20.sp) },
        dismissButton = {
            TextButton(onClick = onDismiss, modifier = Modifier.heightIn(min = 48.dp)) {
                Text("Cancel", color = Neko.TextSecondary, fontSize = 15.sp)
            }
        },
        confirmButton = {
            TextButton(
                onClick = {
                    onDismiss()
                    onConfirm()
                },
                modifier = Modifier.heightIn(min = 48.dp),
            ) {
                Text(confirmLabel, color = Neko.Error, fontSize = 15.sp, fontWeight = FontWeight.SemiBold)
            }
        },
    )
}

/** Glass group with a title, used by the settings screens. */
@Composable
fun SettingsGroup(title: String, content: @Composable () -> Unit) {
    Column(
        Modifier
            .fillMaxWidth()
            .glass()
            .padding(horizontal = 18.dp, vertical = 16.dp),
    ) {
        Text(title, color = Neko.Text, fontSize = 15.sp, fontWeight = FontWeight.SemiBold)
        Spacer(Modifier.height(8.dp))
        content()
    }
}

/** Row that opens another page. */
@Composable
fun NavRow(title: String, subtitle: String?, onClick: () -> Unit) {
    Row(
        Modifier
            .fillMaxWidth()
            .heightIn(min = 56.dp)
            .clip(RoundedCornerShape(NekoRadius.Medium))
            .clickable(role = Role.Button, onClick = onClick)
            .padding(vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f)) {
            Text(title, color = Neko.Text, fontSize = 15.sp)
            if (subtitle != null) Text(subtitle, color = Neko.TextMuted, fontSize = 12.sp)
        }
        Icon(Icons.AutoMirrored.Rounded.KeyboardArrowRight, contentDescription = null, tint = Neko.TextSecondary)
    }
}

@Composable
fun glassFieldColors() = OutlinedTextFieldDefaults.colors(
    focusedTextColor = Neko.Text,
    unfocusedTextColor = Neko.Text,
    disabledTextColor = Neko.TextMuted,
    focusedBorderColor = Neko.Pink,
    unfocusedBorderColor = Neko.Border,
    focusedContainerColor = Neko.BackgroundDeep.copy(alpha = 0.5f),
    unfocusedContainerColor = Neko.BackgroundDeep.copy(alpha = 0.35f),
    focusedLabelColor = Neko.Pink,
    unfocusedLabelColor = Neko.TextSecondary,
    cursorColor = Neko.Pink,
)

/** Glass card for secondary content nested inside a Base panel. */
fun Modifier.innerCard(selected: Boolean = false): Modifier =
    glass(RoundedCornerShape(NekoRadius.Large), level = GlassLevel.Elevated, selected = selected)
