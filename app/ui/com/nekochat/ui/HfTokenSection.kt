// SPDX-FileCopyrightText: 2026 LinuxKernel44
// SPDX-License-Identifier: GPL-3.0-or-later

package com.nekochat.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.nekochat.download.HubToken
import com.nekochat.ui.theme.Neko
import com.nekochat.ui.theme.NekoRadius

/**
 * Settings → Hugging Face: save, replace or remove the access token used for gated and private
 * models. The token is typed into a masked field, verified with Hugging Face, and stored by
 * [HubToken] (Keystore-encrypted). It is kept in plain `remember` state, never `rememberSaveable`,
 * so it can't end up in the saved instance state; the saved token is only ever shown masked.
 */
@Composable
fun HfTokenSection(vm: ChatViewModel) {
    if (!vm.hfTokenSupported) {
        Text(
            "Saving a Hugging Face token needs Android 6.0 or newer. Without one, NekoChat downloads public models only.",
            color = Neko.TextMuted, fontSize = 12.sp, lineHeight = 17.sp,
        )
        return
    }

    var input by remember { mutableStateOf("") }
    var editing by remember { mutableStateOf(false) }
    var confirmRemove by remember { mutableStateOf(false) }
    val masked = vm.hfTokenMasked
    val submit = {
        if (HubToken.isWellFormed(input) && !vm.hfTokenBusy) {
            vm.saveHfToken(input) {
                input = ""
                editing = false
            }
        }
    }

    if (masked != null && !editing) {
        Row(
            Modifier.fillMaxWidth().innerCard().padding(start = 14.dp, top = 10.dp, bottom = 10.dp, end = 10.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            androidx.compose.foundation.layout.Column(Modifier.weight(1f)) {
                Text("Token saved", color = Neko.Text, fontSize = 15.sp, fontWeight = FontWeight.Medium)
                Text(masked, color = Neko.TextSecondary, fontSize = 13.sp)
            }
            Chip("Replace", selected = false) {
                vm.clearHfTokenMessage()
                editing = true
            }
            Spacer(Modifier.width(8.dp))
            Chip("Remove", selected = false, destructive = true) { confirmRemove = true }
        }
    } else {
        OutlinedTextField(
            value = input,
            // Pasted tokens often carry a trailing newline or space.
            onValueChange = { input = it.filterNot(Char::isWhitespace).take(220) },
            label = { Text(if (masked != null) "New access token" else "Access token") },
            placeholder = { Text("hf_…", color = Neko.TextMuted) },
            singleLine = true,
            visualTransformation = PasswordVisualTransformation(),
            shape = RoundedCornerShape(NekoRadius.Large),
            colors = glassFieldColors(),
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password, imeAction = ImeAction.Done),
            keyboardActions = KeyboardActions(onDone = { submit() }),
            modifier = Modifier.fillMaxWidth(),
        )
        Spacer(Modifier.height(10.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
            PrimaryButton(
                if (vm.hfTokenBusy) "Checking…" else "Save token",
                enabled = HubToken.isWellFormed(input) && !vm.hfTokenBusy,
                modifier = Modifier.weight(1f),
            ) { submit() }
            if (editing) {
                Chip("Cancel", selected = false) {
                    input = ""
                    editing = false
                    vm.clearHfTokenMessage()
                }
            }
        }
    }

    vm.hfTokenMessage?.let {
        Text(
            it,
            color = if (vm.hfTokenFailed) Neko.Error else Neko.Success,
            fontSize = 13.sp, lineHeight = 18.sp,
            modifier = Modifier.padding(top = 8.dp),
        )
    }
    Text(
        "Needed for gated or private models. Create a read-only token at huggingface.co/settings/tokens. " +
            "NekoChat checks it with Hugging Face, keeps it encrypted with the Android Keystore on this phone only, " +
            "and sends it to huggingface.co only, never to download mirrors. Remove or replace it any time.",
        color = Neko.TextMuted, fontSize = 12.sp, lineHeight = 17.sp,
        modifier = Modifier.padding(top = 8.dp),
    )

    if (confirmRemove) {
        ConfirmDialog(
            title = "Remove token?",
            message = "NekoChat forgets your Hugging Face token and can then only download public models. " +
                "Your token itself stays valid on huggingface.co; revoke it there to disable it everywhere.",
            confirmLabel = "Remove token",
            onConfirm = { vm.removeHfToken() },
            onDismiss = { confirmRemove = false },
        )
    }
}
