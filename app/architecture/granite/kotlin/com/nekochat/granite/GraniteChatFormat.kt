// SPDX-FileCopyrightText: 2026 AurielSolaris
// SPDX-License-Identifier: GPL-3.0-or-later

package com.nekochat.granite

import com.nekochat.engine.ChatFormat
import com.nekochat.engine.ChatTurn
import com.nekochat.engine.Persona

/**
 * Granite 4.0's chat template without tools or documents, as `apply_chat_template` renders it:
 * `<|start_of_role|>role<|end_of_role|>text<|end_of_text|>\n` per message, the reply starting after the
 * assistant role header. Neko's persona takes the place of Granite's default system message.
 */
object GraniteChatFormat : ChatFormat {
    override fun build(userName: String, persona: String, turns: List<ChatTurn>): String = buildString {
        val who = clean(persona.trim().trimEnd('.').ifEmpty { Persona.DEFAULT })
        append("<|start_of_role|>system<|end_of_role|>")
        append("You are ").append(Persona.NAME).append(", ").append(who).append(". ")
        append("You are chatting with ").append(clean(userName)).append(" on their phone.")
        append("<|end_of_text|>\n")
        for (t in turns) {
            append("<|start_of_role|>user<|end_of_role|>").append(clean(t.user.trim())).append("<|end_of_text|>\n")
            append("<|start_of_role|>assistant<|end_of_role|>")
            if (t.assistant != null) append(clean(t.assistant.trim())).append("<|end_of_text|>\n")
        }
    }

    override fun stopSequences(userName: String): List<String> = listOf("<|end_of_text|>", "<|start_of_role|>")

    // Typed text must never become control tokens: "<|" is split so the tokenizer can't match one.
    private fun clean(s: String) = s.replace("<|", "<​|")
}
