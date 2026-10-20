// SPDX-FileCopyrightText: 2026 AurielSolaris
// SPDX-License-Identifier: GPL-3.0-or-later

package com.nekochat.gpt2

import com.nekochat.engine.ChatFormat
import com.nekochat.engine.ChatTurn

/**
 * ReALM v0.1.0 ("realm-v1"): a GPT-2 fine-tuned on "user: <text>" / "assistant: <text>" lines joined
 * by "\n", conversations separated by "\n\n", with no preamble and no BOS/EOS tokens (ReALM's
 * src/training/data.py and src/inference/chat.py). Each turn is "\nuser: <message>\nassistant:"; the
 * reply starts with a space, may span several lines, and ends at the next "\nuser:" or at a blank line
 * (end of conversation).
 */
object RealmChatFormat : ChatFormat {
    override fun build(userName: String, persona: String, turns: List<ChatTurn>): String = buildString {
        for (t in turns) {
            append("\nuser: ").append(t.user.trim())
            append("\nassistant:")
            if (t.assistant != null) append(' ').append(t.assistant.trim())
        }
    }

    override fun stopSequences(userName: String): List<String> = listOf("\nuser:", "\n\n")

    /** Trained on random 512-token windows (it never saw a conversation start): keep the newest 512 tokens. */
    override val contextTokens: Int get() = 512

    override val usesPersona: Boolean get() = false
}
