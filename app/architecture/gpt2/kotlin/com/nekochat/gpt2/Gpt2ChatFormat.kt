package com.nekochat.gpt2

import com.nekochat.engine.ChatFormat
import com.nekochat.engine.ChatTurn
import com.nekochat.engine.Persona

/**
 * GPT-2 has no chat template, so conversations are rendered as a transcript that the base
 * model continues. A reply ends at the first newline (the model would start the next speaker).
 */
object Gpt2ChatFormat : ChatFormat {
    override fun build(userName: String, persona: String, turns: List<ChatTurn>): String = buildString {
        append("The following is a friendly conversation between ")
        append(userName).append(" and ").append(Persona.NAME)
        append(", ").append(persona.trim().trimEnd('.').ifEmpty { Persona.DEFAULT }).append(".\n\n")
        for (t in turns) {
            append(userName).append(": ").append(t.user.replace('\n', ' ').trim()).append('\n')
            append(Persona.NAME).append(':')
            if (t.assistant != null) append(' ').append(t.assistant.trim()).append('\n')
        }
    }

    override fun stopSequences(userName: String): List<String> = listOf("\n", "$userName:", "${Persona.NAME}:")
}
