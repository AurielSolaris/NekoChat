package com.nekochat.qwen3

import com.nekochat.engine.ChatFormat
import com.nekochat.engine.ChatTurn
import com.nekochat.engine.Persona

/**
 * Qwen3's ChatML template with thinking disabled: the reply starts after an empty
 * <think></think> block, exactly as `apply_chat_template(enable_thinking=False)` renders it.
 * Earlier assistant turns are included without their (empty) think blocks, as the template does.
 */
object Qwen3ChatFormat : ChatFormat {
    override fun build(userName: String, persona: String, turns: List<ChatTurn>): String = buildString {
        val who = clean(persona.trim().trimEnd('.').ifEmpty { Persona.DEFAULT })
        append("<|im_start|>system\n")
        append("You are ").append(Persona.NAME).append(", ").append(who).append(". ")
        append("You are chatting with ").append(clean(userName)).append(" on their phone.")
        append("<|im_end|>\n")
        for (t in turns) {
            append("<|im_start|>user\n").append(clean(t.user.trim())).append("<|im_end|>\n")
            append("<|im_start|>assistant\n")
            if (t.assistant != null) append(clean(t.assistant.trim())).append("<|im_end|>\n")
            else append("<think>\n\n</think>\n\n")
        }
    }

    override fun stopSequences(userName: String): List<String> = listOf("<|im_end|>", "<|im_start|>", "<|endoftext|>")

    // Typed text must never become control tokens: "<|" is split so the tokenizer can't match one.
    private fun clean(s: String) = s.replace("<|", "<​|")
}
