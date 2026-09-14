package com.nekochat.data

import com.nekochat.engine.GenerationParams
import com.nekochat.engine.Persona

enum class Role { User, Assistant, Error }

data class ChatMessage(val id: Long, val role: Role, val text: String)

/** Sampling and persona settings, stored per chat. */
data class ChatSettings(
    val temperature: Float = 0.8f,
    val topK: Int = 40,
    val topP: Float = 0.95f,
    val repeatPenalty: Float = 1.15f,
    val repeatLastN: Int = 64,
    val maxNewTokens: Int = 160,
    val seed: Long = 0L,
    val persona: String = Persona.DEFAULT,
) {
    fun toParams() = GenerationParams(
        maxNewTokens = maxNewTokens,
        temperature = temperature,
        topK = topK,
        topP = topP,
        repeatPenalty = repeatPenalty,
        repeatLastN = repeatLastN,
        seed = seed,
    )

    companion object {
        const val MAX_REPLY_TOKENS = 4096
    }
}

data class Conversation(
    val id: Long,
    val title: String,
    val updatedAt: Long,
    val settings: ChatSettings,
    val messages: List<ChatMessage>,
)
