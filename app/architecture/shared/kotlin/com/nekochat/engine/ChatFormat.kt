package com.nekochat.engine

/** The assistant's identity, shared by every architecture's prompt format. */
object Persona {
    const val NAME = "Neko"
    const val DEFAULT = "a cheerful, curious and helpful AI cat"
}

data class ChatTurn(val user: String, val assistant: String?)

/** Renders a conversation into the text a model of one architecture expects. */
interface ChatFormat {
    /** The last turn has assistant == null: the prompt ends where the model's reply starts. */
    fun build(userName: String, persona: String, turns: List<ChatTurn>): String

    /** Text that ends a reply (the engine also stops on end-of-text / end-of-turn tokens). */
    fun stopSequences(userName: String): List<String>
}

/**
 * Applies stop sequences to streamed text. Text that could still turn into a stop sequence is
 * held back so the UI never flashes e.g. "\nAlice:" before it is cut.
 */
class StopFilter(private val stops: List<String>) {
    private val raw = StringBuilder()
    var stopped = false
        private set

    /** Returns the text that is safe to display. */
    fun push(piece: String): String {
        raw.append(piece)
        var cut = raw.length
        for (s in stops) {
            val i = raw.indexOf(s)
            if (i >= 0 && i < cut) {
                cut = i
                stopped = true
            }
        }
        if (stopped) {
            raw.setLength(cut)
            return display(raw.length)
        }
        var hold = 0
        for (s in stops) {
            for (k in minOf(s.length - 1, raw.length) downTo 1) {
                if (raw.regionMatches(raw.length - k, s, 0, k)) {
                    hold = maxOf(hold, k)
                    break
                }
            }
        }
        return display(raw.length - hold)
    }

    fun finalText(): String = display(raw.length)

    private fun display(end: Int): String = raw.substring(0, end).trimStart()
}
