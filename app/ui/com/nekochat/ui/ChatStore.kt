package com.nekochat.ui

import android.content.Context
import com.nekochat.data.ChatMessage
import com.nekochat.data.ChatSettings
import com.nekochat.data.Conversation
import com.nekochat.data.NekoDatabase
import com.nekochat.data.PrefKeys
import com.nekochat.data.PreferenceEntity
import com.nekochat.data.Role
import com.nekochat.engine.ComputeBackend
import org.json.JSONArray
import java.io.File

/**
 * One-time import of what builds before the Room database saved: SharedPreferences "nekochat"
 * and filesDir/chats.json. Both are removed once imported.
 */
object LegacyImport {
    suspend fun run(context: Context, db: NekoDatabase) {
        val sp = context.getSharedPreferences("nekochat", Context.MODE_PRIVATE)
        val json = File(context.filesDir, "chats.json")
        if (sp.all.isEmpty() && !json.exists()) return

        val prefs = db.preferences()
        val user = sp.getString("username", null)
        val model = sp.getString("model", null)
        user?.let { prefs.put(PreferenceEntity(PrefKeys.USERNAME, it)) }
        model?.let { prefs.put(PreferenceEntity(PrefKeys.MODEL, it)) }
        sp.getString("models_tree", null)?.let { prefs.put(PreferenceEntity(PrefKeys.MODELS_TREE, it)) }
        if (sp.contains("backend")) {
            ComputeBackend.entries.getOrNull(sp.getInt("backend", 0))?.let { prefs.put(PreferenceEntity(PrefKeys.BACKEND, it.name)) }
        }
        if (!user.isNullOrBlank() && model != null) prefs.put(PreferenceEntity(PrefKeys.SETUP_DONE, "true"))

        if (json.exists()) runCatching { parseChats(json.readText()) }.getOrNull()?.forEach { db.chats().save(it) }

        sp.edit().clear().apply()
        json.delete()
    }

    private fun parseChats(text: String): List<Conversation> {
        val arr = JSONArray(text)
        val d = ChatSettings()
        return List(arr.length()) { i ->
            val o = arr.getJSONObject(i)
            val s = o.optJSONObject("settings")
            val msgs = o.optJSONArray("messages") ?: JSONArray()
            Conversation(
                id = o.getLong("id"),
                title = o.optString("title", "New chat"),
                updatedAt = o.optLong("updatedAt"),
                settings = if (s == null) d else ChatSettings(
                    temperature = s.optDouble("temperature", d.temperature.toDouble()).toFloat(),
                    topK = s.optInt("topK", d.topK),
                    topP = s.optDouble("topP", d.topP.toDouble()).toFloat(),
                    repeatPenalty = s.optDouble("repeatPenalty", d.repeatPenalty.toDouble()).toFloat(),
                    repeatLastN = s.optInt("repeatLastN", d.repeatLastN),
                    maxNewTokens = s.optInt("maxNewTokens", d.maxNewTokens),
                    seed = s.optLong("seed", d.seed),
                    persona = s.optString("persona", d.persona),
                ),
                messages = List(msgs.length()) { k ->
                    val m = msgs.getJSONObject(k)
                    ChatMessage(m.getLong("id"), runCatching { Role.valueOf(m.getString("role")) }.getOrDefault(Role.Error),
                        m.optString("text"))
                },
            )
        }
    }
}
