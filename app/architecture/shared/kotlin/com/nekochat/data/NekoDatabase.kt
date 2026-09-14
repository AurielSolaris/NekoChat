package com.nekochat.data

import android.content.Context
import androidx.room.Dao
import androidx.room.Database
import androidx.room.Embedded
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.Transaction
import androidx.room.Upsert
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.launch
import java.util.concurrent.ConcurrentHashMap

// ---------------------------------------------------------------- preferences

/** One user choice (name, model, backend, models folder, ...) as a key/value row. */
@Entity(tableName = "preferences")
data class PreferenceEntity(@PrimaryKey val key: String, val value: String)

@Dao
interface PreferenceDao {
    @Query("SELECT * FROM preferences")
    suspend fun all(): List<PreferenceEntity>

    @Upsert
    suspend fun put(p: PreferenceEntity)

    @Query("DELETE FROM preferences WHERE `key` = :key")
    suspend fun remove(key: String)
}

object PrefKeys {
    const val USERNAME = "username"
    const val MODEL = "model"
    const val BACKEND = "backend"
    const val MODELS_TREE = "models_tree"
    const val SETUP_DONE = "setup_done"
    const val DOWNLOAD_DNS = "download_dns"
}

/** In-memory view of the preferences table; writes go to Room in order on [writes]. */
class PreferenceStore(
    private val dao: PreferenceDao,
    private val scope: CoroutineScope,
    private val writes: CoroutineDispatcher,
) {
    private val values = ConcurrentHashMap<String, String>()

    suspend fun load() {
        dao.all().forEach { values[it.key] = it.value }
    }

    operator fun get(key: String): String? = values[key]

    operator fun set(key: String, value: String?) {
        if (value == null) values.remove(key) else values[key] = value
        scope.launch(writes) { if (value == null) dao.remove(key) else dao.put(PreferenceEntity(key, value)) }
    }
}

// ---------------------------------------------------------------- chats

@Entity(tableName = "chats")
data class ChatEntity(
    @PrimaryKey val id: Long,
    val title: String,
    val updatedAt: Long,
    @Embedded(prefix = "settings_") val settings: ChatSettings,
)

@Entity(
    tableName = "messages",
    foreignKeys = [ForeignKey(entity = ChatEntity::class, parentColumns = ["id"], childColumns = ["chatId"],
        onDelete = ForeignKey.CASCADE)],
    indices = [Index("chatId")],
)
data class MessageEntity(
    @PrimaryKey val id: Long,
    val chatId: Long,
    val position: Int,
    val role: String,
    val text: String,
)

@Dao
abstract class ChatDao {
    @Query("SELECT * FROM chats ORDER BY updatedAt DESC")
    protected abstract suspend fun chats(): List<ChatEntity>

    @Query("SELECT * FROM messages ORDER BY chatId, position")
    protected abstract suspend fun messages(): List<MessageEntity>

    // Upsert (not REPLACE): replacing a chat row would cascade-delete its messages.
    @Upsert
    protected abstract suspend fun upsertChat(chat: ChatEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    protected abstract suspend fun insertMessages(messages: List<MessageEntity>)

    @Query("DELETE FROM messages WHERE chatId = :chatId")
    protected abstract suspend fun clearMessages(chatId: Long)

    @Query("DELETE FROM chats WHERE id = :id")
    abstract suspend fun delete(id: Long)

    @Transaction
    open suspend fun loadAll(): List<Conversation> {
        val byChat = messages().groupBy { it.chatId }
        return chats().map { c ->
            Conversation(c.id, c.title, c.updatedAt, c.settings, byChat[c.id].orEmpty().map { m ->
                ChatMessage(m.id, runCatching { Role.valueOf(m.role) }.getOrDefault(Role.Error), m.text)
            })
        }
    }

    @Transaction
    open suspend fun save(c: Conversation) {
        upsertChat(ChatEntity(c.id, c.title, c.updatedAt, c.settings))
        clearMessages(c.id)
        insertMessages(c.messages.mapIndexed { i, m -> MessageEntity(m.id, c.id, i, m.role.name, m.text) })
    }
}

// ---------------------------------------------------------------- downloads

/** A model download, kept so it can be resumed after the app restarts. */
@Entity(tableName = "downloads")
data class DownloadEntity(
    @PrimaryKey val repoId: String,
    val folderName: String,
    val status: String,
    val bytesTotal: Long,
    /** JSON array of {path, size} for the files being fetched. */
    val files: String,
    val error: String?,
    val createdAt: Long,
)

@Dao
interface DownloadDao {
    @Query("SELECT * FROM downloads ORDER BY createdAt")
    suspend fun all(): List<DownloadEntity>

    @Upsert
    suspend fun put(d: DownloadEntity)

    @Query("DELETE FROM downloads WHERE repoId = :repoId")
    suspend fun remove(repoId: String)
}

// ---------------------------------------------------------------- database

@Database(
    entities = [PreferenceEntity::class, ChatEntity::class, MessageEntity::class, DownloadEntity::class],
    version = 1,
    exportSchema = true,
)
abstract class NekoDatabase : RoomDatabase() {
    abstract fun preferences(): PreferenceDao
    abstract fun chats(): ChatDao
    abstract fun downloads(): DownloadDao

    companion object {
        @Volatile private var instance: NekoDatabase? = null

        fun get(context: Context): NekoDatabase = instance ?: synchronized(this) {
            instance ?: Room.databaseBuilder(context.applicationContext, NekoDatabase::class.java, "nekochat.db")
                .build()
                .also { instance = it }
        }
    }
}
