package com.nekochat.ui

import android.app.Application
import android.content.Intent
import android.net.Uri
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.nekochat.data.ChatMessage
import com.nekochat.data.ChatSettings
import com.nekochat.data.Conversation
import com.nekochat.data.NekoDatabase
import com.nekochat.data.PrefKeys
import com.nekochat.data.PreferenceStore
import com.nekochat.data.Role
import com.nekochat.download.DnsMode
import com.nekochat.download.Downloads
import com.nekochat.download.ModelDownloadService
import com.nekochat.engine.ChatFormat
import com.nekochat.engine.ChatTurn
import com.nekochat.engine.ComputeBackend
import com.nekochat.engine.EngineInfo
import com.nekochat.engine.InferenceEngine
import com.nekochat.engine.LocalModel
import com.nekochat.engine.ModelRepository
import com.nekochat.engine.StopFilter
import com.nekochat.engine.StreamBuffer
import com.nekochat.gpt2.Gpt2ChatFormat
import com.nekochat.qwen3.Qwen3ChatFormat
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Boot shows nothing while Room loads (a few ms). Setup appears only until the first model is
 * chosen; afterwards the app opens on Chats and everything is changed from Settings.
 */
enum class Screen { Boot, Setup, Chats, Chat, ChatSettings, Settings, About, AddModels }

sealed interface LoadState {
    data object Idle : LoadState
    data class Loading(val progress: Float, val stage: String) : LoadState
    data class Ready(val info: EngineInfo, val modelName: String) : LoadState
    data class Failed(val message: String) : LoadState
}

@OptIn(ExperimentalCoroutinesApi::class)
class ChatViewModel(app: Application) : AndroidViewModel(app) {
    private val db = NekoDatabase.get(app)
    private val writes = Dispatchers.IO.limitedParallelism(1)  // keeps saves ordered
    private val prefs = PreferenceStore(db.preferences(), viewModelScope, writes)
    private val engine = InferenceEngine()
    val repository = ModelRepository(app)
    val downloads: ModelDownloadService = Downloads.get(app)

    /** null until the first scan finishes. */
    val models: StateFlow<List<LocalModel>?> =
        repository.observe().stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), null)

    /** Display name of the folder picked with the system folder picker. */
    var modelsFolderName by mutableStateOf<String?>(null)
        private set

    var screen by mutableStateOf(Screen.Boot)
        private set
    private var addModelsReturn = Screen.Settings
    var setupDone by mutableStateOf(false)
        private set
    var username by mutableStateOf("")
        private set
    var selectedModelPath by mutableStateOf<String?>(null)
        private set
    var backend by mutableStateOf(ComputeBackend.Auto)
        private set
    var dnsMode by mutableStateOf(DnsMode.System)
        private set

    private val _load = MutableStateFlow<LoadState>(LoadState.Idle)
    val load: StateFlow<LoadState> = _load.asStateFlow()

    val chats = mutableStateListOf<Conversation>()
    var activeId by mutableStateOf<Long?>(null)
        private set
    val active: Conversation? get() = chats.firstOrNull { it.id == activeId }

    /** Chat that the running generation belongs to (null when idle). */
    var generatingChatId by mutableStateOf<Long?>(null)
        private set
    val generating: Boolean get() = generatingChatId != null
    val stream = StreamBuffer()
    var lastSpeed by mutableStateOf<String?>(null)
        private set

    private var loadedKey: String? = null
    private var nextId = 1L

    init {
        viewModelScope.launch {
            withContext(Dispatchers.IO) {
                LegacyImport.run(app, db)
                prefs.load()
            }
            username = prefs[PrefKeys.USERNAME].orEmpty()
            selectedModelPath = prefs[PrefKeys.MODEL]
            backend = ComputeBackend.entries.firstOrNull { it.name == prefs[PrefKeys.BACKEND] } ?: ComputeBackend.Auto
            dnsMode = DnsMode.entries.firstOrNull { it.name == prefs[PrefKeys.DOWNLOAD_DNS] } ?: DnsMode.System
            downloads.setDnsMode(dnsMode)
            prefs[PrefKeys.MODELS_TREE]?.let { Uri.parse(it) }?.let { uri ->
                repository.setTree(uri)
                modelsFolderName = withContext(Dispatchers.IO) { repository.folderName(uri) }
            }
            val saved = withContext(Dispatchers.IO) { db.chats().loadAll() }
            chats.addAll(saved)
            nextId = 1 + (saved.flatMap { c -> c.messages.map { it.id } + c.id }.maxOrNull() ?: 0L)
            setupDone = prefs[PrefKeys.SETUP_DONE] == "true"
            if (setupDone) {
                screen = Screen.Chats
                autoLoad()
            } else {
                screen = Screen.Setup
            }
        }
    }

    // ------------------------------------------------------------------ navigation

    fun back() {
        screen = when (screen) {
            Screen.ChatSettings -> Screen.Chat
            Screen.Chat, Screen.Settings -> Screen.Chats
            Screen.About -> Screen.Settings
            Screen.AddModels -> addModelsReturn
            Screen.Chats, Screen.Setup, Screen.Boot -> screen
        }
    }

    /** Screens where the system back gesture should leave the app instead. */
    val atRoot: Boolean get() = screen == Screen.Chats || screen == Screen.Setup || screen == Screen.Boot

    fun openAppSettings() {
        screen = Screen.Settings
    }

    fun openAbout() {
        screen = Screen.About
    }

    fun openAddModels() {
        addModelsReturn = if (screen == Screen.Setup) Screen.Setup else Screen.Settings
        screen = Screen.AddModels
    }

    fun openChat(id: Long) {
        activeId = id
        screen = Screen.Chat
    }

    fun openChatSettings() {
        if (active != null) screen = Screen.ChatSettings
    }

    // ------------------------------------------------------------------ preferences

    fun updateUsername(value: String) {
        username = value.take(32)
        if (setupDone && username.isNotBlank()) prefs[PrefKeys.USERNAME] = username.trim()
    }

    /** Result of the system folder picker: keep read access across restarts and scan it. */
    fun setModelsFolder(uri: Uri) {
        val resolver = getApplication<Application>().contentResolver
        runCatching { resolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION) }
        repository.treeUri.value?.takeIf { it != uri }?.let { old ->
            runCatching { resolver.releasePersistableUriPermission(old, Intent.FLAG_GRANT_READ_URI_PERMISSION) }
        }
        prefs[PrefKeys.MODELS_TREE] = uri.toString()
        repository.setTree(uri)
        viewModelScope.launch { modelsFolderName = withContext(Dispatchers.IO) { repository.folderName(uri) } }
    }

    /** During setup this only marks the choice; afterwards it switches the loaded model. */
    fun selectModel(model: LocalModel) {
        if (!model.supported) return
        selectedModelPath = model.id
        if (setupDone) {
            prefs[PrefKeys.MODEL] = model.id
            ensureLoaded(model)
        }
    }

    fun selectBackend(b: ComputeBackend) {
        backend = b
        if (setupDone) {
            prefs[PrefKeys.BACKEND] = b.name
            models.value?.firstOrNull { it.id == selectedModelPath && it.supported }?.let(::ensureLoaded)
        }
    }

    fun selectDnsMode(mode: DnsMode) {
        dnsMode = mode
        prefs[PrefKeys.DOWNLOAD_DNS] = mode.name
        downloads.setDnsMode(mode)
    }

    fun startChatting(model: LocalModel) {
        username = username.trim()
        selectedModelPath = model.id
        prefs[PrefKeys.USERNAME] = username
        prefs[PrefKeys.MODEL] = model.id
        prefs[PrefKeys.BACKEND] = backend.name
        prefs[PrefKeys.SETUP_DONE] = "true"
        setupDone = true
        ensureLoaded(model)
        if (chats.isEmpty()) newChat() else screen = Screen.Chats
    }

    /** "Use model" on a finished download. */
    fun useDownloadedModel(folder: String) {
        viewModelScope.launch {
            val path = File(folder).absolutePath
            val model = withContext(Dispatchers.IO) { repository.scan() }.firstOrNull { it.id == path && it.supported }
                ?: return@launch
            selectModel(model)
            selectedModelPath = model.id
            screen = if (setupDone) Screen.Chats else Screen.Setup
        }
    }

    // ------------------------------------------------------------------ chats

    fun newChat() {
        // Reuse an untouched chat instead of piling up empty ones.
        chats.firstOrNull { it.messages.isEmpty() }?.let {
            openChat(it.id)
            return
        }
        // New chats inherit the most recent chat's settings, so tuning carries over.
        val settings = chats.firstOrNull()?.settings ?: ChatSettings()
        val c = Conversation(nextId++, "New chat", System.currentTimeMillis(), settings, emptyList())
        chats.add(0, c)
        persist(c)
        openChat(c.id)
    }

    fun deleteChat(id: Long) {
        if (generatingChatId == id) engine.cancel()
        chats.removeAll { it.id == id }
        if (activeId == id) activeId = null
        viewModelScope.launch(writes) { db.chats().delete(id) }
    }

    fun clearActiveChat() {
        val id = activeId ?: return
        if (generatingChatId == id) engine.cancel()
        update(id) { it.copy(messages = emptyList(), title = "New chat") }
    }

    fun updateSettings(transform: (ChatSettings) -> ChatSettings) {
        val id = activeId ?: return
        update(id) { it.copy(settings = transform(it.settings)) }
    }

    fun renameActive(title: String) {
        val id = activeId ?: return
        update(id) { it.copy(title = title.take(60)) }
    }

    // ------------------------------------------------------------------ model

    fun retryLoad() {
        loadedKey = null
        autoLoad()
    }

    /** Loads the saved model once the folders have been scanned. */
    private fun autoLoad() {
        viewModelScope.launch {
            _load.value = LoadState.Loading(0f, "Looking for your model")
            val list = models.first { it != null }.orEmpty()
            val model = list.firstOrNull { it.id == selectedModelPath }
            when {
                model == null -> _load.value = LoadState.Failed("Your model wasn't found. Choose one in Settings.")
                !model.supported -> _load.value = LoadState.Failed(model.problem ?: "This model isn't supported.")
                else -> ensureLoaded(model)
            }
        }
    }

    private fun ensureLoaded(model: LocalModel) {
        val key = "${model.id}|${backend.id}"
        if (key == loadedKey && _load.value !is LoadState.Failed) return
        loadedKey = key
        if (generating) engine.cancel()
        lastSpeed = null
        _load.value = LoadState.Loading(0f, "Preparing")
        viewModelScope.launch {
            _load.value = try {
                // Weights are copied into engine-owned buffers, so the folder access only lives for the load.
                val info = withContext(Dispatchers.IO) { repository.open(model) }.use { access ->
                    engine.load(access.dir, backend) { f, s -> _load.value = LoadState.Loading(f, s) }
                }
                LoadState.Ready(info, model.name)
            } catch (e: Throwable) {
                loadedKey = null
                LoadState.Failed(e.message ?: e.toString())
            }
        }
    }

    private fun chatFormat(): ChatFormat = when ((_load.value as? LoadState.Ready)?.info?.architecture) {
        "qwen3" -> Qwen3ChatFormat
        else -> Gpt2ChatFormat
    }

    // ------------------------------------------------------------------ generation

    fun send(text: String) {
        val msg = text.trim()
        val chat = active ?: return
        if (msg.isEmpty() || generating || _load.value !is LoadState.Ready) return
        val chatId = chat.id
        update(chatId) {
            it.copy(
                messages = it.messages + ChatMessage(nextId++, Role.User, msg),
                title = if (it.messages.isEmpty()) msg.take(40) else it.title,
            )
        }
        generatingChatId = chatId
        stream.reset()
        viewModelScope.launch {
            try {
                val format = chatFormat()
                val settings = chats.first { it.id == chatId }.settings
                val params = settings.toParams()
                val prompt = buildPrompt(format, chats.first { it.id == chatId }, params.maxNewTokens)
                val filter = StopFilter(format.stopSequences(username))
                val stats = engine.generate(prompt, params) { piece ->
                    stream.publish(filter.push(piece))
                    !filter.stopped
                }
                val reply = filter.finalText().trim()
                appendMessage(chatId, ChatMessage(nextId++, Role.Assistant, reply.ifEmpty { "…" }))
                lastSpeed = "%.1f tok/s".format(stats.tokensPerSecond)
            } catch (e: Throwable) {
                appendMessage(chatId, ChatMessage(nextId++, Role.Error, e.message ?: e.toString()))
            } finally {
                generatingChatId = null
            }
        }
    }

    fun stop() = engine.cancel()

    /**
     * Renders the transcript, dropping the oldest turns until prompt + reply fit the context. At most
     * half the context is reserved for the reply, so long reply limits never erase the conversation.
     */
    private suspend fun buildPrompt(format: ChatFormat, chat: Conversation, maxNew: Int): IntArray {
        val turns = mutableListOf<ChatTurn>()
        var pendingUser: String? = null
        for (m in chat.messages) {
            when (m.role) {
                Role.User -> pendingUser = m.text
                Role.Assistant -> pendingUser?.let { turns += ChatTurn(it, m.text); pendingUser = null }
                Role.Error -> Unit
            }
        }
        turns += ChatTurn(pendingUser ?: "", null)
        val ctx = engine.contextLength()
        val reserve = minOf(maxNew, ctx / 2)
        var start = 0
        while (true) {
            val text = format.build(username, chat.settings.persona, turns.subList(start, turns.size))
            val tokens = engine.tokenize(text)
            if (tokens.size + reserve <= ctx || start >= turns.size - 1) return tokens
            start++
        }
    }

    private fun appendMessage(chatId: Long, m: ChatMessage) {
        update(chatId) { it.copy(messages = it.messages + m) }
    }

    private fun update(id: Long, transform: (Conversation) -> Conversation) {
        val i = chats.indexOfFirst { it.id == id }
        if (i < 0) return
        val c = transform(chats[i]).copy(updatedAt = System.currentTimeMillis())
        chats[i] = c
        persist(c)
    }

    private fun persist(c: Conversation) {
        viewModelScope.launch(writes) { db.chats().save(c) }
    }

    override fun onCleared() {
        engine.close()
    }
}
