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
import com.nekochat.download.DownloadEngine
import com.nekochat.download.aria2Available
import com.nekochat.speech.DictationFiles
import com.nekochat.speech.DictationModel
import com.nekochat.speech.VoiceInput
import com.nekochat.download.DownloadKind
import com.nekochat.download.DownloadStatus
import com.nekochat.download.Downloads
import com.nekochat.download.ModelDownload
import com.nekochat.download.ModelDownloadService
import com.nekochat.speech.KokoroVoiceInfo
import com.nekochat.speech.KokoroVoices
import com.nekochat.speech.SpeechChoice
import com.nekochat.speech.SpeechManager
import kotlinx.coroutines.flow.map
import kotlin.math.roundToInt
import com.nekochat.engine.ChatFormat
import com.nekochat.engine.ChatTurn
import com.nekochat.engine.ComputeBackend
import com.nekochat.engine.EngineInfo
import com.nekochat.engine.InferenceEngine
import com.nekochat.engine.LocalModel
import com.nekochat.engine.MemoryUsage
import com.nekochat.engine.ModelRepository
import com.nekochat.engine.RealmMarker
import com.nekochat.gpt2.RealmChatFormat
import com.nekochat.engine.StopFilter
import com.nekochat.engine.StreamBuffer
import com.nekochat.engine.WeightPrecision
import com.nekochat.gpt2.Gpt2ChatFormat
import com.nekochat.qwen3.Qwen3ChatFormat
import com.nekochat.granite.GraniteChatFormat
import com.nekochat.ui.theme.Neko
import com.nekochat.ui.theme.NekoTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Boot shows nothing while Room loads (a few ms). Setup appears only until the first model is
 * chosen; afterwards the app opens on Chats and everything is changed from Settings.
 */
enum class Screen(val spoken: String) {
    Boot(""), Setup("Setup"), Chats("Chats"), Chat("Chat"), ChatSettings("Chat settings"), Settings("Settings"),
    About("About NekoChat"), AddModels("Download models"), Themes("Theme"), Accessibility("Accessibility"),
}

sealed interface LoadState {
    data object Idle : LoadState
    data class Loading(val progress: Float, val stage: String) : LoadState
    data class Ready(val info: EngineInfo, val modelName: String, val realm: RealmMarker? = null) : LoadState
    data class Failed(val message: String) : LoadState
}

@OptIn(ExperimentalCoroutinesApi::class)
/** How long the speech-to-text model stays in memory after the last dictation. */
private const val DICTATION_KEEP_MS = 2 * 60_000L

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
    /** False in builds without the aria2 binary; only Fetch is offered then. */
    val hasAria2 = aria2Available(app)
    var downloadEngine by mutableStateOf(if (hasAria2) DownloadEngine.Aria2Multi else DownloadEngine.Fetch)
        private set
    var precision by mutableStateOf(WeightPrecision.FP16)
        private set
    val theme: NekoTheme get() = Neko.theme

    private val _load = MutableStateFlow<LoadState>(LoadState.Idle)
    val load: StateFlow<LoadState> = _load.asStateFlow()

    /** Polled once a second while a screen shows it; never touches the (possibly busy) inference thread. */
    val memory: StateFlow<MemoryUsage?> = flow {
        while (true) {
            emit(engine.memory())
            delay(1000)
        }
    }.flowOn(Dispatchers.Default).stateIn(viewModelScope, SharingStarted.WhileSubscribed(1000), null)

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
            Neko.theme = NekoTheme.entries.firstOrNull { it.name == prefs[PrefKeys.THEME] } ?: NekoTheme.Rose
            readReplies = prefs[PrefKeys.READ_REPLIES] == "true"
            narrate = prefs[PrefKeys.NARRATE] == "true"
            speechChoice = SpeechChoice.entries.firstOrNull { it.name == prefs[PrefKeys.SPEECH_ENGINE] } ?: SpeechChoice.Auto
            speechVoice = KokoroVoices.byId(prefs[PrefKeys.SPEECH_VOICE])
            speechSpeed = prefs[PrefKeys.SPEECH_SPEED]?.toFloatOrNull() ?: 1f
            speechPrecision = WeightPrecision.entries.firstOrNull { it.name == prefs[PrefKeys.SPEECH_WEIGHTS] } ?: WeightPrecision.FP8
            speech.choice = speechChoice
            speech.voice = speechVoice
            speech.speed = speechSpeed
            speech.precision = speechPrecision
            dictationModel = DictationModel.byName(prefs[PrefKeys.DICTATION_MODEL])
            dictationPrecision = WeightPrecision.entries.firstOrNull { it.name == prefs[PrefKeys.DICTATION_WEIGHTS] } ?: WeightPrecision.FP8
            refreshVoiceInstalled()
            if (readReplies) speech.preload()
            // Started here, not at the top of init: properties declared below init only exist once the
            // constructor has finished (this coroutine resumed after loading prefs, so it has).
            launch {
                dictationDownloads.collect { list ->
                    if (list.any { it.status == DownloadStatus.Completed } && dictationInstalled.size < DictationModel.entries.size) {
                        val before = dictationInstalled
                        refreshVoiceInstalled()
                        if (dictationInstalled != before) say("Voice input downloaded")
                    }
                }
            }
            launch {
                voiceDownload.collect { d ->
                    if (d?.status == DownloadStatus.Completed && !kokoroInstalled) {
                        refreshVoiceInstalled()
                        say("Voice downloaded")
                    }
                }
            }
            precision = WeightPrecision.entries.firstOrNull { it.name == prefs[PrefKeys.WEIGHTS] } ?: WeightPrecision.FP16
            username = prefs[PrefKeys.USERNAME].orEmpty()
            selectedModelPath = prefs[PrefKeys.MODEL]
            backend = ComputeBackend.entries.firstOrNull { it.name == prefs[PrefKeys.BACKEND] } ?: ComputeBackend.Auto
            dnsMode = DnsMode.entries.firstOrNull { it.name == prefs[PrefKeys.DOWNLOAD_DNS] } ?: DnsMode.System
            downloads.setDnsMode(dnsMode)
            downloadEngine = DownloadEngine.entries.firstOrNull { it.name == prefs[PrefKeys.DOWNLOAD_ENGINE] }
                ?.takeIf { hasAria2 || it == DownloadEngine.Fetch }
                ?: if (hasAria2) DownloadEngine.Aria2Multi else DownloadEngine.Fetch
            downloads.setEngine(downloadEngine)
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
        // Back leaves edit mode first, like closing any other transient mode.
        if (screen == Screen.Chat && editing != null) {
            cancelEditing()
            return
        }
        screen = when (screen) {
            Screen.ChatSettings -> Screen.Chat
            Screen.Chat, Screen.Settings -> Screen.Chats
            Screen.About, Screen.Themes -> Screen.Settings
            Screen.Accessibility -> accessibilityReturn
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

    fun openThemes() {
        screen = Screen.Themes
    }

    private var accessibilityReturn = Screen.Settings

    fun openAccessibility() {
        accessibilityReturn = if (screen == Screen.Setup) Screen.Setup else Screen.Settings
        screen = Screen.Accessibility
    }

    // ------------------------------------------------------------------ speech (read aloud, narration)

    val speech = SpeechManager(app)
    var readReplies by mutableStateOf(false)
        private set
    var narrate by mutableStateOf(false)
        private set
    var speechChoice by mutableStateOf(SpeechChoice.Auto)
        private set
    var speechVoice by mutableStateOf(KokoroVoices.byId(null))
        private set
    var speechSpeed by mutableStateOf(1f)
        private set
    var speechPrecision by mutableStateOf(WeightPrecision.FP8)
        private set
    var kokoroInstalled by mutableStateOf(false)
        private set
    val systemVoiceAvailable: Boolean by lazy { speech.systemAvailable }

    /** Shown when reading aloud is switched on but no voice can speak yet: offers the Kokoro download. */
    var speechOffer by mutableStateOf(false)
        private set

    /** The Kokoro download, if one was started. */
    val voiceDownload: StateFlow<ModelDownload?> = downloads.downloads
        .map { list -> list.lastOrNull { it.kind == DownloadKind.Speech } }
        .stateIn(viewModelScope, SharingStarted.Eagerly, null)

    private val speechUsable: Boolean get() = speech.activeChoice() != null

    fun updateReadReplies(on: Boolean) {
        readReplies = on
        prefs[PrefKeys.READ_REPLIES] = on.toString()
        if (on && !speechUsable) speechOffer = true
        if (on) speech.preload()
        if (!on) speech.stop()
    }

    fun updateNarrate(on: Boolean) {
        narrate = on
        prefs[PrefKeys.NARRATE] = on.toString()
        if (on && !speechUsable) speechOffer = true
        if (on) say("Narration on")
    }

    fun selectSpeechChoice(c: SpeechChoice) {
        speechChoice = c
        speech.choice = c
        prefs[PrefKeys.SPEECH_ENGINE] = c.name
    }

    fun selectVoice(v: KokoroVoiceInfo) {
        speechVoice = v
        speech.voice = v
        prefs[PrefKeys.SPEECH_VOICE] = v.id
        playSample()
    }

    fun updateSpeechSpeed(s: Float) {
        speechSpeed = (s * 10).roundToInt() / 10f
        speech.speed = speechSpeed
        prefs[PrefKeys.SPEECH_SPEED] = speechSpeed.toString()
    }

    fun selectSpeechPrecision(p: WeightPrecision) {
        speechPrecision = p
        speech.precision = p
        prefs[PrefKeys.SPEECH_WEIGHTS] = p.name
    }

    fun downloadVoice() {
        speechOffer = false
        downloads.start(KokoroVoices.REPO, DownloadKind.Speech)
    }

    fun dismissSpeechOffer() {
        speechOffer = false
    }

    fun playSample() {
        speech.speak("Hi ${username.ifBlank { "there" }}! This is how I sound when I read Neko's replies.")
    }

    /** Reads one message aloud (the speaker button on a reply). */
    fun speakMessage(text: String) {
        if (!speechUsable) {
            speechOffer = true
            return
        }
        speech.speak(text)
    }

    fun stopSpeaking() = speech.stop()

    /** Narration of screens and actions (Accessibility → Narrate actions). */
    private fun say(text: String) {
        if (narrate && text.isNotBlank()) speech.speak(text)
    }

    /** Called by the UI whenever a screen appears. */
    fun onScreenShown(s: Screen) {
        if (s == Screen.Chat) say(active?.title?.let { "Chat: $it" } ?: s.spoken) else say(s.spoken)
    }

    private fun refreshVoiceInstalled() {
        kokoroInstalled = speech.kokoroInstalled
        dictationInstalled = DictationModel.entries.filter { DictationFiles.installed(getApplication(), it) }.toSet()
    }

    // ------------------------------------------------------------------ voice input

    enum class Dictation { Idle, Recording, Transcribing }

    private val voiceInput = VoiceInput(app)
    private var releaseDictation: Job? = null

    var dictation by mutableStateOf(Dictation.Idle)
        private set
    var dictationModel by mutableStateOf(DictationModel.GraniteNc)
        private set
    var dictationInstalled by mutableStateOf<Set<DictationModel>>(emptySet())
        private set
    var dictationPrecision by mutableStateOf(WeightPrecision.FP8)
        private set
    /** Shown when the mic is tapped before a speech model is downloaded. */
    var dictationOffer by mutableStateOf(false)
        private set
    var dictationError by mutableStateOf<String?>(null)
        private set

    /** Speech-to-text downloads, newest per model. */
    val dictationDownloads: StateFlow<List<ModelDownload>> = downloads.downloads
        .map { list -> list.filter { it.kind == DownloadKind.Dictation } }
        .stateIn(viewModelScope, SharingStarted.Eagerly, emptyList())

    val dictationReady: Boolean get() = dictationModel in dictationInstalled
    val dictationLevel: Float get() = voiceInput.level
    val dictationWeightBytes: Long get() = voiceInput.weightBytes

    fun selectDictationModel(m: DictationModel) {
        dictationModel = m
        prefs[PrefKeys.DICTATION_MODEL] = m.name
        voiceInput.release()
    }

    fun selectDictationPrecision(p: WeightPrecision) {
        dictationPrecision = p
        prefs[PrefKeys.DICTATION_WEIGHTS] = p.name
        voiceInput.release()
    }

    fun downloadDictation(m: DictationModel = dictationModel) {
        dictationOffer = false
        if (m != dictationModel) selectDictationModel(m)
        downloads.start(m.repo, DownloadKind.Dictation)
    }

    fun dismissDictationOffer() {
        dictationOffer = false
    }

    fun clearDictationError() {
        dictationError = null
    }

    /** Mic button pressed with the permission granted: starts recording, or offers the download first. */
    fun startDictation() {
        if (dictation != Dictation.Idle) return
        if (!dictationReady) {
            dictationOffer = true
            return
        }
        releaseDictation?.cancel()
        speech.stop()
        if (!voiceInput.start()) {
            dictationError = "The microphone is in use or unavailable."
            return
        }
        dictation = Dictation.Recording
        voiceInput.preload(dictationModel, dictationPrecision)  // loads while you speak
    }

    /** Stops recording and transcribes; [onText] gets the words (on the main thread). */
    fun stopDictation(onText: (String) -> Unit) {
        if (dictation != Dictation.Recording) return
        val audio = voiceInput.stop()
        if (audio.size < VoiceInput.SAMPLE_RATE / 4) {  // a tap, not speech
            dictation = Dictation.Idle
            return
        }
        dictation = Dictation.Transcribing
        voiceInput.transcribe(audio, dictationModel, dictationPrecision) { result ->
            viewModelScope.launch {
                dictation = Dictation.Idle
                result.onSuccess { if (it.isNotBlank()) onText(it) else dictationError = "Didn't catch that. Try again a little closer." }
                    .onFailure { dictationError = "Voice input failed: ${it.message ?: it}" }
                // The speech model holds ~0.5 GB: let it go once dictation has been idle for a while.
                releaseDictation?.cancel()
                releaseDictation = launch {
                    delay(DICTATION_KEEP_MS)
                    voiceInput.release()
                }
            }
        }
    }

    /** The microphone permission was refused. */
    fun micDenied() {
        dictationError = "Voice input needs microphone access. You can allow it in App info → Permissions."
    }

    fun cancelDictation() {
        if (dictation == Dictation.Recording) voiceInput.cancel()
        dictation = Dictation.Idle
    }

    fun openAddModels() {
        addModelsReturn = if (screen == Screen.Setup) Screen.Setup else Screen.Settings
        screen = Screen.AddModels
    }

    fun openChat(id: Long) {
        // An edit belongs to its own chat; never carry it into another one.
        if (id != activeId) editing = null
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

    /** Reloads the current model in the new precision. */
    fun selectPrecision(p: WeightPrecision) {
        precision = p
        prefs[PrefKeys.WEIGHTS] = p.name
        if (setupDone) models.value?.firstOrNull { it.id == selectedModelPath && it.supported }?.let(::ensureLoaded)
    }

    fun selectTheme(t: NekoTheme) {
        Neko.theme = t
        prefs[PrefKeys.THEME] = t.name
    }

    fun selectDnsMode(mode: DnsMode) {
        dnsMode = mode
        prefs[PrefKeys.DOWNLOAD_DNS] = mode.name
        downloads.setDnsMode(mode)
    }

    fun selectDownloadEngine(engine: DownloadEngine) {
        downloadEngine = engine
        prefs[PrefKeys.DOWNLOAD_ENGINE] = engine.name
        downloads.setEngine(engine)
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
        // New chats inherit the most recent chat's settings, so tuning carries over; ReALM chats start
        // from ReALM's own sampling defaults.
        val settings = if (realmLoaded) ChatSettings.REALM else chats.firstOrNull()?.settings ?: ChatSettings()
        val c = Conversation(nextId++, "New chat", System.currentTimeMillis(), settings, emptyList())
        chats.add(0, c)
        persist(c)
        openChat(c.id)
    }

    fun deleteChat(id: Long) {
        if (generatingChatId == id) engine.cancel()
        chats.removeAll { it.id == id }
        if (activeId == id) {
            activeId = null
            editing = null
        }
        viewModelScope.launch(writes) { db.chats().delete(id) }
        say("Chat deleted")
    }

    fun clearActiveChat() {
        val id = activeId ?: return
        if (generatingChatId == id) engine.cancel()
        update(id) { it.copy(messages = emptyList(), title = "New chat") }
        say("Messages cleared")
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
        val key = "${model.id}|${backend.id}|${precision.id}"
        if (key == loadedKey && _load.value !is LoadState.Failed) return
        loadedKey = key
        if (generating) engine.cancel()
        lastSpeed = null
        _load.value = LoadState.Loading(0f, "Preparing")
        viewModelScope.launch {
            _load.value = try {
                // Weights are copied into engine-owned buffers, so the folder access only lives for the load.
                val info = withContext(Dispatchers.IO) { repository.open(model) }.use { access ->
                    engine.load(access.dir, backend, precision) { f, s -> _load.value = LoadState.Loading(f, s) }
                }
                LoadState.Ready(info, model.name, model.realm).also { say("Model ready") }
            } catch (e: Throwable) {
                loadedKey = null
                say("The model failed to load")
                LoadState.Failed(e.message ?: e.toString())
            }
        }
    }

    private fun chatFormat(): ChatFormat {
        val ready = _load.value as? LoadState.Ready
        return when {
            ready?.realm != null -> RealmChatFormat
            ready?.info?.architecture == "qwen3" -> Qwen3ChatFormat
            ready?.info?.architecture == "granitemoehybrid" -> GraniteChatFormat
            else -> Gpt2ChatFormat
        }
    }

    private val realmLoaded: Boolean get() = (_load.value as? LoadState.Ready)?.realm != null

    /** False for formats (ReALM) that ignore the persona text. */
    val usesPersona: Boolean get() = chatFormat().usesPersona

    /** Defaults for the loaded model: ReALM's own sampling for ReALM checkpoints. */
    val defaultSettings: ChatSettings get() = if (realmLoaded) ChatSettings.REALM else ChatSettings()

    fun resetSettings() = updateSettings { defaultSettings }

    // ------------------------------------------------------------------ generation

    /** The user message being edited in the input bar, or null. */
    var editing by mutableStateOf<ChatMessage?>(null)
        private set

    private val canGenerate: Boolean get() = !generating && _load.value is LoadState.Ready

    fun send(text: String) {
        val msg = text.trim()
        val chat = active ?: return
        if (msg.isEmpty() || !canGenerate) return
        editing?.let {
            editMessage(it.id, msg)
            return
        }
        update(chat.id) {
            it.copy(
                messages = it.messages + ChatMessage(nextId++, Role.User, msg),
                title = if (it.messages.isEmpty()) msg.take(40) else it.title,
            )
        }
        speech.stop()
        say("Message sent")
        generate(chat.id)
    }

    /**
     * Replaces the last reply (or error) with a new one. Everything before it is unchanged, so the
     * engine's KV cache covers the whole prompt and only the new reply is computed.
     */
    fun regenerate() {
        val chat = active ?: return
        val last = chat.messages.lastOrNull() ?: return
        if (!canGenerate) return
        if (last.role != Role.User) update(chat.id) { it.copy(messages = it.messages.dropLast(1)) }
        generate(chat.id)
    }

    fun startEditing(m: ChatMessage) {
        if (m.role == Role.User && !generating) editing = m
    }

    fun cancelEditing() {
        editing = null
    }

    /** Messages that editing [m] would replace (everything after it). */
    fun messagesAfter(m: ChatMessage): Int {
        val msgs = active?.messages ?: return 0
        val i = msgs.indexOfFirst { it.id == m.id }
        return if (i < 0) 0 else msgs.size - i - 1
    }

    /**
     * Changes a user message, drops everything after it and answers again. The engine keeps the KV
     * cache for the tokens before the first changed one and recomputes only from there.
     */
    fun editMessage(id: Long, text: String) {
        val msg = text.trim()
        val chat = active ?: return
        val i = chat.messages.indexOfFirst { it.id == id && it.role == Role.User }
        editing = null
        if (i < 0 || msg.isEmpty() || !canGenerate) return
        update(chat.id) { c -> c.copy(messages = c.messages.take(i) + c.messages[i].copy(text = msg)) }
        generate(chat.id)
    }

    /** Answers the conversation as it stands (its last message is the user's). */
    private fun generate(chatId: Long) {
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
                if (readReplies && reply.isNotEmpty()) speech.speak(reply) else say("Reply ready")
            } catch (e: Throwable) {
                appendMessage(chatId, ChatMessage(nextId++, Role.Error, e.message ?: e.toString()))
                say("Something went wrong")
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
        format.contextTokens?.let { window ->
            // Token-stream models: keep the newest tokens, wherever that cuts.
            val tokens = engine.tokenize(format.build(username, chat.settings.persona, turns))
            val limit = minOf(window, ctx - minOf(maxNew, ctx / 2))
            return if (tokens.size > limit) tokens.copyOfRange(tokens.size - limit, tokens.size) else tokens
        }
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
        speech.close()
        voiceInput.close()
    }
}
