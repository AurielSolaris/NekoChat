package com.nekochat.download

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkInfo
import android.util.Log
import com.nekochat.data.DownloadDao
import com.nekochat.data.DownloadEntity
import com.nekochat.data.NekoDatabase
import com.nekochat.engine.SUPPORTED_ARCHITECTURES
import com.nekochat.engine.appModelsDir
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * [ModelDownloadService] for the Hugging Face Hub. It resolves which files a model needs,
 * stages them in models/.downloads/<name> (ignored by the model scanner), tracks progress,
 * persists downloads in Room, and moves the folder into models/ once every file is complete,
 * so half-downloaded models never show up as loadable.
 *
 * The bytes are moved by a [TransferEngine]: aria2 (multi- or single-connection) or Fetch.
 * Each download keeps the engine it started with, because partial files are engine-specific.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class HubDownloadService(private val context: Context) : ModelDownloadService {
    private val modelsDir = appModelsDir(context)
    private val stagingDir = File(modelsDir, ".downloads")
    private val dao: DownloadDao = NekoDatabase.get(context).downloads()

    // Every state change runs on one thread, so tasks and engines need no locking.
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO.limitedParallelism(1))
    private val state = MutableStateFlow<List<ModelDownload>>(emptyList())
    override val downloads: StateFlow<List<ModelDownload>> = state.asStateFlow()

    private val aria2 = Aria2Engine(context, modelsDir)
    private val fetch = FetchEngine()
    private var defaultEngine = DownloadEngine.Aria2Multi

    private class Task(
        val repoId: String,
        val folderName: String,
        val files: List<HuggingFaceHub.RepoFile>,
        val createdAt: Long,
        val engine: DownloadEngine,
    ) {
        val handles = mutableMapOf<String, String>()  // path -> engine handle
    }

    private val tasks = LinkedHashMap<String, Task>()
    private var poller: Job? = null

    init {
        scope.launch {
            for (e in dao.all()) {
                val engine = DownloadEngine.entries.firstOrNull { it.name == e.engine } ?: DownloadEngine.Aria2Multi
                val task = Task(e.repoId, e.folderName, parseFiles(e.files), e.createdAt, engine)
                val status = runCatching { DownloadStatus.valueOf(e.status) }.getOrDefault(DownloadStatus.Failed)
                    .let { if (it == DownloadStatus.Preparing || it == DownloadStatus.Downloading) DownloadStatus.Paused else it }
                if (status != DownloadStatus.Completed) tasks[e.repoId] = task
                put(ModelDownload(e.repoId, status, bytesDone = if (status == DownloadStatus.Completed) e.bytesTotal else onDisk(task),
                    bytesTotal = e.bytesTotal, error = e.error, engine = engine,
                    folder = if (status == DownloadStatus.Completed) File(modelsDir, e.folderName).path else null))
            }
        }
    }

    // ------------------------------------------------------------ ModelDownloadService

    override fun start(repoId: String) {
        val id = repoId.trim().removePrefix("https://").removePrefix("huggingface.co/").trim('/')
        scope.launch {
            val existing = state.value.firstOrNull { it.repoId == id }
            if (existing != null && existing.status != DownloadStatus.Completed) {
                if (!existing.active) resumeTask(id)
                return@launch
            }
            if (!REPO.matches(id)) {
                put(ModelDownload(id, DownloadStatus.Failed, error = "Use the form owner/model, e.g. Qwen/Qwen3-0.6B."))
                return@launch
            }
            val engine = defaultEngine
            put(ModelDownload(id, DownloadStatus.Preparing, engine = engine))
            try {
                checkNetworkAllowed()
                val files = HuggingFaceHub.modelFiles(id)
                val arch = HuggingFaceHub.modelType(id)
                if (arch !in SUPPORTED_ARCHITECTURES) {
                    throw DownloadException("This is a \"$arch\" model. NekoChat runs GPT-2 and Qwen3 models.")
                }
                val task = Task(id, uniqueFolderName(id.substringAfterLast('/')), files, System.currentTimeMillis(), engine)
                tasks[id] = task
                put(ModelDownload(id, DownloadStatus.Downloading, bytesTotal = files.sumOf { it.size }, engine = engine))
                launch(task)
            } catch (e: Exception) {
                fail(id, e)
            }
        }
    }

    override fun pause(repoId: String) {
        scope.launch {
            val task = tasks[repoId] ?: return@launch
            task.handles.values.forEach { engineOf(task).pause(it) }
            update(repoId) { it.copy(status = DownloadStatus.Paused, bytesPerSecond = 0) }
            persist(task)
        }
    }

    override fun resume(repoId: String) {
        scope.launch { resumeTask(repoId) }
    }

    override fun cancel(repoId: String) {
        scope.launch {
            val task = tasks.remove(repoId)
            if (task != null) {
                task.handles.values.forEach { engineOf(task).remove(it) }
                File(stagingDir, task.folderName).deleteRecursively()
            }
            dao.remove(repoId)
            state.update { list -> list.filterNot { it.repoId == repoId } }
        }
    }

    override fun dismiss(repoId: String) {
        scope.launch {
            if (tasks.containsKey(repoId)) return@launch  // only finished entries can be dismissed
            dao.remove(repoId)
            state.update { list -> list.filterNot { it.repoId == repoId } }
        }
    }

    override fun setDnsMode(mode: DnsMode) {
        scope.launch {
            if (mode == aria2.dnsMode) return@launch
            aria2.dnsMode = mode
            // aria2 takes the resolver at startup: restart it now if no aria2 download is running.
            val busy = state.value.any { d -> d.status == DownloadStatus.Downloading && d.engine != DownloadEngine.Fetch }
            if (!busy) aria2.restart()
        }
    }

    override fun setEngine(engine: DownloadEngine) {
        scope.launch { defaultEngine = engine }
    }

    // ------------------------------------------------------------ internals

    private fun engineOf(task: Task): TransferEngine = if (task.engine == DownloadEngine.Fetch) fetch else aria2

    /**
     * Some ROMs switch off network access for newly installed apps ("Allow network access" under
     * App info → Mobile data & Wi-Fi). Every lookup then fails like a DNS error, so check up front
     * and say what's actually wrong.
     */
    @Suppress("DEPRECATION")  // NetworkInfo is the only API that reports a per-app BLOCKED state
    private fun checkNetworkAllowed() {
        val cm = context.getSystemService(ConnectivityManager::class.java) ?: return
        if (cm.activeNetworkInfo?.detailedState == NetworkInfo.DetailedState.BLOCKED) {
            throw DownloadException(
                "Android is blocking NekoChat's internet access. Turn on \"Allow network access\" in " +
                    "NekoChat's Mobile data & Wi-Fi settings, then tap Retry.",
                networkBlocked = true,
            )
        }
    }

    private suspend fun resumeTask(repoId: String) {
        val task = tasks[repoId] ?: return
        try {
            checkNetworkAllowed()
            val engine = engineOf(task)
            // Handles the engine no longer knows (e.g. aria2 restarted) are dropped and re-added:
            // both engines continue from the partial data on disk.
            task.handles.entries.removeAll { !engine.resume(it.value) }
            launch(task)
        } catch (e: Exception) {
            fail(repoId, e)
        }
    }

    private suspend fun launch(task: Task) {
        val engine = engineOf(task)
        val dir = File(stagingDir, task.folderName).apply { mkdirs() }
        for (f in task.files) {
            if (isComplete(dir, f) || task.handles.containsKey(f.path)) continue
            task.handles[f.path] = engine.add(HuggingFaceHub.resolveUrl(task.repoId, f.path), dir, f.path, f.size,
                task.engine.connections)
        }
        update(task.repoId) { it.copy(status = DownloadStatus.Downloading, error = null, networkBlocked = false,
            bytesTotal = task.files.sumOf { f -> f.size }) }
        persist(task)
        DownloadForegroundService.start(context)
        ensurePolling()
    }

    private fun ensurePolling() {
        if (poller?.isActive == true) return
        poller = scope.launch {
            while (state.value.any { it.status == DownloadStatus.Downloading }) {
                for (task in tasks.values.toList()) {
                    if (state.value.firstOrNull { it.repoId == task.repoId }?.status == DownloadStatus.Downloading) poll(task)
                }
                delay(700)
            }
        }
    }

    private suspend fun poll(task: Task) {
        val engine = engineOf(task)
        val dir = File(stagingDir, task.folderName)
        var done = 0L
        var speed = 0L
        var complete = true
        var error: String? = null
        for (f in task.files) {
            val handle = task.handles[f.path]
            if (handle == null) {
                if (isComplete(dir, f)) done += f.size else complete = false
                continue
            }
            val s = engine.status(handle)
            if (s == null) {
                // The engine lost this file (process restarted): add it again; it continues from disk.
                task.handles.remove(f.path)
                task.handles[f.path] = engine.add(HuggingFaceHub.resolveUrl(task.repoId, f.path), dir, f.path, f.size,
                    task.engine.connections)
                complete = false
                continue
            }
            done += s.bytesDone
            speed += s.bytesPerSecond
            when (s.state) {
                FileState.Complete -> Unit
                FileState.Error -> {
                    complete = false
                    error = s.error ?: "Download failed."
                }
                else -> complete = false
            }
        }
        when {
            error != null -> {
                Log.w("NekoChat", "download of ${task.repoId} failed: $error")
                task.handles.values.forEach { engine.pause(it) }
                fail(task.repoId, DownloadException(friendly(error!!)))
            }
            complete && task.files.all { isComplete(dir, it) } -> finish(task)
            else -> update(task.repoId) { it.copy(bytesDone = done, bytesPerSecond = speed) }
        }
    }

    private suspend fun finish(task: Task) {
        task.handles.values.forEach { engineOf(task).remove(it) }
        val from = File(stagingDir, task.folderName)
        from.listFiles()?.filter { it.name.endsWith(".aria2") }?.forEach { it.delete() }
        var target = File(modelsDir, task.folderName)
        if (target.exists()) target = File(modelsDir, uniqueFolderName(task.folderName))
        if (!from.renameTo(target)) {
            fail(task.repoId, DownloadException("Couldn't move the model into place."))
            return
        }
        tasks.remove(task.repoId)
        val total = task.files.sumOf { it.size }
        update(task.repoId) {
            it.copy(status = DownloadStatus.Completed, bytesDone = total, bytesTotal = total, bytesPerSecond = 0, folder = target.path)
        }
        dao.put(entity(task, DownloadStatus.Completed, null).copy(folderName = target.name))
    }

    private suspend fun fail(repoId: String, e: Exception) {
        val msg = (e as? DownloadException)?.message ?: e.message ?: e.toString()
        val blocked = (e as? DownloadException)?.networkBlocked == true
        update(repoId) { it.copy(status = DownloadStatus.Failed, error = msg, bytesPerSecond = 0, networkBlocked = blocked) }
        tasks[repoId]?.let { dao.put(entity(it, DownloadStatus.Failed, msg)) }
    }

    private fun isComplete(dir: File, f: HuggingFaceHub.RepoFile): Boolean {
        val file = File(dir, f.path)
        return file.exists() && file.length() == f.size && !File(dir, f.path + ".aria2").exists()
    }

    private fun onDisk(task: Task): Long {
        val dir = File(stagingDir, task.folderName)
        return task.files.sumOf { f -> if (isComplete(dir, f)) f.size else 0L }
    }

    private fun uniqueFolderName(base: String): String {
        val clean = base.replace(Regex("[^A-Za-z0-9._-]"), "_")
        var name = clean
        var n = 2
        while (File(modelsDir, name).exists() || tasks.values.any { it.folderName == name }) name = "$clean-${n++}"
        return name
    }

    private fun put(d: ModelDownload) = state.update { list -> list.filterNot { it.repoId == d.repoId } + d }

    private fun update(repoId: String, f: (ModelDownload) -> ModelDownload) =
        state.update { list -> list.map { if (it.repoId == repoId) f(it) else it } }

    private suspend fun persist(task: Task) {
        val s = state.value.firstOrNull { it.repoId == task.repoId } ?: return
        dao.put(entity(task, s.status, s.error))
    }

    private fun entity(task: Task, status: DownloadStatus, error: String?) = DownloadEntity(
        repoId = task.repoId,
        folderName = task.folderName,
        status = status.name,
        bytesTotal = task.files.sumOf { it.size },
        files = JSONArray().apply { task.files.forEach { put(JSONObject().put("path", it.path).put("size", it.size)) } }.toString(),
        error = error,
        createdAt = task.createdAt,
        engine = task.engine.name,
    )

    private fun parseFiles(json: String): List<HuggingFaceHub.RepoFile> = runCatching {
        val a = JSONArray(json)
        List(a.length()) { i -> a.getJSONObject(i).let { HuggingFaceHub.RepoFile(it.getString("path"), it.getLong("size")) } }
    }.getOrDefault(emptyList())

    private fun friendly(error: String): String = when {
        error.contains("No space", ignoreCase = true) -> "Not enough storage space on this phone."
        error.contains("resolve", ignoreCase = true) || error.contains("Network", ignoreCase = true) ->
            "Network problem. Check your connection and tap Retry to continue where it stopped."
        else -> "$error Tap Retry to continue where it stopped."
    }

    private companion object {
        val REPO = Regex("^([A-Za-z0-9][A-Za-z0-9._-]*/)?[A-Za-z0-9][A-Za-z0-9._-]*$")
    }
}
