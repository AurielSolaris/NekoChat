package com.nekochat.download

import android.content.Context
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
 * [ModelDownloadService] backed by aria2c (shipped as libaria2c.so): several connections per
 * file and resumable transfers via its .aria2 control files. aria2 runs as a child process
 * controlled over JSON-RPC on 127.0.0.1 with a random secret, and exits with the app.
 *
 * Files are fetched into models/.downloads/<name> (which the model scanner ignores) and the
 * folder is moved into models/ once every file is complete, so half-downloaded models never
 * show up as loadable.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class Aria2DownloadService(private val context: Context) : ModelDownloadService {
    private val modelsDir = appModelsDir(context)
    private val stagingDir = File(modelsDir, ".downloads")
    private val dao: DownloadDao = NekoDatabase.get(context).downloads()

    // Every state change runs on one thread, so jobs and the daemon need no locking.
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO.limitedParallelism(1))
    private val state = MutableStateFlow<List<ModelDownload>>(emptyList())
    override val downloads: StateFlow<List<ModelDownload>> = state.asStateFlow()

    private class Task(val repoId: String, val folderName: String, val files: List<HuggingFaceHub.RepoFile>, val createdAt: Long) {
        val gids = mutableMapOf<String, String>()  // path -> aria2 gid
        var daemonGeneration = -1
    }

    private val tasks = LinkedHashMap<String, Task>()
    private var daemon: Aria2Daemon? = null
    private var dnsMode = DnsMode.System
    private var generation = 0
    private var poller: Job? = null

    init {
        scope.launch {
            for (e in dao.all()) {
                val files = parseFiles(e.files)
                val task = Task(e.repoId, e.folderName, files, e.createdAt)
                val status = runCatching { DownloadStatus.valueOf(e.status) }.getOrDefault(DownloadStatus.Failed)
                    .let { if (it == DownloadStatus.Preparing || it == DownloadStatus.Downloading) DownloadStatus.Paused else it }
                if (status != DownloadStatus.Completed) tasks[e.repoId] = task
                put(ModelDownload(e.repoId, status, bytesDone = if (status == DownloadStatus.Completed) e.bytesTotal else onDisk(task),
                    bytesTotal = e.bytesTotal, error = e.error,
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
            put(ModelDownload(id, DownloadStatus.Preparing))
            try {
                val files = HuggingFaceHub.modelFiles(id)
                val arch = HuggingFaceHub.modelType(id)
                if (arch !in SUPPORTED_ARCHITECTURES) {
                    throw DownloadException("This is a \"$arch\" model. NekoChat runs GPT-2 and Qwen3 models.")
                }
                val task = Task(id, uniqueFolderName(id.substringAfterLast('/')), files, System.currentTimeMillis())
                tasks[id] = task
                put(ModelDownload(id, DownloadStatus.Downloading, bytesTotal = files.sumOf { it.size }))
                launch(task)
            } catch (e: Exception) {
                fail(id, e)
            }
        }
    }

    override fun pause(repoId: String) {
        scope.launch {
            val task = tasks[repoId] ?: return@launch
            daemon?.takeIf { task.daemonGeneration == generation }?.let { d ->
                task.gids.values.forEach { runCatching { d.rpc.call("aria2.forcePause", it) } }
            }
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
                daemon?.takeIf { task.daemonGeneration == generation }?.let { d ->
                    task.gids.values.forEach { gid ->
                        runCatching { d.rpc.call("aria2.forceRemove", gid) }
                        runCatching { d.rpc.call("aria2.removeDownloadResult", gid) }
                    }
                }
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
            if (mode == dnsMode) return@launch
            dnsMode = mode
            // aria2 takes the resolver at startup: restart it now if idle, otherwise on its next start.
            if (state.value.none { it.status == DownloadStatus.Downloading }) {
                daemon?.close()
                daemon = null
            }
        }
    }

    // ------------------------------------------------------------ internals

    private suspend fun resumeTask(repoId: String) {
        val task = tasks[repoId] ?: return
        try {
            val d = daemon()
            if (task.daemonGeneration == generation && task.gids.isNotEmpty()) {
                task.gids.values.forEach { runCatching { d.rpc.call("aria2.unpause", it) } }
                update(repoId) { it.copy(status = DownloadStatus.Downloading, error = null) }
                persist(task)
                ensurePolling()
            } else {
                // New aria2 process (app restarted): re-adding the URLs resumes from the .aria2 control files.
                task.gids.clear()
                launch(task)
            }
        } catch (e: Exception) {
            fail(repoId, e)
        }
    }

    private suspend fun launch(task: Task) {
        val d = daemon()
        val dir = File(stagingDir, task.folderName).apply { mkdirs() }
        for (f in task.files) {
            if (isComplete(dir, f)) continue
            val options = JSONObject().put("dir", dir.path).put("out", f.path)
            task.gids[f.path] = d.rpc.call("aria2.addUri", JSONArray().put(HuggingFaceHub.resolveUrl(task.repoId, f.path)), options) as String
        }
        task.daemonGeneration = generation
        update(task.repoId) { it.copy(status = DownloadStatus.Downloading, error = null, bytesTotal = task.files.sumOf { f -> f.size }) }
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
        val d = daemon ?: return
        val dir = File(stagingDir, task.folderName)
        var done = 0L
        var speed = 0L
        var complete = true
        var error: String? = null
        for (f in task.files) {
            val gid = task.gids[f.path]
            if (gid == null) {
                done += f.size  // was already on disk
                continue
            }
            val s = runCatching {
                d.rpc.call("aria2.tellStatus", gid, JSONArray(listOf("status", "completedLength", "downloadSpeed", "errorMessage"))) as JSONObject
            }.getOrNull()
            if (s == null) {
                error = "The downloader stopped unexpectedly."
                break
            }
            done += s.optString("completedLength").toLongOrNull() ?: 0L
            speed += s.optString("downloadSpeed").toLongOrNull() ?: 0L
            when (s.optString("status")) {
                "complete" -> Unit
                "error" -> {
                    complete = false
                    error = s.optString("errorMessage").ifBlank { "Download failed." }
                }
                else -> complete = false
            }
        }
        when {
            error != null -> {
                task.gids.values.forEach { runCatching { d.rpc.call("aria2.forcePause", it) } }
                fail(task.repoId, DownloadException(friendly(error!!)))
            }
            complete && task.files.all { isComplete(dir, it) } -> finish(task)
            else -> update(task.repoId) { it.copy(bytesDone = done, bytesPerSecond = speed) }
        }
    }

    private suspend fun finish(task: Task) {
        daemon?.let { d -> task.gids.values.forEach { runCatching { d.rpc.call("aria2.removeDownloadResult", it) } } }
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
        update(repoId) { it.copy(status = DownloadStatus.Failed, error = msg, bytesPerSecond = 0) }
        tasks[repoId]?.let { dao.put(entity(it, DownloadStatus.Failed, msg)) }
    }

    private fun daemon(): Aria2Daemon {
        daemon?.takeIf { it.alive }?.let { return it }
        daemon?.close()
        generation++
        return Aria2Daemon.start(context, modelsDir, dnsMode).also { daemon = it }
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
    )

    private fun parseFiles(json: String): List<HuggingFaceHub.RepoFile> = runCatching {
        val a = JSONArray(json)
        List(a.length()) { i -> a.getJSONObject(i).let { HuggingFaceHub.RepoFile(it.getString("path"), it.getLong("size")) } }
    }.getOrDefault(emptyList())

    private fun friendly(aria2Error: String): String = when {
        aria2Error.contains("No space", ignoreCase = true) -> "Not enough storage space on this phone."
        aria2Error.contains("resolve", ignoreCase = true) || aria2Error.contains("Network", ignoreCase = true) ->
            "Network problem. Check your connection and tap Retry to continue where it stopped."
        else -> "$aria2Error Tap Retry to continue where it stopped."
    }

    private companion object {
        val REPO = Regex("^([A-Za-z0-9][A-Za-z0-9._-]*/)?[A-Za-z0-9][A-Za-z0-9._-]*$")
    }
}
