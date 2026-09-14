package com.nekochat.engine

import android.content.Context
import android.net.Uri
import android.os.ParcelFileDescriptor
import android.provider.DocumentsContract
import android.provider.DocumentsContract.Document
import android.system.Os
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import org.json.JSONObject
import java.io.Closeable
import java.io.File
import kotlin.math.abs

/** Where a model's files live. */
sealed interface ModelLocation {
    data class Directory(val dir: File) : ModelLocation

    /** A folder inside a Storage Access Framework tree the user picked; files maps name -> document id. */
    data class Tree(val treeUri: Uri, val files: Map<String, String>) : ModelLocation
}

data class LocalModel(
    val id: String,
    val name: String,
    val location: ModelLocation,
    val architecture: String,
    val format: String,
    val sizeBytes: Long,
    /** Files the engine needs (config, tokenizer, checkpoint shards). */
    val loadFiles: List<String>,
    val problem: String?,
) {
    val supported: Boolean get() = problem == null
}

/** A model folder materialised as a real directory path for the native engine. */
interface ModelAccess : Closeable {
    val dir: File
}

/**
 * Where models come from. The user's picked folder is scanned through SAF; the app-private
 * folder is kept for adb pushes and (later) Hugging Face downloads.
 */
interface ModelSource {
    fun list(): List<LocalModel>
}

/** config.json "model_type" values the native engine can run. */
val SUPPORTED_ARCHITECTURES = setOf("gpt2", "qwen3")

/** App-private models folder: adb push target and Hugging Face download destination. */
fun appModelsDir(context: Context): File =
    (context.getExternalFilesDir("models") ?: File(context.filesDir, "models")).apply { mkdirs() }

private val CHECKPOINT_EXTENSIONS = listOf(".safetensors", ".pt", ".pth", ".bin")
private val SUPPORT_FILES = listOf("config.json", "vocab.json", "merges.txt", "tokenizer.json")

/** Shared rules for deciding whether a folder is a model and what it needs. */
private fun describe(
    id: String,
    name: String,
    files: Map<String, Long>,
    location: ModelLocation,
    readConfig: () -> String?,
): LocalModel? {
    val checkpoints = files.keys.filter { n -> CHECKPOINT_EXTENSIONS.any { n.endsWith(it) } }.sorted()
    if (checkpoints.isEmpty() && "config.json" !in files) return null
    val safetensors = checkpoints.filter { it.endsWith(".safetensors") }
    // Mirrors the native loader: every safetensors shard, otherwise the first PyTorch file.
    val used = safetensors.ifEmpty { checkpoints.take(1) }
    val architecture = if ("config.json" in files) {
        runCatching { JSONObject(readConfig() ?: "{}").optString("model_type", "unknown") }.getOrDefault("unknown")
    } else "unknown"
    val hasTokenizer = "tokenizer.json" in files || ("vocab.json" in files && "merges.txt" in files)
    val problem = when {
        "config.json" !in files -> "Missing config.json"
        checkpoints.isEmpty() -> "No .safetensors or .pt weights"
        !hasTokenizer -> "Missing tokenizer files"
        architecture !in SUPPORTED_ARCHITECTURES -> "Architecture \"$architecture\" isn't supported yet"
        else -> null
    }
    return LocalModel(
        id = id,
        name = name,
        location = location,
        architecture = architecture,
        format = if (safetensors.isNotEmpty()) "safetensors" else "pytorch",
        sizeBytes = used.sumOf { files[it] ?: 0L },
        loadFiles = SUPPORT_FILES.filter { it in files } + used,
        problem = problem,
    )
}

class LocalFolderSource(private val root: File) : ModelSource {
    override fun list(): List<LocalModel> {
        if (!root.isDirectory) root.mkdirs()
        val candidates = listOf(root) + (root.listFiles()?.filter { it.isDirectory } ?: emptyList())
        return candidates.mapNotNull { dir ->
            val files = dir.listFiles()?.filter { it.isFile }?.associate { it.name to it.length() } ?: return@mapNotNull null
            describe(dir.absolutePath, if (dir == root) "App folder" else dir.name, files, ModelLocation.Directory(dir)) {
                File(dir, "config.json").readText()
            }
        }
    }
}

class SafFolderSource(context: Context, private val treeUri: Uri) : ModelSource {
    private val resolver = context.contentResolver

    private data class Doc(val id: String, val name: String, val mime: String, val size: Long)

    private fun children(parentId: String): List<Doc> {
        val uri = DocumentsContract.buildChildDocumentsUriUsingTree(treeUri, parentId)
        val cols = arrayOf(Document.COLUMN_DOCUMENT_ID, Document.COLUMN_DISPLAY_NAME, Document.COLUMN_MIME_TYPE, Document.COLUMN_SIZE)
        return resolver.query(uri, cols, null, null, null)?.use { c ->
            buildList {
                while (c.moveToNext()) {
                    add(Doc(c.getString(0), c.getString(1) ?: "", c.getString(2) ?: "", if (c.isNull(3)) 0L else c.getLong(3)))
                }
            }
        } ?: emptyList()
    }

    private fun inspect(folderId: String, name: String, docs: List<Doc>): LocalModel? {
        val files = docs.filter { it.mime != Document.MIME_TYPE_DIR }
        val ids = files.associate { it.name to it.id }
        return describe("$treeUri#$folderId", name, files.associate { it.name to it.size }, ModelLocation.Tree(treeUri, ids)) {
            ids["config.json"]?.let { id ->
                resolver.openInputStream(DocumentsContract.buildDocumentUriUsingTree(treeUri, id))?.use { it.readBytes().decodeToString() }
            }
        }
    }

    override fun list(): List<LocalModel> {
        val rootId = DocumentsContract.getTreeDocumentId(treeUri)
        val top = children(rootId)
        val out = mutableListOf<LocalModel>()
        inspect(rootId, folderName(resolver, treeUri) ?: "Models folder", top)?.let(out::add)
        top.filter { it.mime == Document.MIME_TYPE_DIR }.forEach { d -> inspect(d.id, d.name, children(d.id))?.let(out::add) }
        return out
    }
}

private fun folderName(resolver: android.content.ContentResolver, treeUri: Uri): String? = runCatching {
    val doc = DocumentsContract.buildDocumentUriUsingTree(treeUri, DocumentsContract.getTreeDocumentId(treeUri))
    resolver.query(doc, arrayOf(Document.COLUMN_DISPLAY_NAME), null, null, null)?.use { c ->
        if (c.moveToFirst()) c.getString(0) else null
    }
}.getOrNull()

@OptIn(ExperimentalCoroutinesApi::class)
class ModelRepository(private val context: Context) {
    /** App-private folder (adb push target, Hugging Face downloads). */
    val appModelsDir: File = appModelsDir(context)

    private val tree = MutableStateFlow<Uri?>(null)
    val treeUri: StateFlow<Uri?> = tree.asStateFlow()

    fun setTree(uri: Uri?) {
        tree.value = uri
    }

    fun folderName(uri: Uri): String? = folderName(context.contentResolver, uri)

    private fun sources(uri: Uri?): List<ModelSource> =
        listOfNotNull(uri?.let { SafFolderSource(context, it) }, LocalFolderSource(appModelsDir))

    fun scan(uri: Uri? = tree.value): List<LocalModel> =
        sources(uri).flatMap { runCatching { it.list() }.getOrDefault(emptyList()) }

    /** Rescans every few seconds so models copied into the folder appear on their own. */
    fun observe(intervalMs: Long = 3000): Flow<List<LocalModel>> = tree
        .flatMapLatest { uri ->
            flow {
                while (true) {
                    emit(scan(uri))
                    delay(intervalMs)
                }
            }
        }
        .distinctUntilChanged()
        .flowOn(Dispatchers.IO)

    /**
     * Gives the native engine a directory path for [model]. SAF files are opened here and exposed
     * as "/proc/self/fd/N" symlinks that the engine dups (see openReadOnly in common.h). The native
     * loader copies weights into its own buffers, so the access can be closed right after loading.
     */
    fun open(model: LocalModel): ModelAccess = when (val loc = model.location) {
        is ModelLocation.Directory -> object : ModelAccess {
            override val dir = loc.dir
            override fun close() = Unit
        }
        is ModelLocation.Tree -> {
            val dir = File(context.cacheDir, "model-links/${abs(model.id.hashCode())}").apply {
                deleteRecursively()
                mkdirs()
            }
            val fds = mutableListOf<ParcelFileDescriptor>()
            val access = object : ModelAccess {
                override val dir = dir
                override fun close() {
                    fds.forEach { runCatching { it.close() } }
                    dir.deleteRecursively()
                }
            }
            try {
                for (name in model.loadFiles) {
                    val docId = loc.files[name] ?: error("Missing $name")
                    val uri = DocumentsContract.buildDocumentUriUsingTree(loc.treeUri, docId)
                    val pfd = context.contentResolver.openFileDescriptor(uri, "r") ?: error("Cannot open $name")
                    fds += pfd
                    Os.symlink("/proc/self/fd/${pfd.fd}", File(dir, name).path)
                }
            } catch (e: Throwable) {
                access.close()
                throw e
            }
            access
        }
    }
}

fun formatBytes(bytes: Long): String = when {
    bytes >= 1L shl 30 -> "%.1f GB".format(bytes / (1L shl 30).toDouble())
    bytes >= 1L shl 20 -> "%.0f MB".format(bytes / (1L shl 20).toDouble())
    else -> "%.0f KB".format(bytes / 1024.0)
}
