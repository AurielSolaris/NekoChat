package com.nekochat.download

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import com.nekochat.engine.formatBytes
import kotlinx.coroutines.Job
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

/** Keeps the process alive while models download, with a progress notification. Stops itself when idle. */
class DownloadForegroundService : Service() {
    private val scope = MainScope()
    private var watcher: Job? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val service = Downloads.get(this)
        ServiceCompat.startForeground(
            this, NOTIFICATION_ID, build(service.downloads.value.filter { it.active }),
            if (Build.VERSION.SDK_INT >= 29) ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC else 0,
        )
        if (watcher == null) {
            watcher = scope.launch {
                service.downloads.collect { list ->
                    val active = list.filter { it.active }
                    if (active.isEmpty()) {
                        ServiceCompat.stopForeground(this@DownloadForegroundService, ServiceCompat.STOP_FOREGROUND_REMOVE)
                        stopSelf()
                    } else {
                        getSystemService(NotificationManager::class.java).notify(NOTIFICATION_ID, build(active))
                    }
                }
            }
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        scope.cancel()
        super.onDestroy()
    }

    private fun build(active: List<ModelDownload>): android.app.Notification {
        if (Build.VERSION.SDK_INT >= 26) {
            getSystemService(NotificationManager::class.java).createNotificationChannel(
                NotificationChannel(CHANNEL, "Model downloads", NotificationManager.IMPORTANCE_LOW),
            )
        }
        val done = active.sumOf { it.bytesDone }
        val total = active.sumOf { it.bytesTotal }
        val speed = active.sumOf { it.bytesPerSecond }
        val title = when (active.size) {
            0 -> "Preparing download"
            1 -> "Downloading ${active[0].name}"
            else -> "Downloading ${active.size} models"
        }
        val open = packageManager.getLaunchIntentForPackage(packageName)?.let {
            PendingIntent.getActivity(this, 0, it, PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        }
        return NotificationCompat.Builder(this, CHANNEL)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentTitle(title)
            .setContentText(if (total > 0) "${formatBytes(done)} of ${formatBytes(total)} · ${formatBytes(speed)}/s" else "Checking files…")
            .setProgress(1000, if (total > 0) (done * 1000 / total).toInt() else 0, total == 0L)
            .setOnlyAlertOnce(true)
            .setOngoing(true)
            .setContentIntent(open)
            .build()
    }

    companion object {
        private const val CHANNEL = "downloads"
        private const val NOTIFICATION_ID = 42

        fun start(context: Context) {
            // Started from a user action while NekoChat is visible; ignore the rare background refusal.
            runCatching { ContextCompat.startForegroundService(context, Intent(context, DownloadForegroundService::class.java)) }
        }
    }
}
