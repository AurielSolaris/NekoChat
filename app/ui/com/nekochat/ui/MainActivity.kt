package com.nekochat.ui

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.animation.Crossfade
import androidx.compose.animation.core.tween
import androidx.lifecycle.viewmodel.compose.viewModel
import com.nekochat.ui.theme.NekoBackground
import com.nekochat.ui.theme.NekoTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            NekoTheme {
                val vm: ChatViewModel = viewModel()
                // Chats (and first-run Setup) are the root: back there leaves the app.
                BackHandler(enabled = !vm.atRoot) { vm.back() }
                NekoBackground {
                    Crossfade(targetState = vm.screen, animationSpec = tween(220), label = "screen") { screen ->
                        when (screen) {
                            Screen.Boot -> Unit
                            Screen.Setup -> SetupScreen(vm)
                            Screen.Chats -> ChatsScreen(vm)
                            Screen.Chat -> ChatScreen(vm)
                            Screen.ChatSettings -> ChatSettingsScreen(vm)
                            Screen.Settings -> SettingsScreen(vm)
                            Screen.About -> AboutScreen(vm)
                            Screen.AddModels -> AddModelsScreen(vm)
                            Screen.Themes -> ThemesScreen(vm)
                        }
                    }
                }
            }
        }
    }
}
