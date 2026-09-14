package com.nekochat.ui.theme

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawWithCache
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Shape
import androidx.compose.ui.unit.dp
import kotlin.math.max

/** Palette from app/branding/palette/colors.md — "One Dark, but cozy". */
object Neko {
    val Background = Color(0xFF282C34)
    val BackgroundDeep = Color(0xFF21252B)
    val Surface = Color(0xFF2C313C)
    val SurfaceElevated = Color(0xFF323842)
    val Border = Color(0xFF3A404C)

    val Pink = Color(0xFFF4A7B9)
    val PinkLight = Color(0xFFF8C7D2)
    val PinkSoft = Color(0xFFEFA0B5)
    val PinkHighlight = Color(0xFFFFD6DF)
    val Cream = Color(0xFFFFF7F0)
    val Gold = Color(0xFFD9A441)

    val Text = Color(0xFFF0F0F0)
    val TextSecondary = Color(0xFFABB2BF)
    val TextMuted = Color(0xFF7F848E)

    val Success = Color(0xFF98C379)
    val Warning = Color(0xFFE5C07B)
    val Error = Color(0xFFE06C75)
    val Info = Color(0xFF61AFEF)

    /** Content drawn on Primary Pink (≈ 8:1 contrast). */
    val OnPink = Background
}

object NekoRadius {
    val Small = 8.dp
    val Medium = 12.dp
    val Large = 16.dp
    val XLarge = 24.dp
    val Pill = RoundedCornerShape(50)
}

@Composable
fun NekoTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = darkColorScheme(
            primary = Neko.Pink,
            onPrimary = Neko.OnPink,
            secondary = Neko.Info,
            tertiary = Neko.Gold,
            background = Neko.Background,
            onBackground = Neko.Text,
            surface = Neko.Surface,
            onSurface = Neko.Text,
            surfaceVariant = Neko.SurfaceElevated,
            onSurfaceVariant = Neko.TextSecondary,
            outline = Neko.Border,
            error = Neko.Error,
        ),
        content = content,
    )
}

/**
 * Static One Dark backdrop with two faint glows (pink, info-blue). Translucent panels over it read
 * as frosted glass with no per-frame blur; drawWithCache rebuilds the brushes only on resize.
 */
@Composable
fun NekoBackground(content: @Composable BoxScope.() -> Unit) {
    Box(
        Modifier
            .fillMaxSize()
            .drawWithCache {
                val base = Brush.verticalGradient(listOf(Neko.BackgroundDeep, Neko.Background, Neko.BackgroundDeep))
                val r = max(size.width, size.height)
                val pinkCenter = Offset(size.width * 0.95f, size.height * 0.08f)
                val blueCenter = Offset(size.width * 0.0f, size.height * 0.85f)
                val pink = Brush.radialGradient(listOf(Neko.Pink.copy(alpha = 0.10f), Color.Transparent), pinkCenter, r * 0.6f)
                val blue = Brush.radialGradient(listOf(Neko.Info.copy(alpha = 0.06f), Color.Transparent), blueCenter, r * 0.6f)
                onDrawBehind {
                    drawRect(base)
                    drawCircle(pink, r * 0.6f, pinkCenter)
                    drawCircle(blue, r * 0.6f, blueCenter)
                }
            },
        content = content,
    )
}

enum class GlassLevel(val base: Color, val baseAlpha: Float, val tintAlpha: Float, val edgeAlpha: Float) {
    /** Cards, top bars, input bar. */
    Base(Neko.Surface, 0.72f, 0.05f, 0.10f),
    /** Selected rows, emphasised bubbles, content nested in a Base panel. */
    Elevated(Neko.SurfaceElevated, 0.86f, 0.08f, 0.16f),
}

private val EdgeCache = HashMap<Float, Brush>()

private fun edge(alpha: Float): Brush = EdgeCache.getOrPut(alpha) {
    Brush.verticalGradient(listOf(Color.White.copy(alpha = alpha), Neko.Border.copy(alpha = 0.5f)))
}

/**
 * Glass surface: translucent dark neutral, a faint top-lit tint (pink by default) and a soft
 * highlight hairline. [selected] swaps the hairline for a pink focus/selection border.
 */
fun Modifier.glass(
    shape: Shape = RoundedCornerShape(NekoRadius.XLarge),
    level: GlassLevel = GlassLevel.Base,
    tint: Color = Neko.Pink,
    selected: Boolean = false,
): Modifier = this
    .clip(shape)
    .background(level.base.copy(alpha = level.baseAlpha), shape)
    .background(Brush.verticalGradient(listOf(tint.copy(alpha = level.tintAlpha), Color.Transparent)), shape)
    .then(
        if (selected) Modifier.border(1.5.dp, Neko.Pink.copy(alpha = 0.9f), shape)
        else Modifier.border(1.dp, edge(level.edgeAlpha), shape)
    )
