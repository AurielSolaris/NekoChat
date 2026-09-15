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
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawWithCache
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Shape
import androidx.compose.ui.unit.dp
import kotlin.math.max

/** One complete set of UI colours. "Accent" is the brand colour of the theme (pink in Rose, blue in Ocean). */
data class Palette(
    val background: Color,
    val backgroundDeep: Color,
    val surface: Color,
    val surfaceElevated: Color,
    val border: Color,
    val accent: Color,
    val accentLight: Color,
    val accentSoft: Color,
    val accentHighlight: Color,
    val secondaryGlow: Color,
    val gold: Color,
    val text: Color,
    val textSecondary: Color,
    val textMuted: Color,
    val success: Color,
    val warning: Color,
    val error: Color,
    val info: Color,
)

enum class NekoTheme(val label: String, val description: String, val palette: Palette) {
    /** Default: app/branding/palette/colors.md — "One Dark, but cozy". */
    Rose(
        "Rose",
        "One Dark with pastel pink. The original NekoChat look.",
        Palette(
            background = Color(0xFF282C34),
            backgroundDeep = Color(0xFF21252B),
            surface = Color(0xFF2C313C),
            surfaceElevated = Color(0xFF323842),
            border = Color(0xFF3A404C),
            accent = Color(0xFFF4A7B9),
            accentLight = Color(0xFFF8C7D2),
            accentSoft = Color(0xFFEFA0B5),
            accentHighlight = Color(0xFFFFD6DF),
            secondaryGlow = Color(0xFF61AFEF),
            gold = Color(0xFFD9A441),
            text = Color(0xFFF0F0F0),
            textSecondary = Color(0xFFABB2BF),
            textMuted = Color(0xFF7F848E),
            success = Color(0xFF98C379),
            warning = Color(0xFFE5C07B),
            error = Color(0xFFE06C75),
            info = Color(0xFF61AFEF),
        ),
    ),

    /**
     * Solarized Dark base tones with its accents replaced by light and dark pastel blues. Status colours stay
     * red / amber / green (softened) so destructive actions still read as destructive.
     */
    Ocean(
        "Ocean",
        "Solarized Dark with light and dark pastel blues.",
        Palette(
            background = Color(0xFF002B36),
            backgroundDeep = Color(0xFF00212B),
            surface = Color(0xFF073642),
            surfaceElevated = Color(0xFF0C4250),
            border = Color(0xFF1B5361),
            accent = Color(0xFF9CCBF0),
            accentLight = Color(0xFFC3E0F7),
            accentSoft = Color(0xFF7FB2E0),
            accentHighlight = Color(0xFFDCEEFB),
            secondaryGlow = Color(0xFF5E8FC7),
            gold = Color(0xFFA9B8EC),
            text = Color(0xFFE6EEF0),
            textSecondary = Color(0xFF9DB0B4),
            textMuted = Color(0xFF6F8B93),
            success = Color(0xFF9CCFA4),
            warning = Color(0xFFE8CB8A),
            error = Color(0xFFF0908F),
            info = Color(0xFF8FB8DE),
        ),
    ),
}

/**
 * Current palette. Every colour reads Compose state, so switching themes recomposes whatever uses it.
 */
object Neko {
    var theme by mutableStateOf(NekoTheme.Rose)
    private val p: Palette get() = theme.palette

    val Background get() = p.background
    val BackgroundDeep get() = p.backgroundDeep
    val Surface get() = p.surface
    val SurfaceElevated get() = p.surfaceElevated
    val Border get() = p.border

    val Accent get() = p.accent
    val AccentLight get() = p.accentLight
    val AccentSoft get() = p.accentSoft
    val AccentHighlight get() = p.accentHighlight
    val Gold get() = p.gold

    val Text get() = p.text
    val TextSecondary get() = p.textSecondary
    val TextMuted get() = p.textMuted

    val Success get() = p.success
    val Warning get() = p.warning
    val Error get() = p.error
    val Info get() = p.info

    /** Content drawn on the accent (≈ 8:1 contrast in both themes). */
    val OnAccent get() = p.background
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
            primary = Neko.Accent,
            onPrimary = Neko.OnAccent,
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
 * Static backdrop with two faint glows (accent, secondary). Translucent panels over it read as frosted glass
 * with no per-frame blur; drawWithCache rebuilds the brushes on resize and on theme change.
 */
@Composable
fun NekoBackground(content: @Composable BoxScope.() -> Unit) {
    val palette = Neko.theme.palette
    key(palette) {
        Box(
            Modifier
                .fillMaxSize()
                .drawWithCache {
                    val base = Brush.verticalGradient(listOf(palette.backgroundDeep, palette.background, palette.backgroundDeep))
                    val r = max(size.width, size.height)
                    val accentCenter = Offset(size.width * 0.95f, size.height * 0.08f)
                    val glowCenter = Offset(size.width * 0.0f, size.height * 0.85f)
                    val accent = Brush.radialGradient(listOf(palette.accent.copy(alpha = 0.10f), Color.Transparent), accentCenter, r * 0.6f)
                    val glow = Brush.radialGradient(listOf(palette.secondaryGlow.copy(alpha = 0.06f), Color.Transparent), glowCenter, r * 0.6f)
                    onDrawBehind {
                        drawRect(base)
                        drawCircle(accent, r * 0.6f, accentCenter)
                        drawCircle(glow, r * 0.6f, glowCenter)
                    }
                },
            content = content,
        )
    }
}

enum class GlassLevel(val baseAlpha: Float, val tintAlpha: Float, val edgeAlpha: Float) {
    /** Cards, top bars, input bar. */
    Base(0.72f, 0.05f, 0.10f),
    /** Selected rows, emphasised bubbles, content nested in a Base panel. */
    Elevated(0.86f, 0.08f, 0.16f);

    val base: Color get() = if (this == Base) Neko.Surface else Neko.SurfaceElevated
}

private val EdgeCache = HashMap<Pair<Float, Color>, Brush>()

private fun edge(alpha: Float): Brush = EdgeCache.getOrPut(alpha to Neko.Border) {
    Brush.verticalGradient(listOf(Color.White.copy(alpha = alpha), Neko.Border.copy(alpha = 0.5f)))
}

/**
 * Glass surface: translucent dark neutral, a faint top-lit tint (the accent by default) and a soft highlight
 * hairline. [selected] swaps the hairline for an accent focus/selection border.
 */
fun Modifier.glass(
    shape: Shape = RoundedCornerShape(NekoRadius.XLarge),
    level: GlassLevel = GlassLevel.Base,
    tint: Color = Neko.Accent,
    selected: Boolean = false,
): Modifier = this
    .clip(shape)
    .background(level.base.copy(alpha = level.baseAlpha), shape)
    .background(Brush.verticalGradient(listOf(tint.copy(alpha = level.tintAlpha), Color.Transparent)), shape)
    .then(
        if (selected) Modifier.border(1.5.dp, Neko.Accent.copy(alpha = 0.9f), shape)
        else Modifier.border(1.dp, edge(level.edgeAlpha), shape)
    )
