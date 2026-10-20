// SPDX-FileCopyrightText: 2026 AurielSolaris
// SPDX-License-Identifier: GPL-3.0-or-later

package com.nekochat.ui.theme

import androidx.compose.material3.Typography
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import app.auriel.realm.nekochat.R

/** Body and UI text (SIL OFL 1.1, bundled in res/font). Upstream has no Medium cut; Medium falls back to Regular. */
val OpenSans = FontFamily(
    Font(R.font.opensans_regular, FontWeight.Normal),
    Font(R.font.opensans_semibold, FontWeight.SemiBold),
    Font(R.font.opensans_bold, FontWeight.Bold),
)

/**
 * Headings only: screen, dialog and card titles (SIL OFL 1.1, bundled in res/font). Only the Regular cut ships,
 * so SemiBold headings are synthesised.
 */
val Spectral = FontFamily(Font(R.font.spectral_regular))

private fun TextStyle.with(family: FontFamily) = copy(fontFamily = family)

/**
 * Material 3 defaults: display/headline/titleLarge in Spectral, everything else in Open Sans. Text and TextField
 * inherit bodyLarge via LocalTextStyle, so plain Text is Open Sans; headings pass fontFamily = Spectral.
 */
val NekoTypography: Typography = Typography().run {
    Typography(
        displayLarge = displayLarge.with(Spectral),
        displayMedium = displayMedium.with(Spectral),
        displaySmall = displaySmall.with(Spectral),
        headlineLarge = headlineLarge.with(Spectral),
        headlineMedium = headlineMedium.with(Spectral),
        headlineSmall = headlineSmall.with(Spectral),
        titleLarge = titleLarge.with(Spectral),
        titleMedium = titleMedium.with(OpenSans),
        titleSmall = titleSmall.with(OpenSans),
        bodyLarge = bodyLarge.with(OpenSans),
        bodyMedium = bodyMedium.with(OpenSans),
        bodySmall = bodySmall.with(OpenSans),
        labelLarge = labelLarge.with(OpenSans),
        labelMedium = labelMedium.with(OpenSans),
        labelSmall = labelSmall.with(OpenSans),
    )
}
