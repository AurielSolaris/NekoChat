# NekoChat — Branding & UI Specification

This document is the authoritative source of truth for NekoChat's visual identity and application UI.

Claude Code MUST consult this document before creating or modifying:

* Branding assets
* Icons
* Mascot artwork
* Wordmarks
* Colors
* Compose UI
* Navigation
* Components
* Screens
* UI animations
* Decorative graphics

Canonical branding assets belong under:

```text
app/branding/
```

The application UI is implemented in the Android/Jetpack Compose codebase.

---

# 1. Brand Identity

**Name:** NekoChat

**Concept:** A local-first AI chat application for Android that runs models directly on the device.

NekoChat should feel:

* Friendly
* Playful
* Soft
* Modern
* Personal
* Technically capable
* Privacy-conscious
* Local-first

The application should feel like a **personal AI companion**, not an enterprise AI dashboard.

The visual identity combines:

> **One Dark + pastel pink + glassmorphism + maneki-neko**

The result should be soft and approachable while retaining a technically sophisticated dark foundation.

---

# 2. Mascot

The primary NekoChat mascot is a **maneki-neko (beckoning/lucky cat)**.

The cat has one raised front paw performing the traditional beckoning gesture.

The raised paw is an important part of the mascot identity.

## Appearance

Primary characteristics:

* Soft pastel pink body
* Rounded, friendly proportions
* Simple facial features
* Friendly expression
* Optional subtle gold details
* Clean silhouette
* Minimal ornamentation

The mascot should remain recognizable at small sizes.

The mascot should NOT look:

* Aggressive
* Sinister
* Corporate
* Overly realistic
* Excessively detailed
* Cyberpunk
* Like a generic AI robot

## Vector First

Canonical mascot artwork MUST be SVG.

```text
app/branding/mascot/
```

Raster images may be generated from the canonical SVG for platform-specific use.

The SVG is the source of truth.

---

# 3. Application Icon

The primary application icon should feature the pastel-pink maneki-neko.

The beckoning paw should remain recognizable whenever practical.

The icon should work without text.

Avoid unnecessarily combining the mascot with generic AI imagery.

Do not add:

* Neural-network diagrams
* Robot heads
* Brain imagery
* Generic AI circuit patterns
* Excessive chat-bubble symbolism

The mascot itself already communicates:

> "Come chat with me."

---

# 4. Visual Foundation

NekoChat uses **One Dark as its structural visual foundation**.

This does NOT mean copying the exact appearance of an IDE.

Instead, use the One Dark philosophy:

* Deep dark background
* Clear surface hierarchy
* Muted neutrals
* Strong but restrained accent colors
* High readability
* Minimal visual noise

The One Dark foundation is then softened using:

* Pastel pink
* Translucent surfaces
* Blur
* Subtle gradients
* Rounded corners
* Soft highlights

The result should feel like:

> **One Dark, but cozy.**

---

# 5. Color System

Pastel pink is the primary NekoChat accent.

The UI should remain predominantly dark.

Suggested base palette:

```text
Background:
#282C34

Background Deep:
#21252B

Surface:
#2C313C

Surface Elevated:
#323842

Border:
#3A404C

Primary Pink:
#F4A7B9

Light Pink:
#F8C7D2

Soft Pink:
#EFA0B5

Pink Highlight:
#FFD6DF

Cream:
#FFF7F0

Gold:
#D9A441

Text Primary:
#F0F0F0

Text Secondary:
#ABB2BF

Text Muted:
#7F848E

Success:
#98C379

Warning:
#E5C07B

Error:
#E06C75

Info:
#61AFEF
```

These values are starting points and may be adjusted for accessibility and visual consistency.

Pink should be the dominant accent.

Gold should remain an accent and should not compete with pink.

---

# 6. Color Usage

## Background

Use deep One Dark neutrals for the application background.

Do not use pure black unless specifically required.

Avoid large areas of saturated color.

## Primary Accent

Pastel pink is used for:

* Primary actions
* Selected navigation
* Important controls
* Focus states
* Active model indicators
* User identity elements
* Important highlights

Pink should NOT be applied to every component.

It is an accent, not the entire UI.

## Text

Primary text should use a soft near-white rather than pure white.

Secondary information should use muted One Dark gray.

Muted text should remain readable but visually subordinate.

---

# 7. Glassmorphism

Glassmorphism is a core part of the NekoChat UI.

However:

> Glassmorphism must enhance hierarchy, not replace it.

Glass surfaces should generally use:

* Translucent dark backgrounds
* Background blur where practical
* Soft borders
* Low-opacity highlights
* Large rounded corners
* Subtle shadows
* Very restrained gradients

Example conceptual surface:

```text
background:
    dark neutral
    + low-opacity pink tint

border:
    soft translucent highlight

blur:
    moderate

corner radius:
    generous

shadow:
    subtle
```

Avoid:

* Excessive transparency
* Extremely strong blur
* Bright neon borders
* Heavy drop shadows
* Excessive gradients
* Glass effects on every tiny element

Not every component needs to be glass.

---

# 8. Surface Hierarchy

The UI should have clear layers.

Conceptually:

```text
┌───────────────────────────────────────┐
│           Application Background      │
│                                       │
│   ┌───────────────────────────────┐   │
│   │       Glass Surface           │   │
│   │                               │   │
│   │   ┌───────────────────────┐   │   │
│   │   │ Elevated Glass        │   │   │
│   │   └───────────────────────┘   │   │
│   │                               │   │
│   └───────────────────────────────┘   │
│                                       │
└───────────────────────────────────────┘
```

Use elevation primarily through:

1. Opacity
2. Blur
3. Brightness
4. Border contrast
5. Subtle shadows

Do not rely solely on color differences.

---

# 9. Shape Language

NekoChat uses rounded geometry.

Preferred:

* Rounded rectangles
* Soft cards
* Circular buttons
* Rounded input fields
* Smooth icon containers

Avoid sharp rectangular UI unless required for technical information.

Suggested corner-radius hierarchy:

```text
Small:
8dp

Medium:
12dp

Large:
16dp

Extra Large:
24dp

Pills:
999dp
```

Use larger radii for major glass surfaces.

---

# 10. Typography

Typography should be clean and highly readable.

The UI should prioritize readability over decorative typography.

Use the platform's standard Android typography unless a specific branded typeface is introduced later.

Hierarchy:

```text
Screen title
    ↓
Section title
    ↓
Primary content
    ↓
Secondary information
    ↓
Metadata / hints
```

Do not use excessive font weights or sizes.

Code/model metadata may use a monospace font where technically appropriate.

---

# 11. Main Application Experience

NekoChat's UI should remain intentionally simple.

The initial setup should primarily ask for:

```text
Username
Model
```

The model selector should scan the application's designated models directory.

Conceptually:

```text
NekoChat
│
├── Username
│
├── Model
│
└── Chat
```

Do not introduce unnecessary configuration screens.

Advanced functionality can be added later without complicating the initial experience.

---

# 12. Model Picker

The model picker should feel like a native part of NekoChat.

A model entry should communicate useful information such as:

```text
Model Name
Architecture
Format
Size
Status
```

Example:

```text
┌─────────────────────────────────┐
│  🐱 GPT-2                       │
│     GPT-2 · SafeTensors         │
│     548 MB                      │
│                           ✓     │
└─────────────────────────────────┘
```

The actual implementation should use NekoChat's iconography rather than emoji where appropriate.

Future model sources such as Hugging Face should integrate into the same model-selection experience.

---

# 13. Chat UI

The chat screen should be the visual centerpiece of the application.

Keep it clean.

User and assistant messages should have clear visual distinction without becoming giant opaque blocks.

Preferred approach:

```text
              Assistant
        ┌───────────────────┐
        │ Hello!             │
        │ How can I help?    │
        └───────────────────┘

User

                    ┌───────────────────┐
                    │ Run GPT-2 locally │
                    └───────────────────┘
```

Message containers may use subtle glass surfaces.

Avoid excessive bubbles, borders, or decorative elements.

---

# 14. Input Bar

The message input should be a prominent glass component.

Conceptually:

```text
╭────────────────────────────────────────╮
│ Message NekoChat...             ＋  ➤ │
╰────────────────────────────────────────╯
```

Characteristics:

* Large rounded shape
* Glass background
* Subtle pink focus state
* Clear send button
* Comfortable touch target
* Minimal visual clutter

The send button may use the primary pastel pink.

---

# 15. Navigation

Navigation should remain minimal.

Avoid building a large sidebar full of options simply because the application supports them.

Potential destinations may include:

```text
Chat
Models
Settings
```

Additional destinations should only exist when they provide meaningful functionality.

---

# 16. Icons

Icons should use a consistent visual language.

Prefer:

* Simple line icons
* Rounded geometry
* Consistent stroke width
* Minimal detail

Brand mascot artwork and functional UI icons should remain visually distinct.

Do not use the mascot as a replacement for every UI icon.

---

# 17. Animation

Animations should feel soft and responsive.

Preferred:

* Short fades
* Gentle scale transitions
* Subtle glass transitions
* Smooth message appearance
* Small mascot interactions

Avoid:

* Excessive bouncing
* Long transitions
* Constant motion
* Distracting particle effects
* Animations that interfere with typing or scrolling

Performance is a priority.

Animations must not meaningfully interfere with inference or rendering performance.

---

# 18. Performance

NekoChat is an on-device inference application.

UI design MUST respect the performance requirements of the inference engine.

The application separates:

```text
UI / Render Thread
        │
        │
        └──── Inference Thread
```

UI effects must not introduce unnecessary CPU/GPU workload.

In particular:

* Avoid excessive blur layers.
* Avoid unnecessary recomposition.
* Avoid continuously animated backgrounds.
* Avoid expensive effects over large surfaces.
* Avoid unnecessary bitmap generation.
* Prefer vector assets.
* Reuse rendered resources where possible.

Glassmorphism should be implemented efficiently.

A beautiful interface that causes inference performance to collapse is a bug, not a feature.

---

# 19. Accessibility

Visual style MUST NOT compromise usability.

Ensure:

* Sufficient text contrast
* Clear focus states
* Adequate touch targets
* Readable text sizes
* Important information is not communicated by color alone
* Pink accent states remain distinguishable
* Glass transparency does not reduce text readability

When a glass effect conflicts with readability, readability wins.

---

# 20. Branding Asset Structure

Canonical branding assets should live under:

```text
app/branding/
```

Recommended structure:

```text
app/branding/
├── BRANDING+UI.md
│
├── mascot/
│   ├── neko-mascot.svg
│   ├── neko-mascot-light.svg
│   └── neko-mascot-dark.svg
│
├── icons/
│   ├── app-icon.svg
│   └── ...
│
├── wordmark/
│   ├── nekochat-wordmark.svg
│   └── ...
│
└── palette/
    └── colors.md
```

The structure may evolve as the branding system grows.

---

# 21. Source Asset Rules

Canonical assets should be:

* SVG where possible
* Editable
* Resolution independent
* Free of unnecessary embedded raster images
* Free of external dependencies
* Appropriately optimized
* Reusable across platforms

Platform-specific derivatives should be generated from canonical source assets.

Do not maintain unrelated copies of the same artwork.

---

# 22. Design Rules for Claude Code

When asked to create or modify NekoChat UI:

1. Read this document first.
2. Inspect the existing UI before making changes.
3. Reuse existing components and design tokens.
4. Preserve the One Dark foundation.
5. Use pastel pink as the primary accent.
6. Use glassmorphism selectively.
7. Preserve the maneki-neko mascot identity.
8. Prefer SVG for canonical branding artwork.
9. Optimize for Android performance.
10. Do not introduce unrelated visual styles.
11. Do not replace existing branding without explicit instruction.
12. Update this document when establishing a permanent new branding rule.

When uncertain between a flashy design and a simpler design:

> **Choose the simpler design.**

NekoChat should feel polished, not overloaded.

---

# 23. Core Design Principle

NekoChat's visual identity can be summarized as:

> **A soft pastel-pink maneki-neko living inside a cozy One Dark glass interface.**

The UI should feel like a technically sophisticated local AI application that happens to have a cute cat mascot—not a children's application pretending to be an AI runtime.

**Cute, but competent.**

**Soft, but technical.**

**Glass, but performant.**

**Pink, but not neon.**

---

# 24. Implementation Notes (established rules)

These record decisions taken while implementing this spec; keep them in sync with the code.

* **Tokens.** Colors, radii and glass levels live in `app/ui/com/nekochat/ui/theme/Glass.kt`
  (`object Neko`, `object NekoRadius`, `Modifier.glass(level)`). Screens never hard-code hex values.
* **Blur.** Glass is rendered *without* a live backdrop blur. The backdrop is a static One Dark
  gradient with two very low-opacity glows (pink, info-blue), drawn once via `drawWithCache`; panels
  are translucent Surface / Surface Elevated fills with a faint pink tint and a hairline highlight
  border. This reads as frosted glass at zero per-frame cost (§18 wins over §7 "blur where practical").
* **Glass levels.** `GlassLevel.Base` (cards, top bars, input bar) and `GlassLevel.Elevated`
  (selected rows, dialogs, bubbles that need emphasis). Nested glass never exceeds two levels.
* **On-pink content.** Text/icons on Primary Pink use Background `#282C34` (≈ 8:1), never white.
* **Selection is never color-only.** Selected rows/chips also show a check mark or a filled indicator.
* **Touch targets.** Interactive elements are at least 48dp in their smaller dimension (chips 44dp min).
* **Mascot usage.** The mascot appears in the setup header, empty states and the model-loading card
  (`res/drawable/neko_mascot.xml`, derived from `mascot/neko-mascot.svg`). Functional icons are
  Material rounded line/filled icons, never the mascot.
* **Derivatives.** `res/drawable/neko_mascot.xml` and `res/drawable/ic_launcher_foreground.xml` are
  path-for-path conversions of the canonical SVGs (the SVGs use paths only for this reason).
  When the SVG changes, regenerate both.
* **Motion.** Screen changes: 220 ms crossfade. New messages: 180 ms fade/placement via `animateItem`.
  The only looping animation is the typing indicator, shown only while waiting for the first token.
