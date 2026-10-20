# SPDX-FileCopyrightText: 2026 AurielSolaris
# SPDX-License-Identifier: GPL-3.0-or-later

# JNI entry points and the callback interfaces invoked from native code.
-keepclasseswithmembernames class * { native <methods>; }
-keep interface com.nekochat.engine.LoadProgress { *; }
-keep interface com.nekochat.engine.TokenSink { *; }
