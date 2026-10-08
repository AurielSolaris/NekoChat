#!/bin/bash
# SPDX-FileCopyrightText: 2026 AurielSolaris
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Builds aria2c for arm64-v8a from source and drops it into app/src/main/jniLibs as libaria2c.so, which is
# where Aria2Engine expects to find it (only files in the native library directory may be executed on
# Android 10+). It also copies the license of every library linked into it to app/src/main/assets/licenses.
#
# This exists so the APK's only prebuilt binary can be reproduced: F-Droid (and anyone else) can delete the
# committed libaria2c.so and run this instead. NekoChat builds and runs without it; downloads then use Fetch.
#
# aria2 is linked statically against OpenSSL, zlib, expat, c-ares and libssh2, the same set as aria2's own
# Dockerfile.android, but built for the app's minSdk instead of API 33 and with OpenSSL 3.5 LTS instead of
# the end-of-life 1.1.1. Every source is pinned by version and SHA-256.
#
# Needs a Linux host (WSL is fine) with curl, tar, xz, make, perl and pkg-config.
# Everything else, the NDK included, is downloaded into $WORK.
#
# Usage:  tools/build-aria2.sh

set -euo pipefail

WORK="${NEKO_ARIA2_WORK:-$HOME/nekochat-aria2}"
# Same NDK as app/build.gradle.kts (ndkVersion 28.2.13676358 is r28c).
NDK_VERSION="${NEKO_NDK_VERSION:-r28c}"
# Must not exceed the app's minSdk.
ANDROID_API=21
ABI=arm64-v8a
HOST=aarch64-linux-android

ARIA2_VERSION=1.37.0
OPENSSL_VERSION=3.5.4
ZLIB_VERSION=1.3.1
EXPAT_VERSION=2.7.3
CARES_VERSION=1.34.5
LIBSSH2_VERSION=1.11.1

# name|url|sha256
SOURCES=(
    "aria2-$ARIA2_VERSION.tar.xz|https://github.com/aria2/aria2/releases/download/release-$ARIA2_VERSION/aria2-$ARIA2_VERSION.tar.xz|60a420ad7085eb616cb6e2bdf0a7206d68ff3d37fb5a956dc44242eb2f79b66b"
    "openssl-$OPENSSL_VERSION.tar.gz|https://github.com/openssl/openssl/releases/download/openssl-$OPENSSL_VERSION/openssl-$OPENSSL_VERSION.tar.gz|967311f84955316969bdb1d8d4b983718ef42338639c621ec4c34fddef355e99"
    "zlib-$ZLIB_VERSION.tar.gz|https://github.com/madler/zlib/releases/download/v$ZLIB_VERSION/zlib-$ZLIB_VERSION.tar.gz|9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23"
    "expat-$EXPAT_VERSION.tar.xz|https://github.com/libexpat/libexpat/releases/download/R_${EXPAT_VERSION//./_}/expat-$EXPAT_VERSION.tar.xz|71df8f40706a7bb0a80a5367079ea75d91da4f8c65c58ec59bcdfbf7decdab9f"
    "c-ares-$CARES_VERSION.tar.gz|https://github.com/c-ares/c-ares/releases/download/v$CARES_VERSION/c-ares-$CARES_VERSION.tar.gz|7d935790e9af081c25c495fd13c2cfcda4792983418e96358ef6e7320ee06346"
    "libssh2-$LIBSSH2_VERSION.tar.xz|https://libssh2.org/download/libssh2-$LIBSSH2_VERSION.tar.xz|9954cb54c4f548198a7cbebad248bdc87dd64bd26185708a294b2b50771e3769"
)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JNI_LIBS="$REPO_ROOT/app/src/main/jniLibs/$ABI"
LICENSES="$REPO_ROOT/app/src/main/assets/licenses"

NDK="$WORK/android-ndk-$NDK_VERSION"
TOOLCHAIN="$NDK/toolchains/llvm/prebuilt/linux-x86_64"
SRC="$WORK/src"
PREFIX="$WORK/prefix"
BUILD="$(uname -m)-pc-linux-gnu"

step() { printf '\n=== %s ===\n' "$1"; }

step "fetching sources into $WORK"
mkdir -p "$WORK/dl" "$SRC" "$PREFIX"

if [ ! -d "$NDK" ]; then
    curl -fL -o "$WORK/ndk.zip" "https://dl.google.com/android/repository/android-ndk-${NDK_VERSION}-linux.zip"
    unzip -q -d "$WORK" "$WORK/ndk.zip"
    rm -f "$WORK/ndk.zip"
fi

for entry in "${SOURCES[@]}"; do
    IFS='|' read -r name url sha <<<"$entry"
    archive="$WORK/dl/$name"
    [ -f "$archive" ] || curl -fL -o "$archive" "$url"
    echo "$sha  $archive" | sha256sum -c --quiet - || { echo "checksum mismatch: $name" >&2; exit 1; }
    dir="$SRC/${name%.tar.*}"
    [ -d "$dir" ] || tar -xf "$archive" -C "$SRC"
done

# Deterministic output: no build id, no debug info, no build paths in the binary.
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1700000000}"
export AR="$TOOLCHAIN/bin/llvm-ar" RANLIB="$TOOLCHAIN/bin/llvm-ranlib" STRIP="$TOOLCHAIN/bin/llvm-strip"
export NM="$TOOLCHAIN/bin/llvm-nm"
export CC="$TOOLCHAIN/bin/$HOST$ANDROID_API-clang" CXX="$TOOLCHAIN/bin/$HOST$ANDROID_API-clang++"
export CFLAGS="-Os -fPIE -ffile-prefix-map=$WORK=."
export CXXFLAGS="$CFLAGS"
export CPPFLAGS="-I$PREFIX/include"
export LDFLAGS="-L$PREFIX/lib -Wl,--build-id=none"
export PKG_CONFIG_LIBDIR="$PREFIX/lib/pkgconfig"
export PATH="$TOOLCHAIN/bin:$PATH"
JOBS="$(nproc)"

autotools() {  # dir, extra configure args...
    local dir="$1"; shift
    (
        cd "$SRC/$dir"
        ./configure --host="$HOST" --build="$BUILD" --prefix="$PREFIX" --disable-shared --enable-static "$@"
        make -j"$JOBS"
        make install
    )
}

step "OpenSSL $OPENSSL_VERSION"
if [ ! -f "$PREFIX/lib/libssl.a" ]; then
    (
        cd "$SRC/openssl-$OPENSSL_VERSION"
        ANDROID_NDK_ROOT="$NDK" ./Configure android-arm64 -D__ANDROID_API__="$ANDROID_API" \
            no-shared no-tests no-docs no-module --prefix="$PREFIX" --libdir=lib
        make -j"$JOBS"
        make install_sw
    )
fi

step "zlib $ZLIB_VERSION"
if [ ! -f "$PREFIX/lib/libz.a" ]; then
    (
        cd "$SRC/zlib-$ZLIB_VERSION"
        ./configure --prefix="$PREFIX" --static
        make -j"$JOBS"
        make install
    )
fi

step "expat $EXPAT_VERSION"
[ -f "$PREFIX/lib/libexpat.a" ] ||
    autotools "expat-$EXPAT_VERSION" --without-docbook --without-examples --without-tests --without-xmlwf

step "c-ares $CARES_VERSION"
[ -f "$PREFIX/lib/libcares.a" ] || autotools "c-ares-$CARES_VERSION" --disable-tests

step "libssh2 $LIBSSH2_VERSION"
[ -f "$PREFIX/lib/libssh2.a" ] ||
    autotools "libssh2-$LIBSSH2_VERSION" --with-crypto=openssl --with-libssl-prefix="$PREFIX" \
        --disable-examples-build --disable-tests --disable-docker-tests

step "aria2 $ARIA2_VERSION"
(
    cd "$SRC/aria2-$ARIA2_VERSION"
    [ -f Makefile ] && make distclean >/dev/null || true
    # The release tarball ships a generated configure, so no autoreconf is needed.
    ./configure --host="$HOST" --build="$BUILD" \
        --disable-nls --without-gnutls --with-openssl --without-sqlite3 --without-libxml2 \
        --with-libexpat --with-libcares --with-libz --with-libssh2 \
        LDFLAGS="$LDFLAGS -fPIE -pie -static-libstdc++"
    make -j"$JOBS"
)

step "stripping and copying into the app"
mkdir -p "$JNI_LIBS"
cp "$SRC/aria2-$ARIA2_VERSION/src/aria2c" "$JNI_LIBS/libaria2c.so"
"$STRIP" --strip-all "$JNI_LIBS/libaria2c.so"
ls -la "$JNI_LIBS/libaria2c.so"
sha256sum "$JNI_LIBS/libaria2c.so"

step "copying licenses"
# Shown on the About page; every statically linked library requires its notice to travel with the binary.
cp "$SRC/aria2-$ARIA2_VERSION/COPYING" "$LICENSES/aria2-COPYING.txt"
cp "$SRC/openssl-$OPENSSL_VERSION/LICENSE.txt" "$LICENSES/openssl-LICENSE.txt"
cp "$SRC/zlib-$ZLIB_VERSION/LICENSE" "$LICENSES/zlib-LICENSE.txt"
cp "$SRC/expat-$EXPAT_VERSION/COPYING" "$LICENSES/expat-COPYING.txt"
cp "$SRC/c-ares-$CARES_VERSION/LICENSE.md" "$LICENSES/c-ares-LICENSE.txt"
cp "$SRC/libssh2-$LIBSSH2_VERSION/COPYING" "$LICENSES/libssh2-COPYING.txt"

step "done"
