#!/bin/sh
set -eu

REPOSITORY_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PACKAGE_PATH="$REPOSITORY_ROOT/native/macos/henry-audio"
DESTINATION="$REPOSITORY_ROOT/src/henry_speech/audio/adapters/avfaudio/bin"
CACHE_ROOT="${TMPDIR:-/tmp}/henry-swift-build-cache"

mkdir -p "$CACHE_ROOT/clang" "$CACHE_ROOT/swiftpm"
CLANG_MODULE_CACHE_PATH="$CACHE_ROOT/clang" \
SWIFTPM_MODULECACHE_OVERRIDE="$CACHE_ROOT/swiftpm" \
swift build --package-path "$PACKAGE_PATH" -c release
mkdir -p "$DESTINATION"
cp "$PACKAGE_PATH/.build/release/henry-audio" "$DESTINATION/henry-audio"
