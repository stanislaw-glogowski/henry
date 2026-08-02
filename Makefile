NATIVE_AUDIO := src/henry_speech/audio/adapters/avfaudio/bin/henry-audio
NATIVE_AUDIO_INPUTS := \
	native/macos/henry-audio/Package.swift \
	$(shell find native/macos/henry-audio/Sources -type f -name '*.swift') \
	$(shell find native/macos/henry-audio/Sources -type d)

.PHONY: init run

run: $(NATIVE_AUDIO)
	uv run henry-cli

init: $(NATIVE_AUDIO)
	./scripts/init.sh

$(NATIVE_AUDIO): $(NATIVE_AUDIO_INPUTS)
	./scripts/build-native-audio.sh
