# henry-audio

`henry-audio` is Henry's macOS audio-device process. It owns microphone capture,
48 kHz TTS playback, Apple Voice Processing, and playback interruption in one
`AVAudioEngine` session. Capture extracts channel zero from Voice Processing's
native multichannel buffer without using `AVAudioConverter`; Python resamples
the resulting mono stream to the 16 kHz model format.
It can also lower and restore playback gain while Python confirms whether
microphone activity is a real barge-in.

The helper communicates over versioned framed binary messages on standard input
and standard output. Protocol version 3 carries interleaved Float32 audio,
request identifiers, playback outcomes, gain acknowledgements, diagnostics,
and selected-device information. It is started and stopped by the Python
`AVFAudioDriver`; it is not a system daemon and does not own a separate user
configuration.

Build, test, and package it from the repository root:

```bash
swift format lint --recursive native/macos/henry-audio
swift test --package-path native/macos/henry-audio
./scripts/build-native-audio.sh
```

A protocol change must update the Swift helper, Python adapter, fake helper,
tests, and version number together.
