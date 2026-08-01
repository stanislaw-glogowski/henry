"""Command-line audio capture and full-duplex diagnostics."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from threading import Barrier

import numpy as np

from henry_resources import LocalStore
from henry_speech.audio import (
    AudioBuffer,
    AudioDriver,
    AudioFormat,
    AudioFrame,
    AudioInput,
    AudioPlaybackOutcome,
    AudioSettings,
    get_audio_driver,
)

_DEFAULT_SECONDS = 5.0
_DUPLEX_SAMPLE_RATE = 48_000


def _capture_for(capture: AudioInput, seconds: float) -> AudioFrame:
    first_frame = capture.read()
    target_samples = max(1, round(seconds * first_frame.format.sample_rate))
    buffer = AudioBuffer()
    buffer.append(first_frame)
    captured_samples = first_frame.samples_count

    while captured_samples < target_samples:
        frame = capture.read()
        buffer.append(frame)
        captured_samples += frame.samples_count

    buffered = buffer.build()
    if buffered is None:
        raise RuntimeError("Audio input produced no frames")
    return buffered.format.build_frame(
        np.ascontiguousarray(buffered.samples[:target_samples], dtype=np.float32)
    )


def _levels(frame: AudioFrame) -> tuple[float, float]:
    peak = float(np.max(np.abs(frame.samples), initial=0.0))
    rms = float(np.sqrt(np.mean(np.square(frame.samples), dtype=np.float64)))
    return rms, peak


def _test_signal(seconds: float) -> AudioFrame:
    sample_count = max(1, round(seconds * _DUPLEX_SAMPLE_RATE))
    time = np.arange(sample_count, dtype=np.float64) / _DUPLEX_SAMPLE_RATE
    frequencies = (220, 330, 440, 660, 880, 1_320, 1_760, 2_640, 3_200)
    signal = sum(
        np.sin(2 * np.pi * frequency * time + index * 0.73)
        for index, frequency in enumerate(frequencies)
    )
    signal *= 0.55 + 0.45 * np.sin(2 * np.pi * 2.3 * time) ** 2

    fade_samples = min(round(0.02 * _DUPLEX_SAMPLE_RATE), sample_count // 2)
    if fade_samples:
        fade = np.sin(np.linspace(0, np.pi / 2, fade_samples)) ** 2
        signal[:fade_samples] *= fade
        signal[-fade_samples:] *= fade[::-1]

    peak = float(np.max(np.abs(signal), initial=1.0))
    samples = np.asarray(signal * (0.18 / peak), dtype=np.float32)
    return AudioFrame(
        format=AudioFormat(sample_rate=_DUPLEX_SAMPLE_RATE, channels=1),
        samples=samples,
    )


def record_and_play(driver: AudioDriver, seconds: float) -> AudioFrame:
    """Capture raw input for a fixed duration and play the captured frame once."""
    if seconds <= 0:
        raise ValueError(f"Recording duration must be positive; got {seconds}")

    with ExitStack() as resources:
        resources.enter_context(driver)
        capture = resources.enter_context(driver.input)
        playback = resources.enter_context(driver.output)

        print(f"Recording raw microphone audio for {seconds:g} seconds…")
        recorded = _capture_for(capture, seconds)
        rms, peak = _levels(recorded)
        print(
            "Captured "
            f"{recorded.samples_count / recorded.format.sample_rate:.3f} seconds: "
            f"sample_rate={recorded.format.sample_rate}, "
            f"channels={recorded.format.channels}, rms={rms:.6f}, peak={peak:.6f}"
        )
        print("Playing captured audio…")
        if playback.write(recorded) is AudioPlaybackOutcome.INTERRUPTED:
            raise RuntimeError("Playback was interrupted before completion")
        print("Playback completed.")
        return recorded


def measure_full_duplex(driver: AudioDriver, seconds: float) -> AudioFrame:
    """Play a test signal while capturing the residual microphone signal."""
    if seconds <= 0:
        raise ValueError(f"Recording duration must be positive; got {seconds}")

    playback_signal = _test_signal(seconds)
    with ExitStack() as resources:
        resources.enter_context(driver)
        capture = resources.enter_context(driver.input)
        playback = resources.enter_context(driver.output)

        # Prime device capture and the streaming resampler before aligning the
        # playback and measurement threads.
        capture.read()
        start = Barrier(2)

        def play() -> AudioPlaybackOutcome:
            start.wait()
            return playback.write(playback_signal)

        print(
            f"Playing a test signal while recording for {seconds:g} seconds… "
            "Remain silent."
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            playback_future = executor.submit(play)
            start.wait()
            residual = _capture_for(capture, seconds)
            outcome = playback_future.result()

        if outcome is AudioPlaybackOutcome.INTERRUPTED:
            raise RuntimeError("Test-signal playback was interrupted")

        source_rms, _ = _levels(playback_signal)
        residual_rms, residual_peak = _levels(residual)
        relative_db = 20 * np.log10(max(residual_rms, 1e-12) / source_rms)
        print(
            "Captured full-duplex residual: "
            f"sample_rate={residual.format.sample_rate}, "
            f"channels={residual.format.channels}, rms={residual_rms:.6f}, "
            f"peak={residual_peak:.6f}, "
            f"relative_to_playback={relative_db:.1f} dB"
        )
        print("Playing the residual microphone capture…")
        if playback.write(residual) is AudioPlaybackOutcome.INTERRUPTED:
            raise RuntimeError("Residual playback was interrupted")
        print("Full-duplex diagnostic completed.")
        return residual


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record and play raw microphone audio without VAD or wake word."
    )
    parser.add_argument("--seconds", type=float, default=_DEFAULT_SECONDS)
    parser.add_argument("--driver", choices=("avfaudio", "pyaudio"))
    parser.add_argument(
        "--duplex",
        action="store_true",
        help="play a test signal while measuring microphone echo",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    configured = LocalStore().load_settings().speech.audio
    settings = AudioSettings(driver=args.driver or configured.driver)
    driver = get_audio_driver(settings)
    if args.duplex:
        measure_full_duplex(driver, args.seconds)
    else:
        record_and_play(driver, args.seconds)
