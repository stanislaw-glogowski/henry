import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Build and correctly tag Henry's bundled Apple Silicon audio helper."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:
        root = Path(self.root)
        package = root / "native" / "macos" / "henry-audio"
        executable = (
            root
            / "src"
            / "henry_speech"
            / "audio"
            / "adapters"
            / "avfaudio"
            / "bin"
            / "henry-audio"
        )
        sources = [package / "Package.swift", *package.glob("Sources/**/*.swift")]

        if not executable.exists() or any(
            source.stat().st_mtime > executable.stat().st_mtime for source in sources
        ):
            subprocess.run(
                [str(root / "scripts" / "build-native-audio.sh")],
                cwd=root,
                check=True,
            )

        if self.target_name == "wheel":
            build_data["tag"] = "py3-none-macosx_14_0_arm64"
