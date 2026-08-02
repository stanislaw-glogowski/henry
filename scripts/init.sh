#!/bin/sh
set -eu

REPOSITORY_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

cd "$REPOSITORY_ROOT"
uv sync
uv run python -m tools.install
