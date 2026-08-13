#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

python -m rootokin.cli init-lattice --output output/simple_lattice.json
python -m rootokin.cli run output/simple_lattice.json "$ROOT_DIR/examples/simple_run/story.txt" --duration 0.5 --output-dir output/simple_run
