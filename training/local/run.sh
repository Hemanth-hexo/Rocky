#!/bin/bash
# Usage: ./run.sh [config.yml]   (default: rocky_config.yml)
# Run from training/local/, with the Rocky venv activated. That config's
# positive/negative clips must already be in place under
# my_custom_model/<model_name>/; run download_data.py first if data/ isn't
# populated yet (shared across all configs, downloaded once).
set -e

cd "$(dirname "$0")"
CONFIG="${1:-rocky_config.yml}"
TRAIN_PY="$(python3 -c 'import openwakeword, os; print(os.path.join(os.path.dirname(openwakeword.__file__), "train.py"))')"

echo "=== [$CONFIG] Augmenting clips + computing features ==="
python3 "$TRAIN_PY" --training_config "$CONFIG" --augment_clips

echo "=== [$CONFIG] Training model ==="
python3 "$TRAIN_PY" --training_config "$CONFIG" --train_model

echo "=== Done. ==="
