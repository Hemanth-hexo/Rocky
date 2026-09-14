#!/bin/bash
# Usage: ./deploy_model.sh rocky   (or: ./deploy_model.sh stop)
# Copies a trained model into place for voice/wake_word.py.
# The external-data filename ("<model>.onnx.data") is baked into the .onnx
# graph at export time and does NOT follow a rename of the main file — it
# must keep that exact name next to whatever the .onnx file itself is called.
set -e
cd "$(dirname "$0")"
MODEL="${1:?usage: deploy_model.sh <model_name>, e.g. rocky or stop}"
DEST_ONNX="../../voice/${MODEL}_wakeword.onnx"

cp "my_custom_model/${MODEL}.onnx" "$DEST_ONNX"
cp "my_custom_model/${MODEL}.onnx.data" "../../voice/${MODEL}.onnx.data"
echo "Deployed to $DEST_ONNX (+ voice/${MODEL}.onnx.data)"
