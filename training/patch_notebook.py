"""One-off script: patches openWakeWord's automatic_model_training.ipynb for
Colab's current Python 3.13 runtime and for Kokoro-generated (not piper-generated)
positive clips. Not part of Rocky's runtime — run once, then discard."""

import json

NB_IN = "/tmp/oww_notebook.ipynb"
NB_OUT = "/Users/hemanthsarode/Desktop/PROJECTS/Rocky/training/rocky_wakeword_training.ipynb"

nb = json.load(open(NB_IN))
cells = nb["cells"]


def code_cell(source: str):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


def md_cell(source: str):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


# --- Cell 4: install cell, rewritten ---
# Drops piper-sample-generator/piper-phonemize/webrtcvad (only needed for the
# --generate_clips step we're skipping) and tensorflow-cpu/tensorflow_probability/
# onnx_tf (only needed for the optional tflite export we don't need — confirmed by
# reading train.py: export_model() only calls torch.onnx.export, tflite conversion
# is a separate opt-in path). Installs openwakeword with --no-deps since its
# speexdsp-ns pin (an optional noise-suppression feature) has no Python 3.13 wheel,
# then installs the two deps it actually needs directly. Upgrades datasets instead
# of pinning old pyarrow, since pyarrow<14 has no cp313 wheel either — the actual
# fix for the PyExtensionType error is a newer datasets, not an older pyarrow.
cells[4] = code_cell('''## Environment setup

# install openwakeword (full installation to support training)
import os
if not os.path.exists("openwakeword"):
    !git clone https://github.com/dscripka/openwakeword
!pip install --no-deps -e ./openwakeword
!pip install -q "onnxruntime>=1.10.0,<2" "ai-edge-litert>=2.0.2,<3"

# install other dependencies
!pip install mutagen==1.47.0
!pip install torchinfo==1.8.0
!pip install torchmetrics==1.2.0
!pip install speechbrain==0.5.14
!pip install audiomentations==0.33.0
!pip install torch-audiomentations==0.11.0
!pip install acoustics==0.2.6
!pip install pronouncing==0.2.0
!pip install -q -U datasets
!pip install deep-phonemizer==0.0.19

# Download required models (workaround for Colab)
os.makedirs("./openwakeword/openwakeword/resources/models", exist_ok=True)
!wget -nc https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx -O ./openwakeword/openwakeword/resources/models/embedding_model.onnx
!wget -nc https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.tflite -O ./openwakeword/openwakeword/resources/models/embedding_model.tflite
!wget -nc https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx -O ./openwakeword/openwakeword/resources/models/melspectrogram.onnx
!wget -nc https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.tflite -O ./openwakeword/openwakeword/resources/models/melspectrogram.tflite
''')

cells.insert(5, md_cell('''**Note (patched for this project):** the original cell here also installed
`piper-sample-generator`/`piper-phonemize` (for synthetic TTS clip generation) and
`tensorflow-cpu`/`tensorflow_probability`/`onnx_tf` (for optional tflite export).
Both are dropped: piper-phonemize has no Python 3.13 wheel at all (Colab default as
of this notebook), and the tflite export path isn't needed since we only use the
ONNX model. Both the positive clips ("rocky") and the synthetic adversarial negative
clips (decoy words/phrases) that piper would normally have generated are instead
produced locally with Kokoro — see the upload cell below, which replaces the
original synthetic-clip-generation step entirely.'''))

# --- Locate original cells after the insertion shifted indices by 1 ---
# Find "Modify values in the config" cell and the old --generate_clips cell by content.
def find_cell_idx(marker: str) -> int:
    for i, c in enumerate(cells):
        if marker in "".join(c["source"]):
            return i
    raise ValueError(f"cell containing {marker!r} not found")


config_idx = find_cell_idx('config["target_phrase"] = ["hey sebastian"]')
src = "".join(cells[config_idx]["source"])
src = src.replace('config["target_phrase"] = ["hey sebastian"]', 'config["target_phrase"] = ["rocky"]')
cells[config_idx]["source"] = src.splitlines(keepends=True)

# --- Replace the piper-based generate_clips cell with an upload+unzip step ---
generate_idx = find_cell_idx("--generate_clips")
cells[generate_idx] = code_cell('''# Step 1 (patched): upload rocky_clips.zip (built locally with training/generate_rocky_clips.py)
# and unzip it into the positive_train/positive_test/negative_train/negative_test dirs
# this pipeline expects, instead of running train.py --generate_clips (which needs
# piper-phonemize for both the positive TTS clips AND the synthetic adversarial negatives).

from google.colab import files
import zipfile
import shutil

uploaded = files.upload()  # select rocky_clips.zip when prompted
zip_name = next(iter(uploaded))

with zipfile.ZipFile(zip_name) as zf:
    zf.extractall(".")  # expects rocky_clips/{positive,negative}_{train,test}/*.wav

for split in ["positive_train", "positive_test", "negative_train", "negative_test"]:
    dest_dir = os.path.join(config["output_dir"], config["model_name"], split)
    os.makedirs(dest_dir, exist_ok=True)
    src_dir = os.path.join("rocky_clips", split)
    for fname in os.listdir(src_dir):
        shutil.copy(os.path.join(src_dir, fname), dest_dir)
    print(f"{split}: {len(os.listdir(dest_dir))} clips")
''')

# --- Fix RIR/Audioset/FMA download cells: newer `datasets` decodes Audio columns via
# torchcodec, returning an AudioDecoder object instead of the old {'path':..., 'array':...}
# dict, so row['audio']['path'] now raises TypeError (confirmed by reproducing locally
# with the same datasets/torch/torchcodec/ffmpeg stack). Also note: AudioDecoder rows
# don't carry the original filename at all anymore, so clips get index-based names instead.
rir_idx = find_cell_idx("# Download room impulse responses collected by MIT")
cells[rir_idx] = code_cell('''# Download room impulse responses collected by MIT
# https://mcdermottlab.mit.edu/Reverb/IR_Survey.html
# (patched: newer `datasets` returns an AudioDecoder object for the "audio" column,
# not the old {'path':..., 'array':...} dict — use .get_all_samples() instead)

output_dir = "./mit_rirs"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
rir_dataset = datasets.load_dataset("davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True)

# Save clips to 16-bit PCM wav files
for i, row in enumerate(tqdm(rir_dataset)):
    samples = row["audio"].get_all_samples()
    audio = samples.data.numpy()
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    scipy.io.wavfile.write(os.path.join(output_dir, f"rir_{i:05d}.wav"), samples.sample_rate, (audio * 32767).astype(np.int16))
''')

background_idx = find_cell_idx("## Download noise and background audio")
cells[background_idx] = code_cell('''## Download noise and background audio

# Audioset Dataset (https://research.google.com/audioset/dataset/index.html)
# Download one part of the audioset .tar files, extract, and convert to 16khz
# For full-scale training, it's recommended to download the entire dataset from
# https://huggingface.co/datasets/agkphysics/AudioSet, and
# even potentially combine it with other background noise datasets (e.g., FSD50k, Freesound, etc.)
# (patched: converts the already-local .flac files directly with librosa instead of
# going through datasets.Audio/torchcodec — simpler and sidesteps the AudioDecoder change)

import librosa

if not os.path.exists("audioset"):
    os.mkdir("audioset")

fname = "bal_train09.tar"
out_dir = f"audioset/{fname}"
link = "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/" + fname
!wget -nc -O {out_dir} {link}
!cd audioset && tar -xvf bal_train09.tar

output_dir = "./audioset_16k"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for path in tqdm(list(Path("audioset/audio").glob("**/*.flac"))):
    audio, sr = librosa.load(str(path), sr=16000, mono=True)
    scipy.io.wavfile.write(os.path.join(output_dir, path.name.replace(".flac", ".wav")), 16000, (audio * 32767).astype(np.int16))

# Free Music Archive dataset (https://github.com/mdeff/fma)
# (patched: same AudioDecoder fix as the MIT RIR cell above — streaming HF dataset,
# so we still need datasets' own decoder, just the current API for it)
output_dir = "./fma"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
fma_dataset = datasets.load_dataset("rudraml/fma", name="small", split="train", streaming=True)
fma_dataset = iter(fma_dataset)

n_hours = 1  # use only 1 hour of clips for this example notebook, recommend increasing for full-scale training
n_clips = n_hours * 3600 // 30  # this works because the FMA dataset is all 30 second clips
for i in tqdm(range(n_clips)):
    row = next(fma_dataset)
    samples = row["audio"].get_all_samples()
    audio = samples.data.numpy()
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    if samples.sample_rate != 16000:
        audio = librosa.resample(audio, orig_sr=samples.sample_rate, target_sr=16000)
    scipy.io.wavfile.write(os.path.join(output_dir, f"fma_{i:05d}.wav"), 16000, (audio * 32767).astype(np.int16))
''')

# --- Drop the optional tflite conversion cell (needs tensorflow/onnx_tf, not installed) ---
tflite_idx = find_cell_idx("def convert_onnx_to_tflite(onnx_model_path, output_path):")
cells[tflite_idx] = md_cell('''**Skipped (patched for this project):** the optional tflite re-export retry cell
was here. It needs `tensorflow`/`onnx_tf`, which this patched notebook doesn't
install (no Python 3.13 wheel for the pinned `tensorflow-cpu==2.8.1`, and not
needed since Rocky only uses the `.onnx` model). `train.py --train_model` already
exports `my_custom_model/rocky.onnx` on its own — that's the file to download.''')

with open(NB_OUT, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Wrote {NB_OUT}")
print(f"Total cells: {len(cells)}")
