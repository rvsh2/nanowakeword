#!/usr/bin/env bash
set -euo pipefail

# Trenuje modele NanoWakeWord dla wake worda "Agata" we wszystkich architekturach.
#
# Dlaczego poprzednia wersja konczyla sie na 0%:
#   Trening szedl na przykladowych danych z repo (po JEDNYM pliku .wav w
#   folderach positive/negative). Z jednej probki sampler nie byl w stanie
#   ulozyc ani jednego batcha (targets=32 + negatives=96), wiec DataLoader
#   byl pusty i petla treningowa konczyla sie natychmiast.
#
# Ten skrypt najpierw generuje syntetyczne dane (Piper TTS, polskie glosy),
# a dopiero potem trenuje wszystkie architektury.

PYTHON_BIN="${PYTHON_BIN:-python3}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKIP_VENV_CHECK="${SKIP_VENV_CHECK:-0}"

if [[ "${SKIP_VENV_CHECK}" != "1" && -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" ]]; then
  echo "[ERROR] Uruchom ten skrypt w aktywnym virtualenvu."
  echo "[HINT] Przyklad:"
  echo "  source .venv/bin/activate"
  echo "  ./train_agata_models.sh"
  echo "[INFO] Jesli chcesz uruchomic mimo to, ustaw SKIP_VENV_CHECK=1."
  exit 1
fi

# ===== Ustawienia danych =====
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/training_data/agata}"
POSITIVE_DIR="${1:-${POSITIVE_DIR:-$DATA_ROOT/positive}}"
NEGATIVE_DIR="${2:-${NEGATIVE_DIR:-$DATA_ROOT/negative}}"
POSITIVE_VAL_DIR="${POSITIVE_VAL_DIR:-$DATA_ROOT/positive_val}"
NEGATIVE_VAL_DIR="${NEGATIVE_VAL_DIR:-$DATA_ROOT/negative_val}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/trained_models/agata}"

# ===== Generowanie danych (Piper TTS) =====
GENERATE_DATA="${GENERATE_DATA:-auto}"   # auto = generuj gdy danych jest za malo; 1 = kasuje stare dane i generuje od zera; 0 = nigdy
N_POS="${N_POS:-400}"
N_NEG="${N_NEG:-1200}"
N_POS_VAL="${N_POS_VAL:-60}"
N_NEG_VAL="${N_NEG_VAL:-200}"
TTS_MODELS_DIR="${TTS_MODELS_DIR:-$PROJECT_ROOT/NwwResourcesModel/tts_models}"
TTS_VOICES=(darkman gosia mc_speech)     # polskie glosy Piper (pl_PL, medium)

# ===== Szumy tla i pogłos (RIR) do augmentacji =====
NOISE_ROOT="${NOISE_ROOT:-$PROJECT_ROOT/training_data/noise}"
BACKGROUND_DIR="${BACKGROUND_DIR:-$NOISE_ROOT/background}"   # ESC-50 (dzwieki otoczenia)
RIR_DIR="${RIR_DIR:-$NOISE_ROOT/rir}"                        # MIT IR Survey (pogłos pomieszczen)
DOWNLOAD_NOISE="${DOWNLOAD_NOISE:-auto}"  # auto = pobierz gdy foldery puste; 0 = nie pobieraj

# ===== Opcje treningu =====
MODEL_TYPES=(dnn rnn cnn lstm gru crnn tcn quartznet conformer e_branchformer bcresnet)
MODEL_PREFIX="${MODEL_PREFIX:-agata}"
WAKE_WORD="${WAKE_WORD:-Agata}"
STEPS="${STEPS:-30000}"
BATCH_TARGETS="${BATCH_TARGETS:-32}"
BATCH_NEGATIVES="${BATCH_NEGATIVES:-96}"
AUG_ROUNDS="${AUG_ROUNDS:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
# UWAGA: --overwrite jest domyslnie wlaczone, bo po nieudanych probach w
# trained_models/agata/*/features zostaly pliki cech z 1 probka. Bez
# nadpisania transform_clips pominie regeneracje i trening znow nie ruszy.
OVERWRITE="${OVERWRITE:-1}"
DISTILL="${DISTILL:-1}"

count_wavs() {
  [ -d "$1" ] || { echo 0; return; }
  find "$1" -maxdepth 1 -type f -iname '*.wav' | wc -l
}

download_voice() {
  local voice="$1"
  local base_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/${voice}/medium"
  local model="pl_PL-${voice}-medium"
  if [ ! -f "$TTS_MODELS_DIR/$model.onnx" ]; then
    echo "[INFO] Pobieram glos TTS: $model"
    # pobieranie do pliku tymczasowego + mv, zeby przerwany download
    # nie zostawil niekompletnego .onnx (skrypt sprawdza tylko istnienie pliku)
    curl -sSL --fail --retry 5 --retry-delay 3 --retry-all-errors \
      -o "$TTS_MODELS_DIR/$model.onnx.tmp" "$base_url/$model.onnx"
    curl -sSL --fail --retry 5 --retry-delay 3 --retry-all-errors \
      -o "$TTS_MODELS_DIR/$model.onnx.json" "$base_url/$model.onnx.json"
    mv "$TTS_MODELS_DIR/$model.onnx.tmp" "$TTS_MODELS_DIR/$model.onnx"
  fi
}

generate_training_data() {
  echo "[INFO] Generuje syntetyczne dane treningowe (Piper TTS)..."

  if ! "$PYTHON_BIN" -c "import piper" 2>/dev/null; then
    echo "[INFO] Brak pakietu piper-tts - instaluje w aktywnym srodowisku..."
    "$PYTHON_BIN" -m pip install piper-tts
  fi

  mkdir -p "$TTS_MODELS_DIR" "$POSITIVE_DIR" "$NEGATIVE_DIR" "$POSITIVE_VAL_DIR" "$NEGATIVE_VAL_DIR"
  for v in "${TTS_VOICES[@]}"; do
    download_voice "$v"
  done

  local gen_config
  gen_config="$(mktemp --suffix=.yaml)"
  cat > "$gen_config" <<EOF
target_phrase: "Agata"
tts_settings:
  models_dir: "$TTS_MODELS_DIR"
  length_scales: [0.85, 1.0, 1.15]
  noise_scales: [0.5, 0.667, 0.8]
  noise_w_scales: [0.7, 0.9]

data_generation_tasks:
  - name: "Pozytywne - Agata"
    output_dir: "$POSITIVE_DIR"
    num_samples: $N_POS
    file_prefix: "pos"
    text_source:
      type: "fixed_phrase"

  - name: "Pozytywne walidacyjne - Agata"
    output_dir: "$POSITIVE_VAL_DIR"
    num_samples: $N_POS_VAL
    file_prefix: "pos_val"
    text_source:
      type: "fixed_phrase"

  - name: "Negatywne - trudne i codzienne frazy"
    output_dir: "$NEGATIVE_DIR"
    num_samples: $N_NEG
    file_prefix: "neg"
    text_source:
      type: "from_list"
      phrases: &negative_phrases
        # fonetycznie podobne do "Agata" (trudne negatywy)
        - "agawa"
        - "armata"
        - "chata"
        - "tata"
        - "mata"
        - "wata"
        - "data"
        - "rata"
        - "sałata"
        - "herbata"
        - "sonata"
        - "fregata"
        - "brygada"
        - "łopata"
        - "adwokata"
        - "a gama"
        - "aga tam"
        - "agenda"
        - "agencja"
        # zwykla mowa (typowe negatywy)
        - "włącz światło w kuchni"
        - "wyłącz telewizor"
        - "jaka będzie jutro pogoda"
        - "przypomnij mi o spotkaniu"
        - "dzień dobry, co słychać"
        - "zrób głośniej muzykę"
        - "która jest teraz godzina"
        - "zamknij drzwi wejściowe"
        - "dobranoc wszystkim"
        - "poproszę kawę i herbatę"
        - "gdzie są moje klucze"
        - "ustaw minutnik na pięć minut"
        - "opowiedz mi coś ciekawego"
        - "na dworze pada deszcz"
        - "kot śpi na kanapie"

  - name: "Negatywne walidacyjne"
    output_dir: "$NEGATIVE_VAL_DIR"
    num_samples: $N_NEG_VAL
    file_prefix: "neg_val"
    text_source:
      type: "from_list"
      phrases: *negative_phrases
EOF

  "$PYTHON_BIN" - "$gen_config" <<'PY'
import sys, yaml
from nanowakeword.generate_clips import generate_clips
with open(sys.argv[1], encoding="utf-8") as f:
    config = yaml.safe_load(f)
generate_clips(config)
PY
  rm -f "$gen_config"
}

prepare_noise_data() {
  # Szumy tla: ESC-50 (2000 klipow dzwiekow otoczenia, licencja CC)
  if [ "$(count_wavs "$BACKGROUND_DIR")" -eq 0 ]; then
    echo "[INFO] Pobieram zbior szumow tla ESC-50 (~600 MB)..."
    mkdir -p "$BACKGROUND_DIR"
    local esc_zip="$NOISE_ROOT/esc50.zip"
    curl -sSL --fail --retry 5 --retry-delay 3 --retry-all-errors \
      -o "$esc_zip" "https://github.com/karolpiczak/ESC-50/archive/master.zip"
    echo "[INFO] Rozpakowuje i konwertuje do 16 kHz mono..."
    "$PYTHON_BIN" - "$esc_zip" "$BACKGROUND_DIR" <<'PY'
import sys, zipfile, io, os
import torch, torchaudio
zip_path, out_dir = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(zip_path) as z:
    names = [n for n in z.namelist() if n.endswith(".wav")]
    for i, name in enumerate(names):
        with z.open(name) as f:
            wav, sr = torchaudio.load(io.BytesIO(f.read()))
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        torchaudio.save(os.path.join(out_dir, os.path.basename(name)), wav, 16000)
        if (i + 1) % 200 == 0:
            print(f"  skonwertowano {i + 1}/{len(names)}")
print(f"Gotowe: {len(names)} plikow szumu w {out_dir}")
PY
    rm -f "$esc_zip"
  fi

  # RIR: MIT Environmental Impulse Responses (271 plikow, juz 16 kHz)
  if [ "$(count_wavs "$RIR_DIR")" -eq 0 ]; then
    echo "[INFO] Pobieram odpowiedzi impulsowe MIT IR Survey (~100 MB)..."
    mkdir -p "$RIR_DIR"
    "$PYTHON_BIN" - "$RIR_DIR" <<'PY'
import sys, json, os, urllib.request
out_dir = sys.argv[1]
api = "https://huggingface.co/api/datasets/davidscripka/MIT_environmental_impulse_responses/tree/main/16khz"
files = [e["path"] for e in json.load(urllib.request.urlopen(api)) if e["path"].endswith(".wav")]
base = "https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses/resolve/main/"
for i, path in enumerate(files):
    dest = os.path.join(out_dir, os.path.basename(path))
    if not os.path.exists(dest):
        urllib.request.urlretrieve(base + path.replace(" ", "%20"), dest)
    if (i + 1) % 50 == 0:
        print(f"  pobrano {i + 1}/{len(files)}")
print(f"Gotowe: {len(files)} plikow RIR w {out_dir}")
PY
  fi
}

# ===== Etap 1: dane =====
POS_COUNT=$(count_wavs "$POSITIVE_DIR")
NEG_COUNT=$(count_wavs "$NEGATIVE_DIR")

NEED_DATA=0
if [ "$POS_COUNT" -lt "$BATCH_TARGETS" ] || [ "$NEG_COUNT" -lt "$BATCH_NEGATIVES" ]; then
  NEED_DATA=1
fi

if [ "$GENERATE_DATA" = "1" ]; then
  echo "[INFO] GENERATE_DATA=1 - kasuje stare dane i generuje od zera"
  for d in "$POSITIVE_DIR" "$NEGATIVE_DIR" "$POSITIVE_VAL_DIR" "$NEGATIVE_VAL_DIR"; do
    [ -d "$d" ] && find "$d" -maxdepth 1 -type f -iname '*.wav' -delete
  done
fi

if [ "$GENERATE_DATA" = "1" ] || { [ "$GENERATE_DATA" = "auto" ] && [ "$NEED_DATA" = "1" ]; }; then
  generate_training_data
  POS_COUNT=$(count_wavs "$POSITIVE_DIR")
  NEG_COUNT=$(count_wavs "$NEGATIVE_DIR")
fi

if [ "$POS_COUNT" -lt "$BATCH_TARGETS" ] || [ "$NEG_COUNT" -lt "$BATCH_NEGATIVES" ]; then
  echo "[ERROR] Za malo danych treningowych:"
  echo "  pozytywne: $POS_COUNT plikow w $POSITIVE_DIR (minimum: $BATCH_TARGETS)"
  echo "  negatywne: $NEG_COUNT plikow w $NEGATIVE_DIR (minimum: $BATCH_NEGATIVES)"
  echo "  Przy mniejszej liczbie probek sampler nie ulozy ani jednego batcha"
  echo "  i trening zakonczy sie natychmiast na 0%."
  exit 1
fi

POS_VAL_COUNT=$(count_wavs "$POSITIVE_VAL_DIR")
NEG_VAL_COUNT=$(count_wavs "$NEGATIVE_VAL_DIR")

if [ "$DOWNLOAD_NOISE" != "0" ]; then
  prepare_noise_data
fi
BG_COUNT=$(count_wavs "$BACKGROUND_DIR")
RIR_COUNT=$(count_wavs "$RIR_DIR")

# ===== Etap 2: trening =====
mkdir -p "$OUTPUT_DIR"

echo "[INFO] Start treningow dla Agaty"
echo "[INFO] positive_data=$POSITIVE_DIR ($POS_COUNT plikow)"
echo "[INFO] negative_data=$NEGATIVE_DIR ($NEG_COUNT plikow)"
echo "[INFO] walidacja: $POS_VAL_COUNT pozytywnych / $NEG_VAL_COUNT negatywnych"
echo "[INFO] augmentacja: $BG_COUNT plikow szumu tla / $RIR_COUNT plikow RIR"
echo "[INFO] output_dir=$OUTPUT_DIR"
echo "[INFO] steps=$STEPS"

for mt in "${MODEL_TYPES[@]}"; do
  MODEL_NAME="${MODEL_PREFIX}_${mt}"
  CMD=(
    "$PYTHON_BIN" "$PROJECT_ROOT/train_wakeword.py"
    "--positive-data" "$POSITIVE_DIR"
    "--negative-data" "$NEGATIVE_DIR"
    "--output-dir" "$OUTPUT_DIR"
    "--model-name" "$MODEL_NAME"
    "--model-type" "$mt"
    "--wake-word" "$WAKE_WORD"
    "--steps" "$STEPS"
    "--batch-targets" "$BATCH_TARGETS"
    "--batch-negatives" "$BATCH_NEGATIVES"
    "--augmentation-rounds" "$AUG_ROUNDS"
    "--num-workers" "$NUM_WORKERS"
  )

  if [ "$POS_VAL_COUNT" -gt 0 ] && [ "$NEG_VAL_COUNT" -gt 0 ]; then
    CMD+=("--positive-val" "$POSITIVE_VAL_DIR" "--negative-val" "$NEGATIVE_VAL_DIR")
  fi
  if [ "$BG_COUNT" -gt 0 ]; then
    CMD+=("--background-dir" "$BACKGROUND_DIR")
  fi
  if [ "$RIR_COUNT" -gt 0 ]; then
    CMD+=("--rir-dir" "$RIR_DIR")
  fi
  if [ "$OVERWRITE" = "1" ]; then
    CMD+=("--overwrite")
  fi
  if [ "$DISTILL" = "1" ]; then
    CMD+=("--distill")
  fi

  echo "[RUN] ${CMD[*]}"
  "${CMD[@]}"
  echo "[OK] Model $MODEL_NAME zakonczony"
  echo ""
done

echo "[DONE] Wszystkie modele zostaly wytrenowane. Wyniki w: $OUTPUT_DIR"
