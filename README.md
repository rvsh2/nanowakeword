# NanoWakeWord (training fork)

Fork of [arcosoph/nanowakeword](https://github.com/arcosoph/nanowakeword) —
a lightweight wake word detection engine with a full training pipeline.
This fork adds the changes used to train the **Agata** wake word models
served by [wyoming-nanowakeword](https://github.com/rvsh2/wyoming-nanowakeword),
plus the training scripts and data recipes themselves.

Upstream documentation still applies:
[README](https://github.com/arcosoph/nanowakeword#readme) ·
[CONFIGURATION_GUIDE](CONFIGURATION_GUIDE.md).

## What this fork changes

**Augmentation controls** (`nanowakeword/data/augment_clips.py`) — two new
`augmentation_settings` keys:

- `background_prob` (default `1.0`, upstream behavior): probability of
  mixing background noise into a clip. Upstream mixes noise into 100% of
  samples, so a model never sees clean audio — the distribution streaming
  inference actually runs on. `0.5` keeps half of the clips clean; in our
  training this repaired recall that noise-in-everything had destroyed.
- `end_align_prob` (default `0.0`, upstream behavior): probability of
  placing the foreground near the end of the training window. Streaming
  detection scores windows in which the wake word has just ended, so
  end-aligned placement matches deployment; `0.75` worked well.

**Small-dataset sampler fix** (`nanowakeword/data/data_sampler.py`): a pool
smaller than its batch quota now still yields one batch (sampling with
replacement) instead of producing an empty DataLoader and a training run
that silently ends at step 0.

**Offline model resolution** (`nanowakeword/interpreter/models/_registry.py`):
set `NANOWAKEWORD_MODELS_DIR=/path/to/models` to resolve the
melspectrogram/embedding/VAD models from a local directory instead of
downloading — for CI and air-gapped hosts.

## Training scripts (Agata)

- `train_wakeword.py` — single-file helper: builds the YAML config and runs
  the `nanowakeword` CLI for any supported architecture
  (`--list-models` prints them). All pipeline stages in one command.
- `train_agata_models.sh` — end-to-end batch run: generates synthetic
  Polish data with Piper TTS, downloads ESC-50 background noise and MIT RIR
  impulse responses, then trains every architecture.
- `generate_agata_v2.yaml` — data-generation recipe: positives include the
  accepted variants ("Agatka", "Agato"), negatives cover hard phonetic
  neighbours ("armata", "agenda", "tata", ...) plus assistant commands and
  everyday speech; validation uses held-out voices and held-out phrases.
- `generate_agata_hard_negatives.yaml` — hard phonetic negatives as a
  separate feature pool. The batch sampler picks feature files uniformly,
  so a dedicated pool gives hard negatives a fixed share of draws instead
  of drowning among easy speech.
- `make_noise_negatives.py` — speech-free negatives (ESC-50 crops,
  white/pink noise, near-silence). A wake word device hears noise or
  silence most of its life; without these examples labelled 0, models
  scored white noise as high as 0.95.

## Lessons that shaped these changes

Measured across 13 trained model variants for one wake word (details in the
wyoming-nanowakeword changelog):

1. Models trained with noise mixed into every sample behave arbitrarily on
   clean streaming audio (`background_prob` fixes this).
2. Voice diversity in positives is what generalizes: 7 TTS voices produced
   a model deaf to unfamiliar speakers; 29 voices (Piper + ElevenLabs)
   scored ≥0.999 on held-out voices.
3. Synthetic-only data yields either a sensitive model that false-triggers
   on real speech or a conservative one with no usable margin — never both.
   Ship a sensitive detector and verify candidates with ASR (implemented in
   wyoming-nanowakeword as `--verify-asr-url`).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[train]

# everything at once (data generation + all architectures):
./train_agata_models.sh

# or one architecture with explicit data:
python train_wakeword.py \
  --positive-data training_data/agata_v2/positive \
  --negative-data training_data/agata_v2/negative \
  --model-type conformer --wake-word Agata \
  --steps 60000 --batch-targets 48 --batch-negatives 96 \
  --background-dir training_data/noise/background \
  --rir-dir training_data/noise/rir --distill
```

Generated data lands in `training_data/`, trained models in
`trained_models/<name>/model/<name>.onnx` (plus a distilled `_lite` cascade
gate). Both directories are git-ignored.

## License

Apache-2.0, same as upstream.
