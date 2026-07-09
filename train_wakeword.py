#!/usr/bin/env python3
"""Single-file helper to train a wake word model with any NanoWakeWord architecture."""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from typing import Optional

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "PyYAML is required for this script. Install it with: pip install pyyaml"
    ) from exc


SUPPORTED_MODELS = [
    "dnn",
    "rnn",
    "cnn",
    "lstm",
    "gru",
    "crnn",
    "tcn",
    "quartznet",
    "transformer",
    "conformer",
    "e_branchformer",
    "bcresnet",
    "custom",
    "custom_model",
]


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a wake word model using NanoWakeWord from one script. "
            "The script builds a temporary YAML config and runs `nanowakeword`."
        )
    )
    parser.add_argument("--positive-data", default=None, help="Folder with wake-word .wav files")
    parser.add_argument("--negative-data", default=None, help="Folder with non wake-word .wav files")
    parser.add_argument("--output-dir", default="./trained_models", help="Base output directory")
    parser.add_argument("--model-name", default="", help="Model name (default: auto-generated)")
    parser.add_argument(
        "--model-type",
        default="dnn",
        choices=SUPPORTED_MODELS,
        help="Architecture for the full model",
    )
    parser.add_argument("--wake-word", default="wake_word", help="Wake word text for metadata")
    parser.add_argument("--steps", type=int, default=None, help="Training steps override")
    parser.add_argument("--n-blocks", type=int, default=None, help="Model depth override")
    parser.add_argument("--layer-size", type=int, default=None, help="Model width override")
    parser.add_argument("--embedding-dim", type=int, default=None, help="Embedding size override")
    parser.add_argument("--dropout-prob", type=float, default=None, help="Dropout override")
    parser.add_argument("--learning-rate", type=float, default=None, help="Max learning rate override")
    parser.add_argument("--batch-targets", type=int, default=None, help="Batch positives per step")
    parser.add_argument("--batch-negatives", type=int, default=None, help="Batch negatives per step")
    parser.add_argument("--background-dir", action="append", default=[], help="Noise/background folder")
    parser.add_argument("--rir-dir", action="append", default=[], help="RIR folder")
    parser.add_argument("--positive-val", default=None, help="Optional positive validation folder")
    parser.add_argument("--negative-val", default=None, help="Optional negative validation folder")
    parser.add_argument("--augmentation-rounds", type=int, default=1, help="Per-input augmentation rounds")
    parser.add_argument("--custom-module", default=None, help="For --model-type custom: module path")
    parser.add_argument("--custom-class", default=None, help="For --model-type custom: class name")
    parser.add_argument("--num-workers", type=int, default=None, help="Num workers for training/transform")
    parser.add_argument("--resume", default=None, help="Resume from project directory")
    parser.add_argument("--distill", action="store_true", help="Build lite model with distillation")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite generated feature files")
    parser.add_argument("--force-verify", action="store_true", help="Re-verify audio directories")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated YAML and exits without running training",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable for training call")
    parser.add_argument("--list-models", action="store_true", help="Print supported --model-type values and exit")
    return parser.parse_args()


def _sanitize_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip().lower()) or "wakeword"


def _validate_audio_dir(path: str, label: str) -> str:
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{label} folder not found: {path}")
    return path


def _safe_optional_dir(path: Optional[str], label: str) -> Optional[str]:
    if not path:
        return None
    return _validate_audio_dir(path, label)


def _build_config(args: argparse.Namespace, output_root: str, model_name: str) -> dict:
    positive_dir = _validate_audio_dir(args.positive_data, "positive-data")
    negative_dir = _validate_audio_dir(args.negative_data, "negative-data")

    background_dirs = [
        _validate_audio_dir(d, "background-dir")
        for d in args.background_dir
    ]
    rir_dirs = [
        _validate_audio_dir(d, "rir-dir")
        for d in args.rir_dir
    ]

    positive_val = _safe_optional_dir(args.positive_val, "positive-val")
    negative_val = _safe_optional_dir(args.negative_val, "negative-val")

    if (args.positive_val is not None) ^ (args.negative_val is not None):
        raise ValueError(
            "Provide both --positive-val and --negative-val together, or omit both."
        )

    if args.model_type in {"custom", "custom_model"}:
        if not args.custom_module or not args.custom_class:
            raise ValueError(
                "When --model-type is custom/custom_model you must pass "
                "--custom-module and --custom-class."
            )

    feature_dir = os.path.join(output_root, model_name, "features")
    positive_feature_path = os.path.join(feature_dir, "positive_features.npy")
    negative_feature_path = os.path.join(feature_dir, "negative_features.npy")

    config = {
        "model_name": model_name,
        "output_dir": output_root,
        "positive_data_path": positive_dir,
        "negative_data_path": negative_dir,
        "background_paths": background_dirs,
        "rir_paths": rir_dirs,
        "generate_clips": False,
        "transform_clips": True,
        "train_model": True,
        "distillation": {"enabled": bool(args.distill)},
        "target_phrase": args.wake_word,
        "model_type": args.model_type,
        "feature_generation_manifest": {
            "targets": {
                "output_filename": "positive_features.npy",
                "input_audio_dirs": [positive_dir],
                "output_type": "features",
                "use_background_noise": bool(background_dirs),
                "use_rir": bool(rir_dirs),
                "augmentation_rounds": args.augmentation_rounds,
            },
            "negatives": {
                "output_filename": "negative_features.npy",
                "input_audio_dirs": [negative_dir],
                "output_type": "features",
                "use_background_noise": bool(background_dirs),
                "use_rir": bool(rir_dirs),
                "augmentation_rounds": args.augmentation_rounds,
            },
        },
        "feature_manifest": {
            "targets": {
                "t": positive_feature_path,
            },
            "negatives": {
                "n": negative_feature_path,
            },
        },
    }

    if positive_val and negative_val:
        config["feature_generation_manifest"]["targets_val"] = {
            "output_filename": "positive_val_features.npy",
            "input_audio_dirs": [positive_val],
            "output_type": "features",
            "use_background_noise": bool(background_dirs),
            "use_rir": bool(rir_dirs),
            "augmentation_rounds": max(1, args.augmentation_rounds // 2),
        }
        config["feature_generation_manifest"]["negatives_val"] = {
            "output_filename": "negative_val_features.npy",
            "input_audio_dirs": [negative_val],
            "output_type": "features",
            "use_background_noise": bool(background_dirs),
            "use_rir": bool(rir_dirs),
            "augmentation_rounds": max(1, args.augmentation_rounds // 2),
        }
        config["feature_manifest"]["targets_val"] = {
            "t_v": os.path.join(feature_dir, "positive_val_features.npy")
        }
        config["feature_manifest"]["negatives_val"] = {
            "n_v": os.path.join(feature_dir, "negative_val_features.npy")
        }

    if args.steps is not None:
        config["steps"] = args.steps
    if args.n_blocks is not None:
        config["n_blocks"] = args.n_blocks
    if args.layer_size is not None:
        config["layer_size"] = args.layer_size
    if args.embedding_dim is not None:
        config["embedding_dim"] = args.embedding_dim
    if args.dropout_prob is not None:
        config["dropout_prob"] = args.dropout_prob
    if args.learning_rate is not None:
        config["learning_rate_max"] = args.learning_rate
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers

    if args.batch_targets is not None or args.batch_negatives is not None:
        config["batch_composition"] = {
            "targets": args.batch_targets if args.batch_targets is not None else 32,
            "negatives": args.batch_negatives if args.batch_negatives is not None else 96,
        }

    if args.model_type in {"custom", "custom_model"}:
        config["custom_model_config"] = {
            "module_path": args.custom_module,
            "class_name": args.custom_class,
        }

    if args.force_verify:
        config["force_verify"] = True

    return config


def _write_yaml(config: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def _run_training(args: argparse.Namespace, config_path: str, model_name: str, output_root: str):
    project_dir = os.path.abspath(os.path.join(output_root, model_name))
    cli_argv = [
        "nanowakeword",
        "-c",
        config_path,
        "-t",
        "-T",
    ]
    if args.distill:
        cli_argv.append("-d")
    if args.overwrite:
        cli_argv.append("--overwrite")
    if args.force_verify:
        cli_argv.append("-f")
    if args.resume:
        cli_argv.extend(["--resume", args.resume])

    runner = (
        "import sys; "
        "from nanowakeword.cli import main; "
        f"sys.argv = {repr(cli_argv)}; "
        "main()"
    )

    completed = subprocess.run([args.python, "-c", runner])
    if completed.returncode != 0:
        raise SystemExit(f"Training failed with exit code {completed.returncode}")

    print(f"Model directory: {project_dir}")
    print(f"ONNX model: {os.path.join(project_dir, 'model', model_name + '.onnx')}")


def main():
    args = _parse_args()

    if args.list_models:
        print("Supported model_type values:")
        for item in SUPPORTED_MODELS:
            print(f" - {item}")
        return

    if not args.positive_data or not args.negative_data:
        raise SystemExit(
            "Both --positive-data and --negative-data are required unless --list-models is used."
        )

    args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    model_name = args.model_name.strip()
    if not model_name:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{_sanitize_name(args.wake_word)}_{args.model_type}_{ts}"

    config = _build_config(args, args.output_dir, model_name)
    config_path = os.path.join(args.output_dir, f"{model_name}_train.yaml")

    if args.dry_run:
        print(yaml.safe_dump(config, sort_keys=False))
        print(f"Config path would be: {config_path}")
        return

    _write_yaml(config, config_path)
    _run_training(args, config_path, model_name, args.output_dir)


if __name__ == "__main__":
    main()
