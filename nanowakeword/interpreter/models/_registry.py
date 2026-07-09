from pathlib import Path
import os
from nanowakeword.utils.download_files import download_file


class ModelRegistry:
    """
    Central registry for NanoWakeWord model files.

    This class provides lazy access to required model files. Model paths
    are resolved dynamically using attribute access. If a requested model
    is not present locally, it is automatically downloaded and cached
    in a structured directory layout.

    Example:
        models = ModelRegistry()
        path = models.embedding_model_onnx
    """

    def __init__(self):
        """
        Initialize the model registry.

        Models are stored under:
            nanowakeword/interpreter/models/

        Each model type is placed inside its own subdirectory.
        """
        self._base_dir = (
            # Path(__file__).parent / "interpreter" / "models"
            Path(__file__).parent.resolve()
        )
        self._base_dir.mkdir(parents=True, exist_ok=True)

        self._registry = {
            "melspectrogram.onnx": {
                "subdir": "mel_spectrogram",
                "url": "https://github.com/arcosoph/nanowakeword/releases/download/models3/melspectrogram.onnx",
            },
            "embedding_model.onnx": {
                "subdir": "embedding",
                "url": "https://github.com/arcosoph/nanowakeword/releases/download/models3/embedding_model.onnx",
            },
            "silero_vad.onnx": {
                "subdir": "vad",
                "url": "https://github.com/arcosoph/nanowakeword/releases/download/models3/silero_vad.onnx",
            },
        }

    def _download_if_needed(self, filename: str) -> Path:
        """
        Ensure that the requested model file exists locally.

        If the file is not found, it is downloaded from the registered URL
        and stored in its designated subdirectory.

        Args:
            filename: Name of the model file (e.g., "embedding_model.onnx").

        Returns:
            Path to the local model file.

        Raises:
            FileNotFoundError: If the filename is not registered.
            IOError: If the file cannot be downloaded.
        """
        if filename not in self._registry:
            raise FileNotFoundError(filename)

        entry = self._registry[filename]
        model_dir = self._base_dir / entry["subdir"]
        model_dir.mkdir(parents=True, exist_ok=True)

        # Optional environment override: a directory containing all files from registry.
        # This is useful in offline environments (e.g. CI, air-gapped hosts).
        offline_models_dir = os.environ.get("NANOWAKEWORD_MODELS_DIR")
        if offline_models_dir:
            offline_root = Path(offline_models_dir)
            offline_candidates = [
                offline_root / filename,
                offline_root / entry["subdir"] / filename,
            ]
            for offline_path in offline_candidates:
                if offline_path.exists():
                    return offline_path

        model_path = model_dir / filename

        if not model_path.exists():
            # print(f"[nanowakeword] Downloading model '{filename}'...")
            try:
                download_file(entry["url"], str(model_dir))
            except Exception as e:
                hint = (
                    "If you are offline, set NANOWAKEWORD_MODELS_DIR to a directory "
                    "containing the required model files."
                )
                raise IOError(f"Failed to download model: {filename}. {hint}") from e

        return model_path

    def __getattr__(self, name: str) -> str:
        """
        Dynamically resolve model paths via attribute access.

        Attribute names must follow the pattern:
            <model_name>_<extension>

        Example:
            models.embedding_model_onnx  -> embedding_model.onnx

        Args:
            name: Attribute name being accessed.

        Returns:
            String path to the resolved model file.

        Raises:
            AttributeError: If the model cannot be resolved or downloaded.
        """
        if "_" not in name:
            raise AttributeError(name)

        base, ext = name.rsplit("_", 1)
        filename = f"{base}.{ext}"

        try:
            return str(self._download_if_needed(filename))
        except FileNotFoundError:
            raise AttributeError(
                f"Model '{filename}' is not registered."
            )


models = ModelRegistry()
