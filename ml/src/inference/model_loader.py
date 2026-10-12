"""
Singleton Model Loader for Stage 5 Persona Engine.
Loads Qwen2.5-1.5B-Instruct with the stage5_v1 LoRA adapter once per process.
Supports 4-bit NF4 quantization on CUDA and graceful CPU fallback for development/testing.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("PersonaModelLoader")

# Root directory of ml
ML_ROOT = Path(__file__).resolve().parents[2]


class ModelLoader:
    """Thread-safe singleton model loader preventing repeated model loading."""

    _instance: Optional["ModelLoader"] = None
    _model: Optional[Any] = None
    _tokenizer: Optional[Any] = None
    _metadata: Optional[Dict[str, Any]] = None
    _adapter_path: Optional[Any] = None
    _adapter_loaded: bool = False
    _is_mock: bool = False
    _device: Optional[str] = None
    _quant_mode: Optional[str] = None
    _load_duration: float = 0.0

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    @classmethod
    def load_registry(cls, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Loads metadata from ml/models/registry/{model_id}.json."""
        import os
        target_id = model_id or os.environ.get("MODEL_VARIANT", "stage5_v1")
        # Handle candidate aliases if needed
        if target_id in {"stage7a_3b_candidate", "stage7a_candidate"}:
            target_id = "stage7a_qwen2.5_3b"

        registry_path = ML_ROOT / "models" / "registry" / f"{target_id}.json"
        if not registry_path.is_file():
            raise FileNotFoundError(f"Model registry entry not found at: {registry_path}")
        with open(registry_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def is_loaded(cls) -> bool:
        """Returns True if the model and tokenizer are loaded in memory."""
        return cls._model is not None and cls._tokenizer is not None

    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Returns current runtime model loading metadata."""
        import os
        default_id = os.environ.get("MODEL_VARIANT", "stage5_v1")

        adapter_loaded = cls._adapter_loaded
        if cls._model is not None and not cls._is_mock:
            try:
                from peft import PeftModel
                adapter_loaded = isinstance(cls._model, PeftModel)
            except (ImportError, TypeError):
                pass

        return {
            "loaded": cls.is_loaded(),
            "model_id": cls._metadata.get("model_id") if cls._metadata else default_id,
            "base_model": cls._metadata.get("base_model") if cls._metadata else "Qwen/Qwen2.5-1.5B-Instruct",
            "adapter_path": str(cls._adapter_path) if cls._adapter_path else (cls._metadata.get("adapter_path") if cls._metadata else "ml/models/adapters/stage5_v1"),
            "adapter_loaded": adapter_loaded,
            "is_mock": cls._is_mock,
            "device": cls._device or "unloaded",
            "quantization": cls._quant_mode or "none",
            "load_duration_seconds": round(cls._load_duration, 3),
        }

    @classmethod
    def load_model(
        cls,
        model_id: Optional[str] = None,
        force_cpu: bool = False,
        mock_model: Optional[Any] = None,
        mock_tokenizer: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        """
        Loads base model + LoRA adapter once per process.
        Strictly enforces adapter existence and loading when an adapter is specified.
        Silent fallback to the base model is strictly prohibited.
        """
        import os
        target_model_id = model_id or os.environ.get("MODEL_VARIANT", "stage5_v1")
        if target_model_id in {"stage7a_3b_candidate", "stage7a_candidate"}:
            target_model_id = "stage7a_qwen2.5_3b"

        if cls.is_loaded():
            if cls._metadata and cls._metadata.get("model_id") == target_model_id:
                logger.info(f"Model {target_model_id} already loaded in memory. Returning cached instance.")
                return cls._model, cls._tokenizer
            else:
                logger.info(f"Switching loaded model from {cls._metadata.get('model_id') if cls._metadata else 'unknown'} to {target_model_id}...")
                cls.unload()

        # Allow injecting mocks for tests without downloading weights
        if mock_model is not None and mock_tokenizer is not None:
            cls._model = mock_model
            cls._tokenizer = mock_tokenizer
            cls._metadata = cls.load_registry(target_model_id)
            cls._adapter_path = cls._metadata.get("adapter_path")
            cls._device = "mock"
            cls._quant_mode = "none"
            cls._is_mock = True
            cls._adapter_loaded = False
            cls._load_duration = 0.0
            return cls._model, cls._tokenizer

        import time
        start_time = time.time()

        cls._metadata = cls.load_registry(target_model_id)
        base_model_name = cls._metadata["base_model"]
        adapter_rel_path = cls._metadata.get("adapter_path")
        requires_adapter = bool(
            adapter_rel_path and str(adapter_rel_path).strip().lower() not in {"", "none", "null"}
        )

        override_adapter = os.environ.get("ADAPTER_OVERRIDE_PATH")
        if override_adapter:
            adapter_path = Path(override_adapter)
        elif adapter_rel_path and Path(adapter_rel_path).is_absolute():
            adapter_path = Path(adapter_rel_path)
        elif adapter_rel_path:
            adapter_path = ML_ROOT.parent / adapter_rel_path
        else:
            adapter_path = None

        cls._adapter_path = adapter_path

        # -------------------------------------------------------------
        # STRICT ADAPTER INTEGRITY CHECKS (Before touching model weights)
        # -------------------------------------------------------------
        if requires_adapter:
            if adapter_path is None or not adapter_path.is_dir():
                raise FileNotFoundError(
                    f"Adapter directory for variant '{target_model_id}' does not exist at: {adapter_path}. "
                    "A valid LoRA adapter directory containing trained weights is strictly required. "
                    "Silent fallback to base model is disabled."
                )

            config_file = adapter_path / "adapter_config.json"
            if not config_file.is_file():
                raise FileNotFoundError(
                    f"Adapter config missing for variant '{target_model_id}' at: {config_file}. "
                    "A valid adapter_config.json is required."
                )
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    adapter_cfg = json.load(f)
                if not isinstance(adapter_cfg, dict) or not adapter_cfg:
                    raise ValueError("adapter_config.json is empty or not a JSON dictionary.")
            except Exception as json_err:
                raise ValueError(
                    f"Invalid adapter_config.json for variant '{target_model_id}' at {config_file}: {json_err}"
                ) from json_err

            # Check for non-empty adapter weight binary
            candidate_weights = ["adapter_model.safetensors", "adapter_model.bin"]
            found_weights = [
                f for f in candidate_weights
                if (adapter_path / f).is_file() and (adapter_path / f).stat().st_size > 0
            ]
            if not found_weights:
                raise FileNotFoundError(
                    f"Adapter weights not found for variant '{target_model_id}' at: {adapter_path}. "
                    "Expected non-empty 'adapter_model.safetensors' or 'adapter_model.bin'. "
                    "Config-only adapters are not loadable; silent fallback to base model is disabled."
                )

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        has_cuda = torch.cuda.is_available() and not force_cpu
        cls._device = "cuda" if has_cuda else "cpu"
        cls._quant_mode = "4-bit NF4" if has_cuda else "float32"

        logger.info(f"PersonaGenerator: Loading model {target_model_id}...")
        logger.info(f"  Base Model       : {base_model_name}")
        logger.info(f"  Adapter          : {adapter_path}")
        logger.info(f"  Device           : {cls._device}")
        logger.info(f"  Quantization Mode: {cls._quant_mode}")

        # 1. Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 2. Load Base Model
        if has_cuda:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            logger.warning("Running model on CPU. Quantization disabled (inference will be slow).")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )

        # 3. Load LoRA Adapter
        if requires_adapter:
            logger.info(f"Applying LoRA adapter from: {adapter_path}")
            try:
                model = PeftModel.from_pretrained(base_model, str(adapter_path))
            except Exception as peft_err:
                logger.error(f"Failed to apply PeftModel from {adapter_path}: {peft_err}")
                raise RuntimeError(
                    f"Failed to load LoRA adapter for variant '{target_model_id}' from {adapter_path}: {peft_err}"
                ) from peft_err
            cls._adapter_loaded = True
        else:
            model = base_model
            cls._adapter_loaded = False

        model.eval()
        cls._model = model
        cls._tokenizer = tokenizer
        cls._is_mock = False
        cls._load_duration = time.time() - start_time

        logger.info(f"Model {target_model_id} loaded successfully in {cls._load_duration:.2f}s on {cls._device} (Adapter applied: {cls._adapter_loaded}).")
        return cls._model, cls._tokenizer

    @classmethod
    def unload(cls) -> None:
        """Unload model from memory (mainly for unit tests)."""
        cls._model = None
        cls._tokenizer = None
        cls._metadata = None
        cls._adapter_path = None
        cls._adapter_loaded = False
        cls._is_mock = False
        cls._device = None
        cls._quant_mode = None
        cls._load_duration = 0.0
