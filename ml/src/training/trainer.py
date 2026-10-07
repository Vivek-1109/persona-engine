"""
Persona Engine — Persona Trainer
Implements QLoRA / PEFT fine-tuning for open-weight foundation models (e.g. Qwen 2.5, Llama 3)
on conversational persona datasets. Designed for Google Colab GPU runtimes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Hyperparameters and configuration for QLoRA fine-tuning."""

    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    method: str = "qlora"  # "qlora" | "lora" | "full"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 1.5e-4
    num_epochs: int = 4
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 512
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]
    )
    output_dir: str = "models/adapters"
    seed: int = 42
    extra_params: Dict[str, Any] = field(default_factory=dict)


class PersonaTrainer:
    """
    Orchestrates QLoRA / PEFT fine-tuning on persona-specific datasets.
    Decoupled from notebooks: loads quantized base model, injects LoRA adapters,
    formats training sequences, executes training, and exports adapter checkpoints.
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.is_prepared: bool = False
        self.train_texts: List[str] = []
        self._model = None
        self._tokenizer = None

    def load_model_and_tokenizer(self):
        """
        Loads the tokenizer and base model.
        In QLoRA mode on CUDA, applies 4-bit NormalFloat4 (NF4) quantization
        and attaches PEFT LoRA adapters.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading tokenizer: {self.config.base_model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name,
            trust_remote_code=True,
            padding_side="right",
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        logger.info(
            f"Loading base model: {self.config.base_model_name} (method: {self.config.method})"
        )

        has_cuda = torch.cuda.is_available()

        if self.config.method == "qlora" and has_cuda:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

            base_model = prepare_model_for_kbit_training(base_model)

            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=self.config.target_modules,
            )

            self._model = get_peft_model(base_model, lora_config)

            # Ensure all trainable LoRA parameters are float32 for stable optimization
            for p in self._model.parameters():
                if p.requires_grad:
                    p.data = p.data.to(torch.float32)

            self._model.print_trainable_parameters()

        elif has_cuda:
            from peft import LoraConfig, get_peft_model

            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=self.config.target_modules,
            )

            self._model = get_peft_model(base_model, lora_config)

        else:
            # CPU execution (local tests / dry-runs)
            logger.warning("CUDA not detected. Loading model in float32 on CPU.")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )

        return self._model, self._tokenizer

    def prepare_dataset(
        self,
        train_data_path: Union[str, Path],
        val_data_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Loads dataset records from a JSONL file and prepares ChatML text sequences.
        Handles both prompt-response pairs, message lists, and raw text examples.
        """
        from ..data.loader import load_jsonl, normalize_conversation

        train_path = Path(train_data_path)
        if not train_path.exists():
            raise FileNotFoundError(f"Training dataset not found: {train_path}")

        raw_records = load_jsonl(train_path)
        texts: List[str] = []

        for r in raw_records:
            if "text" in r and isinstance(r["text"], str) and r["text"].strip():
                texts.append(r["text"].strip())
            elif "messages" in r and isinstance(r["messages"], list):
                # Standard ChatML or SFT structure
                msgs = list(r["messages"])
                if "target" in r and r["target"]:
                    msgs.append({"role": "assistant", "content": str(r["target"])})

                parts = []
                for m in msgs:
                    role = m.get("role") or m.get("speaker") or "user"
                    content = m.get("content") or m.get("text") or ""
                    role_mapped = "assistant" if str(role).lower() in {"assistant", "persona", "vivek"} else ("system" if str(role).lower() == "system" else "user")
                    parts.append(f"<|im_start|>{role_mapped}\n{content}<|im_end|>\n")
                full_seq = "".join(parts).strip()
                if full_seq:
                    texts.append(full_seq)

            elif "context" in r and "response" in r:
                norm = normalize_conversation(r)
                parts = []
                persona_name = str(norm.get("persona_id", "persona")).lower()
                for m in norm.get("messages", []):
                    spk = str(m.get("speaker", "user")).lower()
                    role_mapped = "assistant" if spk in {"persona", "assistant", persona_name} else "user"
                    parts.append(f"<|im_start|>{role_mapped}\n{m.get('text', '')}<|im_end|>\n")
                full_seq = "".join(parts).strip()
                if full_seq:
                    texts.append(full_seq)

        self.train_texts = texts
        self.is_prepared = True
        logger.info(f"PersonaTrainer: Prepared {len(texts)} training sequences from {train_path}")

        return {
            "status": "READY",
            "num_examples": len(texts),
            "train_path": str(train_path),
            "val_path": str(val_data_path) if val_data_path else None,
        }

    def train(
        self,
        train_data_path: Optional[Union[str, Path]] = None,
        val_data_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes fine-tuning.
        If dry_run is True, verifies readiness without initiating weight updates.
        """
        out_dir = Path(output_dir or self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if not self.train_texts:
            data_path = (
                train_data_path
                or Path("data/training/train.jsonl")
            )
            self.prepare_dataset(data_path, val_data_path=val_data_path)

        if dry_run:
            logger.info("Dry-run requested: skipping weight training.")
            return {
                "status": "DRY_RUN_PASSED",
                "num_examples": len(self.train_texts),
                "output_dir": str(out_dir),
            }

        if self._model is None or self._tokenizer is None:
            self.load_model_and_tokenizer()

        import torch
        from datasets import Dataset

        dataset = Dataset.from_dict({"text": self.train_texts})
        logger.info(f"Starting QLoRA training with {len(dataset)} examples...")

        try:
            import inspect
            from trl import SFTConfig, SFTTrainer

            sft_sig = inspect.signature(SFTConfig.__init__).parameters
            config_kwargs: Dict[str, Any] = {
                "output_dir": str(out_dir),
                "num_train_epochs": self.config.num_epochs,
                "per_device_train_batch_size": self.config.batch_size,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "learning_rate": self.config.learning_rate,
                "logging_steps": 5,
                "save_strategy": "epoch",
                "fp16": False,
                "bf16": False,
                "optim": "paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
                "report_to": "none",
                "seed": self.config.seed,
            }

            # Compatibility across TRL versions (max_length vs max_seq_length)
            if "max_length" in sft_sig:
                config_kwargs["max_length"] = self.config.max_seq_length
            elif "max_seq_length" in sft_sig:
                config_kwargs["max_seq_length"] = self.config.max_seq_length

            if "dataset_text_field" in sft_sig:
                config_kwargs["dataset_text_field"] = "text"

            training_args = SFTConfig(**config_kwargs)

            trainer_sig = inspect.signature(SFTTrainer.__init__).parameters
            trainer_kwargs: Dict[str, Any] = {
                "model": self._model,
                "train_dataset": dataset,
                "args": training_args,
            }

            # Compatibility for tokenizer vs processing_class
            if "processing_class" in trainer_sig:
                trainer_kwargs["processing_class"] = self._tokenizer
            elif "tokenizer" in trainer_sig:
                trainer_kwargs["tokenizer"] = self._tokenizer

            # Mask prompt tokens so loss is strictly computed on assistant target completion
            try:
                from trl import DataCollatorForCompletionOnlyLM

                response_template = "<|im_start|>assistant\n"
                response_template_ids = self._tokenizer.encode(
                    response_template, add_special_tokens=False
                )
                collator = DataCollatorForCompletionOnlyLM(
                    response_template=response_template_ids,
                    tokenizer=self._tokenizer,
                )
                trainer_kwargs["data_collator"] = collator
                logger.info("PersonaTrainer: DataCollatorForCompletionOnlyLM enabled (loss masked for prompt tokens).")
            except Exception as collator_err:
                logger.warning(f"Could not initialize DataCollatorForCompletionOnlyLM: {collator_err}")

            trainer = SFTTrainer(**trainer_kwargs)

            train_result = trainer.train()
            self.save_adapter(out_dir)

            return {
                "status": "COMPLETED",
                "global_step": train_result.global_step,
                "train_loss": train_result.training_loss,
                "output_dir": str(out_dir),
            }

        except Exception as trl_err:
            logger.warning(
                f"TRL SFTTrainer initialization failed ({trl_err}). "
                f"Falling back to standard Hugging Face Trainer."
            )
            # Fallback to standard Hugging Face Trainer with completion-only masking
            from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

            resp_ids = self._tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

            def tokenize_func(examples):
                tokens = self._tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=self.config.max_seq_length,
                )
                labels = []
                for input_ids in tokens["input_ids"]:
                    label_ids = list(input_ids)
                    resp_start = -1
                    for i in range(len(input_ids) - len(resp_ids) + 1):
                        if input_ids[i : i + len(resp_ids)] == resp_ids:
                            resp_start = i + len(resp_ids)
                            break
                    if resp_start != -1:
                        # Mask all user/system prompt tokens with -100
                        for j in range(resp_start):
                            label_ids[j] = -100
                    labels.append(label_ids)
                tokens["labels"] = labels
                return tokens

            tokenized_dataset = dataset.map(tokenize_func, batched=True, remove_columns=["text"])

            training_args = TrainingArguments(
                output_dir=str(out_dir),
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                logging_steps=5,
                save_strategy="epoch",
                fp16=False,
                bf16=False,
                optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
                report_to="none",
                seed=self.config.seed,
            )

            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self._tokenizer,
                pad_to_multiple_of=8,
                return_tensors="pt",
            )

            trainer = Trainer(
                model=self._model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
            )

            train_result = trainer.train()
            self.save_adapter(out_dir)

            return {
                "status": "COMPLETED",
                "global_step": train_result.global_step,
                "train_loss": train_result.training_loss,
                "output_dir": str(out_dir),
            }

    def save_adapter(self, output_path: Union[str, Path]) -> Path:
        """Saves trained LoRA adapter weights and tokenizer configuration."""
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)
        if self._model is not None:
            self._model.save_pretrained(str(out))
        if self._tokenizer is not None:
            self._tokenizer.save_pretrained(str(out))
        logger.info(f"LoRA adapter and tokenizer successfully saved to {out}")
        return out
