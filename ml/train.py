"""
Persona Engine — Standalone Training Script for Google Colab / GPU
Executes 4-bit QLoRA fine-tuning on the persona dataset and saves the adapter checkpoint.

Usage:
    python train.py
    python train.py --epochs 3 --batch-size 2 --lr 2e-4
"""

import argparse
import sys
from pathlib import Path

# Add ml root to path
ml_root = Path(__file__).resolve().parent
if str(ml_root) not in sys.path:
    sys.path.insert(0, str(ml_root))

from src.training import PersonaTrainer, TrainingConfig


def main():
    parser = argparse.ArgumentParser(description="Train Persona Engine QLoRA Adapter")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model name",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to training data JSONL (defaults to train.jsonl)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=64,
        help="LoRA rank dimension",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/adapters/vivek_adapter",
        help="Output adapter directory",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PERSONA ENGINE — QLoRA FINE-TUNING")
    print("=" * 60)
    print(f"Base Model    : {args.model}")
    print(f"Epochs        : {args.epochs}")
    print(f"Batch Size    : {args.batch_size} (accum: {args.grad_accum})")
    print(f"Learning Rate : {args.lr}")
    print(f"LoRA Rank (r) : {args.lora_r}")
    print(f"Output Path   : {args.output_dir}")
    print("=" * 60)

    # Resolve data path
    data_path = None
    if args.data:
        data_path = Path(args.data)
    else:
        candidates = [
            ml_root / "data" / "training" / "train.jsonl",
            ml_root / "data" / "sample" / "sample_conversations.jsonl",
        ]
        for c in candidates:
            if c.exists():
                data_path = c
                break

    if not data_path or not data_path.exists():
        print(f"ERROR: No dataset found at {data_path}")
        sys.exit(1)

    print(f"Using Dataset : {data_path}")

    config = TrainingConfig(
        base_model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        output_dir=args.output_dir,
    )

    trainer = PersonaTrainer(config=config)
    print("\n[1/3] Preparing dataset...")
    status = trainer.prepare_dataset(data_path)
    print(f"  Loaded {status['num_examples']} training examples.")

    print("\n[2/3] Loading base model with 4-bit quantization and LoRA...")
    trainer.load_model_and_tokenizer()

    print("\n[3/3] Training model...")
    result = trainer.train(output_dir=args.output_dir)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Status      : {result.get('status')}")
    print(f"Global Steps: {result.get('global_step')}")
    print(f"Final Loss  : {result.get('train_loss', 'N/A')}")
    print(f"Saved to    : {result.get('output_dir')}")
    print("=" * 60)
    print(f"\nNext: Test your model by running:")
    print(f"  python chat.py --adapter {args.output_dir}")


if __name__ == "__main__":
    main()
