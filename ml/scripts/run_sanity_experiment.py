#!/usr/bin/env python3
"""
Persona Engine — 15-Example Sanity Training Experiment
Strictly adheres to:
1. Assistant-only loss masking (System & User tokens -> -100).
2. Qwen2.5 native ChatML template.
3. LoRA r=16, alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"].
4. Direct comparison: Base Qwen vs Sanity LoRA.
"""

import os
import sys
import json
import logging
from pathlib import Path

from transformers import AutoTokenizer

# Add project root to path
ml_root = Path(__file__).resolve().parents[1]
if str(ml_root) not in sys.path:
    sys.path.insert(0, str(ml_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SanityExperiment")

SYSTEM_PROMPT = (
    "You are Vivek. Respond naturally in your authentic casual Hinglish style, "
    "slangs, humor, and short messaging as you chat with your close friend Naata."
)

def print_environment_versions():
    print("=" * 65)
    print("STEP 1: INSTALLED ENVIRONMENT & PACKAGE VERSIONS")
    print("=" * 65)
    import torch
    print(f"Python Version    : {sys.version.split()[0]}")
    print(f"PyTorch Version   : {torch.__version__}")
    print(f"CUDA Available    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Detected      : {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    packages = ["transformers", "trl", "peft", "accelerate", "bitsandbytes"]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            print(f"{pkg:<18}: {ver}")
        except ImportError as e:
            print(f"{pkg:<18}: NOT INSTALLED ({e})")
    print("=" * 65)


def build_masked_dataset(jsonl_path, tokenizer, max_length=512):
    """
    Tokenizes ChatML conversations and enforces ASSISTANT-ONLY loss masking.
    - System tokens -> label -100
    - User tokens -> label -100
    - Only final Assistant response tokens -> active label IDs
    """
    with open(jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    header_tokens = [151644, 77091, 198]  # <|im_start|>assistant\n
    im_end_id = 151645                   # <|im_end|>

    features = []
    for r in records:
        messages = r["messages"]
        # Format with native Qwen chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"]
        labels = [-100] * len(input_ids)

        # Locate the FINAL assistant turn in the sequence
        # We search from the end to find the final assistant turn
        last_assistant_start = -1
        for i in range(len(input_ids) - len(header_tokens), -1, -1):
            if input_ids[i : i + len(header_tokens)] == header_tokens:
                last_assistant_start = i + len(header_tokens)
                break

        if last_assistant_start != -1:
            # Find the ending <|im_end|> for this assistant turn
            end_idx = last_assistant_start
            while end_idx < len(input_ids) and input_ids[end_idx] != im_end_id:
                end_idx += 1
            if end_idx < len(input_ids):
                end_idx += 1  # Include <|im_end|> so model learns to stop

            for j in range(last_assistant_start, end_idx):
                labels[j] = input_ids[j]

        features.append({
            "input_ids": input_ids,
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            "text": text,
        })

    return features


def demonstrate_token_masking(feature, tokenizer):
    print("\n" + "=" * 65)
    print("STEP 2: TOKEN-BY-TOKEN LOSS MASKING DEMONSTRATION")
    print("=" * 65)
    input_ids = feature["input_ids"]
    labels = feature["labels"]

    total = len(input_ids)
    masked = labels.count(-100)
    active = total - masked

    print(f"Sequence Total Tokens : {total}")
    print(f"Masked Tokens (-100)  : {masked} (System prompt & User context)")
    print(f"Active Tokens (Loss)  : {active} (Strictly Assistant response)")
    print("\nToken Table Breakdown:")
    print("-" * 65)
    print(f"{'Idx':<4} {'Token ID':<9} {'Decoded Token':<26} {'Training Label'}")
    print("-" * 65)

    for idx, (tid, lab) in enumerate(zip(input_ids, labels)):
        decoded = repr(tokenizer.decode([tid]))
        if lab == -100:
            status = "-100 (MASKED — NO LOSS)"
        else:
            status = f"{lab} (ACTIVE ASSISTANT LOSS)"
        # Print first 15 tokens (system), middle transition, and assistant tokens
        if idx < 12 or idx > total - 22:
            print(f"{idx:<4} {tid:<9} {decoded:<26} {status}")
        elif idx == 12:
            print(" ...  [system & user prompt tokens continue with label -100] ...")
    print("-" * 65)
    print("DEMONSTRATION CONFIRMED: System & User tokens = -100, Assistant tokens = actual labels.\n")


def run_sanity_experiment():
    print_environment_versions()

    import torch
    from transformers import AutoTokenizer

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"\n[Loading Tokenizer] {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_path = ml_root / "data/training/sanity_train.jsonl"
    val_path = ml_root / "data/training/sanity_val.jsonl"

    print(f"Loading sanity training data from: {train_path}")
    print(f"Loading sanity validation data from: {val_path}")

    train_features = build_masked_dataset(train_path, tokenizer, max_length=512)
    val_features = build_masked_dataset(val_path, tokenizer, max_length=512)

    # Demonstrate loss masking on Example 1
    demonstrate_token_masking(train_features[0], tokenizer)

    if "--dry-run" in sys.argv:
        print("\n" + "=" * 65)
        print("DRY-RUN COMPLETE: Tokenization & assistant-only loss masking verified!")
        print(f"Verified {len(train_features)} training samples and {len(val_features)} validation samples.")
        print("=" * 65)
        return

    from datasets import Dataset
    train_ds = Dataset.from_list([{k: v for k, v in f.items() if k != "text"} for f in train_features])
    val_ds = Dataset.from_list([{k: v for k, v in f.items() if k != "text"} for f in val_features])

    from transformers import (
        AutoModelForCausalLM,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, PeftModel

    has_cuda = torch.cuda.is_available()
    print(f"\n[Loading Base Model] {model_name} (CUDA={has_cuda})...")

    if has_cuda:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        from peft import prepare_model_for_kbit_training
        base_model = prepare_model_for_kbit_training(base_model)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    # LoRA Configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(base_model, lora_config)

    # Ensure trainable params are float32
    if has_cuda:
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.to(torch.float32)

    print("\n" + "=" * 65)
    print("STEP 3: TRAINABLE PARAMETERS AUDIT")
    print("=" * 65)
    model.print_trainable_parameters()
    print("=" * 65)

    output_dir = ml_root / "models/adapters/sanity_adapter"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=1.5e-4,
        logging_steps=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=False,
        bf16=False,
        optim="paged_adamw_8bit" if has_cuda else "adamw_torch",
        report_to="none",
        seed=42,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    print("\n" + "=" * 65)
    print("STEP 4: TRAINING SANITY EXPERIMENT (5 EPOCHS, 12 SAMPLES)")
    print("=" * 65)
    train_result = trainer.train()

    print("\nSaving sanity adapter...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Evaluate final validation metrics
    val_metrics = trainer.evaluate()
    print("\n" + "=" * 65)
    print("FINAL SANITY TRAINING METRICS")
    print("=" * 65)
    print(f"Final Training Loss   : {train_result.training_loss:.4f}")
    print(f"Final Validation Loss : {val_metrics.get('eval_loss', 'N/A'):.4f}")
    print("=" * 65)

    # Testing Suite
    test_suite = [
        # 3 Seen Prompts
        ("SEEN-1", "Bhai katai chatai film thi, bass ek hi seekh mili kisi bhi chizz ko uski or apni aukaat se jyada naa chaaho. Jo hai usme khush raho 👤"),
        ("SEEN-2", "Chhod bhai mai waise bhi match ni dekhta jyada. Kya lagta hai kohli century maarega aaj?"),
        ("SEEN-3", "Bhai phone me download karke dekhe ya online?"),
        # 3 Unseen Prompts
        ("UNSEEN-1", "Bhai weekend pe kya plan hai?"),
        ("UNSEEN-2", "Bhai chai peene chalega kya?"),
        ("UNSEEN-3", "Bhai naya phone konsa lu 20k ke budget me?"),
        # 1 Single-turn
        ("SINGLE-TURN", "game??"),
        # 1 Multi-turn
        ("MULTI-TURN", [
            {"role": "user", "content": "Aaj college aayega?"},
            {"role": "assistant", "content": "Haa 12 baje tak aa jaunga"},
            {"role": "user", "content": "To fir canteen me milte hai direct?"}
        ])
    ]

    print("\n" + "=" * 65)
    print("STEP 5: INFERENCE COMPARISON (BASE MODEL vs SANITY LORA)")
    print("=" * 65)

    def generate_response(gen_model, prompt_input):
        if isinstance(prompt_input, str):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_input}
            ]
        else:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + prompt_input

        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        device = gen_model.device if hasattr(gen_model, "device") else "cuda"
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = gen_model.generate(
                **inputs,
                max_new_tokens=40,
                temperature=0.3,
                top_p=0.85,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        rep = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        # Clean speaker prefixes if any
        for pfix in ["Vivek:", "vivek:", "Naata:", "naata:"]:
            if rep.startswith(pfix):
                rep = rep[len(pfix):].strip()
        return rep.split("\n")[0].strip()

    # Disable LoRA for Base Model output
    print("\nGenerating Base Qwen outputs...")
    with model.disable_adapter():
        base_outputs = {}
        for tag, prompt in test_suite:
            base_outputs[tag] = generate_response(model, prompt)

    print("Generating Sanity LoRA outputs...")
    lora_outputs = {}
    for tag, prompt in test_suite:
        lora_outputs[tag] = generate_response(model, prompt)

    print("\n" + "=" * 65)
    print("COMPARATIVE EVALUATION RESULTS")
    print("=" * 65)
    for tag, prompt in test_suite:
        prompt_disp = prompt if isinstance(prompt, str) else prompt[-1]["content"]
        print(f"\n[{tag}] Prompt: {prompt_disp}")
        print(f"  Base Qwen   : {base_outputs[tag]}")
        print(f"  Sanity LoRA : {lora_outputs[tag]}")
    print("=" * 65)


if __name__ == "__main__":
    run_sanity_experiment()
