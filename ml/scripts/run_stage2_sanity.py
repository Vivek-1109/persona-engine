#!/usr/bin/env python3
"""
Persona Engine — Stage 2: 12-Example Sanity Training Experiment
Strictly adheres to:
1. Qwen2.5-1.5B-Instruct with native ChatML template.
2. Assistant-only loss masking (System & User tokens -> -100).
3. LoRA: r=16, alpha=32, dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"].
4. 72 optimizer steps (12 examples, batch=2, accum=1, 12 epochs).
5. Comprehensive evaluation across Categories A, B, C, D.
"""

import sys
import json
import logging
from pathlib import Path

# Add project root to sys.path
ml_root = Path(__file__).resolve().parents[1]
if str(ml_root) not in sys.path:
    sys.path.insert(0, str(ml_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Stage2Sanity")

SYSTEM_PROMPT = (
    "You are Vivek. Respond naturally in your authentic casual Hinglish style, "
    "slangs, humor, and short messaging as you chat with your close friend Naata."
)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def build_masked_dataset(jsonl_path, tokenizer, max_length=512):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    header_tokens = [151644, 77091, 198]  # <|im_start|>assistant\n
    im_end_id = 151645                   # <|im_end|>

    features = []
    for r in records:
        messages = r["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"]
        labels = [-100] * len(input_ids)

        # Locate final assistant turn
        last_assistant_start = -1
        for i in range(len(input_ids) - len(header_tokens), -1, -1):
            if input_ids[i : i + len(header_tokens)] == header_tokens:
                last_assistant_start = i + len(header_tokens)
                break

        if last_assistant_start != -1:
            end_idx = last_assistant_start
            while end_idx < len(input_ids) and input_ids[end_idx] != im_end_id:
                end_idx += 1
            if end_idx < len(input_ids):
                end_idx += 1  # Include <|im_end|>

            for j in range(last_assistant_start, end_idx):
                labels[j] = input_ids[j]

        features.append({
            "input_ids": input_ids,
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            "text": text,
        })

    return features


def main():
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset

    print("=" * 70)
    print("STAGE 2 — 12-EXAMPLE SANITY TRAINING EXPERIMENT")
    print("=" * 70)

    # 1. Hyperparameter & Step Math
    num_examples = 12
    batch_size = 2
    gradient_accumulation = 1
    steps_per_epoch = num_examples // (batch_size * gradient_accumulation)
    num_epochs = 12
    total_optimizer_steps = steps_per_epoch * num_epochs

    print("\nTRAINING STEP & BATCH CALCULATIONS:")
    print("-" * 50)
    print(f"  Training Examples          : {num_examples}")
    print(f"  Per-Device Batch Size      : {batch_size}")
    print(f"  Gradient Accumulation      : {gradient_accumulation}")
    print(f"  Effective Batch Size       : {batch_size * gradient_accumulation}")
    print(f"  Steps per Epoch            : {steps_per_epoch}")
    print(f"  Training Epochs            : {num_epochs}")
    print(f"  Total Optimizer Steps      : {total_optimizer_steps} (Target: 50-100 steps)")
    print("-" * 50)

    # 2. Tokenizer & Data Loading
    print(f"\n[1/5] Loading Tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_path = ml_root / "data/training/sanity_train.jsonl"
    val_path = ml_root / "data/training/sanity_val.jsonl"

    print(f"[2/5] Preparing datasets with assistant-only loss masking...")
    train_features = build_masked_dataset(train_path, tokenizer, max_length=512)
    val_features = build_masked_dataset(val_path, tokenizer, max_length=512)

    train_ds = Dataset.from_list([{k: v for k, v in f.items() if k != "text"} for f in train_features])
    val_ds = Dataset.from_list([{k: v for k, v in f.items() if k != "text"} for f in val_features])

    print(f"  Train dataset prepared: {len(train_ds)} examples")
    print(f"  Validation dataset    : {len(val_ds)} examples")

    # 3. Base Model & LoRA Initialization
    has_cuda = torch.cuda.is_available()
    print(f"\n[3/5] Loading Base Model on {'CUDA (Tesla T4)' if has_cuda else 'CPU'}...")

    if has_cuda:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        from peft import prepare_model_for_kbit_training
        base_model = prepare_model_for_kbit_training(base_model)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(base_model, lora_config)

    print("\nTRAINABLE PARAMETERS:")
    print("-" * 50)
    model.print_trainable_parameters()
    print("-" * 50)

    # 4. Training
    output_dir = ml_root / "models/adapters/stage2_sanity_adapter"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=4e-4,
        lr_scheduler_type="cosine",
        warmup_steps=6,
        logging_steps=6,
        eval_strategy="epoch",
        save_strategy="no",
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

    print(f"\n[4/5] Executing Stage 2 Training ({total_optimizer_steps} optimizer steps)...")
    train_result = trainer.train()

    print("\nSaving Stage 2 adapter...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    val_metrics = trainer.evaluate()
    final_train_loss = train_result.training_loss
    final_val_loss = val_metrics.get("eval_loss", 0.0)

    print("\n" + "=" * 70)
    print("STAGE 2 TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Final Training Loss   : {final_train_loss:.4f}")
    print(f"  Final Validation Loss : {final_val_loss:.4f}")
    print(f"  Total Optimizer Steps : {train_result.global_step}")
    print("=" * 70)

    # 5. Evaluation Suite
    print("\n[5/5] Evaluating Stage 2 Model (Categories A, B, C, D)...")

    def generate(messages):
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        device = model.device if hasattr(model, "device") else "cuda"
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=35,
                temperature=0.2,
                top_p=0.85,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        rep = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        for pfix in ["Vivek:", "vivek:", "Naata:", "naata:"]:
            if rep.startswith(pfix):
                rep = rep[len(pfix):].strip()
        return rep.split("\n")[0].strip()

    model.eval()

    # Category A: 3 SEEN Examples
    seen_tests = [
        (
            "SEEN-1",
            "Bhai katai chatai film thi, bass ek hi seekh mili kisi bhi chizz ko uski or apni aukaat se jyada naa chaaho. Jo hai usme khush raho 👤",
            "Bohot achii sikh mil gyi tujhe",
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Bhai katai chatai film thi, bass ek hi seekh mili kisi bhi chizz ko uski or apni aukaat se jyada naa chaaho. Jo hai usme khush raho 👤"}
            ]
        ),
        (
            "SEEN-2",
            "Chhod bhai mai waise bhi match ni dekhta jyada. Kya lagta hai kohli century maarega aaj?",
            "Jarur maarega",
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "assistant", "content": "Isiliye to tujhe bheja tha🤝👽"},
                {"role": "user", "content": "Chhod bhai mai waise bhi match ni dekhta jyada. Kya lagta hai kohli century maarega aaj?"}
            ]
        ),
        (
            "SEEN-3",
            "Bhai phone me download karke dekhe ya online?",
            "Yahi dekh bas👍🏻",
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Bhai phone me download karke dekhe ya online?"}
            ]
        )
    ]

    # Category B: 3 UNSEEN Prompts
    unseen_tests = [
        (
            "UNSEEN-1",
            "Bhai weekend pe kya plan hai?",
            None,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Bhai weekend pe kya plan hai?"}
            ]
        ),
        (
            "UNSEEN-2",
            "Bhai chai peene chalega kya?",
            None,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Bhai chai peene chalega kya?"}
            ]
        ),
        (
            "UNSEEN-3",
            "Bhai naya phone konsa lu 20k ke budget me?",
            None,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Bhai naya phone konsa lu 20k ke budget me?"}
            ]
        )
    ]

    # Category C: 2 MULTI-TURN Prompts
    multiturn_tests = [
        (
            "MULTI-TURN-1 (Validation)",
            "Context: [Vivek: '1 baje tak aa jaunga fir dentist pe jaana hai'] -> Naata: 'Msg maar diyo aake'",
            "Chalega kya?",
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "assistant", "content": "1 baje tak aa jaunga fir dentist pe jaana hai"},
                {"role": "user", "content": "Msg maar diyo aake"}
            ]
        ),
        (
            "MULTI-TURN-2 (Unseen)",
            "Context: [Naata: 'Aaj college aayega?'] -> Vivek: 'Haa 12 baje tak aa jaunga' -> Naata: 'To fir canteen me milte hai direct?'",
            None,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Aaj college aayega?"},
                {"role": "assistant", "content": "Haa 12 baje tak aa jaunga"},
                {"role": "user", "content": "To fir canteen me milte hai direct?"}
            ]
        )
    ]

    # Category D: 2 SINGLE-TURN Prompts
    singleturn_tests = [
        (
            "SINGLE-TURN-1 (Seen)",
            "game??",
            "Aaja room pe",
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "game??"}
            ]
        ),
        (
            "SINGLE-TURN-2 (Unseen)",
            "kaha hai?",
            None,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "kaha hai?"}
            ]
        )
    ]

    print("\n" + "=" * 70)
    print("STAGE 2 EVALUATION RESULTS")
    print("=" * 70)

    print("\n--- CATEGORY A: 3 SEEN EXAMPLES ---")
    for tag, prompt, expected, msgs in seen_tests:
        gen = generate(msgs)
        print(f"\n[{tag}]")
        print(f"  Prompt   : {prompt}")
        print(f"  Expected : {expected}")
        print(f"  Generated: {gen}")

    print("\n--- CATEGORY B: 3 UNSEEN PROMPTS ---")
    for tag, prompt, expected, msgs in unseen_tests:
        gen = generate(msgs)
        print(f"\n[{tag}]")
        print(f"  Prompt   : {prompt}")
        print(f"  Generated: {gen}")

    print("\n--- CATEGORY C: 2 MULTI-TURN PROMPTS ---")
    for tag, prompt, expected, msgs in multiturn_tests:
        gen = generate(msgs)
        print(f"\n[{tag}]")
        print(f"  Prompt   : {prompt}")
        if expected:
            print(f"  Expected : {expected}")
        print(f"  Generated: {gen}")

    print("\n--- CATEGORY D: 2 SINGLE-TURN PROMPTS ---")
    for tag, prompt, expected, msgs in singleturn_tests:
        gen = generate(msgs)
        print(f"\n[{tag}]")
        print(f"  Prompt   : {prompt}")
        if expected:
            print(f"  Expected : {expected}")
        print(f"  Generated: {gen}")

    print("\n" + "=" * 70)
    print("END OF STAGE 2 SANITY EXPERIMENT")
    print("=" * 70)


if __name__ == "__main__":
    main()
