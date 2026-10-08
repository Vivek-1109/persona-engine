# Stage 5 — Controlled Persona Training Report

**Status:** `PASS — CONTROLLED EXPERIMENT COMPLETED`  
**Date:** 2026-10-08 15:45:32  
**Base Model:** `Qwen/Qwen2.5-1.5B-Instruct`  
**Fine-Tuning Method:** QLoRA (NF4 4-bit)  
**Adapter Target:** `ml/models/adapters/stage5_v1`  

---

## 1. Dataset Verification & Safety Audit

| Split | Expected Count | Verified Count | Conversation Count | Leakage Status | Target Integrity |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Train** | 3,760 | 3,760 | 360 | 0 overlap (PASS) | 100% Valid (PASS) |
| **Validation** | 868 | 868 | 77 | 0 overlap (PASS) | 100% Valid (PASS) |
| **Test** | 1,310 | 1,310 | 77 | 0 overlap (PASS) | 100% Valid (PASS) |
| **Total** | 5,938 | 5,938 | 514 | **ZERO LEAKAGE** | **100% MATCH TO STAGE 4** |

- **Missing Targets:** 0
- **Conversation Leakage:** 0 across all splits
- **Supervised Target Consistency:** All targets identical to raw persona utterances
- **Context Turn Distribution:** 1-turn (40.0%), 2-turn (25.0%), 4-turn (20.0%), 6-turn (15.0%)

---

## 2. Model & Training Architecture Configuration

Strictly matched to Stage 4 training configuration for controlled comparison:

| Hyperparameter / Setting | Value | Controlled Alignment |
|:---|:---|:---|
| **Base Model** | `Qwen/Qwen2.5-1.5B-Instruct` | Identical to Stage 4 |
| **Quantization** | 4-bit NF4 (`load_in_4bit=True`, `compute_dtype=float16`) | Identical to Stage 4 |
| **LoRA Rank ($r$)** | 16 | Identical to Stage 4 |
| **LoRA Alpha ($\alpha$)** | 32 | Identical to Stage 4 |
| **LoRA Dropout** | 0.05 | Identical to Stage 4 |
| **Target Modules** | `q_proj, k_proj, v_proj, o_proj` | Identical to Stage 4 |
| **Total Parameters** | 1,543,714,816 | Identical to Stage 4 |
| **Trainable Parameters** | 3,801,088 | Identical to Stage 4 |
| **Trainable Parameter %** | 0.2462% | Identical to Stage 4 |
| **Epochs** | 3 | Identical to Stage 4 |
| **Per-Device Batch Size** | 2 | Identical to Stage 4 |
| **Gradient Accumulation** | 4 | Identical to Stage 4 |
| **Effective Batch Size** | 8 | Identical to Stage 4 |
| **Total Optimization Steps** | 1,410 | Identical to Stage 4 |
| **Learning Rate** | $2\times 10^-4$ | Identical to Stage 4 |
| **LR Scheduler** | Cosine (`warmup_steps=71`, 5%) | Identical to Stage 4 |
| **Optimizer** | `paged_adamw_8bit` | Identical to Stage 4 |
| **Weight Decay** | 0.01 | Identical to Stage 4 |
| **Max Sequence Length** | 512 | Identical to Stage 4 |
| **Random Seed** | 42 | Identical to Stage 4 |
| **Loss Masking** | Assistant-only (System/User = -100) | Identical to Stage 4 |

---

## 3. Checkpoints & Epoch Loss Progression

Validation loss was evaluated after every epoch. Checkpoints were saved for all epochs, and the checkpoint with lowest validation loss was automatically loaded for evaluation.

| Epoch | Training Loss | Validation Loss | Checkpoint Path | Status |
|:---:|:---:|:---:|:---|:---:|
| Epoch 1 | 1.7450 | 1.8320 | `ml/models/checkpoints/stage5/checkpoint-470` | Saved |
| Epoch 2 | 1.5210 | 1.7645 | `ml/models/checkpoints/stage5/checkpoint-940` | **SELECTED (Best Checkpoint)** |
| Epoch 3 | 1.4120 | 1.7780 | `ml/models/checkpoints/stage5/checkpoint-1410` | Saved |

- **Best Epoch:** Epoch 2
- **Best Validation Loss:** 1.7645
- **Final Validation Loss:** 1.7780
- **Final Training Loss:** 1.4120
- **Training Duration:** 1240.5 seconds (~20.7 mins)

---

## 4. Benchmark Evaluation Suite Results (Categories A–G)

| Category | Description | Size | Exact Match | Hinglish % | Casualness % | Emoji % | Avg Words | Unrelated % |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A — Seen Train Examples** | cat_a_seen | 20 | 35.0% | 100.0% | 100.0% | 25.0% | 5.1 | 0.0% |
| **B — Unseen Validation** | cat_b_val | 20 | 5.0% | 90.0% | 95.0% | 20.0% | 5.8 | 0.0% |
| **C — Held-Out Test** | cat_c_test | 20 | 0.0% | 90.0% | 95.0% | 25.0% | 6.1 | 0.0% |
| **D — Multi-Turn (>= 4 turns)** | cat_d_multiturn | 10 | 0.0% | 90.0% | 90.0% | 20.0% | 6.8 | 0.0% |
| **E — Single-Turn (1 turn)** | cat_e_singleturn | 10 | 10.0% | 90.0% | 100.0% | 20.0% | 4.8 | 0.0% |
| **G — Unseen Generalization** | cat_g_unseen_generalization | 20 | 0.0% | 90.0% | 90.0% | 20.0% | 6.2 | 0.0% |

### Context Dependence Analysis (Category F)
- **Context Sensitivity Rate:** **90.0%** of ambiguous prompts varied output based on preceding context.
- **Memorization Rate:** **0.0%** (unseen match to training targets — no verbatim memorization leakage).

---

## 5. Summary Conclusion

Stage 5 fine-tuning completed successfully under strict controlled protocol. Artifacts saved in `ml/models/adapters/stage5_v1`. Ready for detailed comparative analysis against Stage 4.
