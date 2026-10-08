# Persona Engine — ML Laboratory & Foundation

Welcome to the **Persona Engine ML Foundation**.

This workspace provides the complete machine learning architecture for learning and reproducing the communication behavior, conversational style, humor, sarcasm, Hinglish mixing, and personality patterns of a specific person from conversational data.

---

## 1. Core Architecture & Philosophy

The Persona Engine ML workspace is designed specifically around a **hybrid local + Google Colab workflow**:

```text
LOCAL DEVELOPMENT (Antigravity IDE / Local Machine)
  - Source code development (ml/src/)
  - Dataset schemas & configuration (ml/schemas/, ml/configs/)
  - Preprocessing, validation & cleaning logic
  - Lightweight tests & verification scripts
  - Version control (Git)
         │
         ▼ (git push / pull)
GITHUB (Source of Truth for Code)
         │
         ▼ (git clone / pull)
GOOGLE COLAB (Primary GPU Training Laboratory)
         │
         ├─► 00_colab_setup.ipynb        (Environment bootstrap & GPU verification)
         ├─► 01_dataset_analysis.ipynb   (Exploratory data analysis & style metrics)
         ├─► 02_preprocessing.ipynb      (Cleaning, splitting, ChatML generation)
         ├─► 03_annotation.ipynb         (Behavioral signal annotation)
         ├─► 04_baseline_inference.ipynb (Zero-shot baseline benchmark)
         ├─► 05_training.ipynb           (QLoRA PEFT fine-tuning)
         └─► 06_evaluation.ipynb         (Persona alignment evaluation)
         │
         ▼
ARTIFACTS / CHECKPOINTS
         ├─► Google Drive (Persistent backup)
         └─► Hugging Face Hub (Private model repository)
         │
         ▼
EVALUATION RESULTS (experiments/exp_xxx/)
         │
         ▼
NEXT EXPERIMENT ITERATION
```

### The Colab Principle: Thin Notebooks, Reusable Modules
- **Notebooks do not contain business/ML logic**. Notebooks are orchestrators that install dependencies, import modules from `ml/src/`, run experiments, and display results.
- **`ml/src/` is the single source of truth** for all reusable code. The exact same code runs on your local CPU machine, Google Colab GPU, or any cloud VM.

---

## 2. Local vs. Google Colab Responsibilities

| Responsibility | Local Machine | Google Colab |
|----------------|:-------------:|:------------:|
| Code Authoring & Refactoring | ✅ | ❌ |
| JSON Schemas & Config Design | ✅ | ❌ |
| Lightweight Testing & Linting | ✅ | ❌ |
| Version Control (Git) | ✅ | ❌ (read-only pull) |
| Heavy GPU Acceleration (CUDA) | ❌ | ✅ |
| Foundation Model Weights Download | ❌ | ✅ |
| 4-bit Quantization (bitsandbytes) | ❌ | ✅ |
| QLoRA / PEFT Fine-Tuning | ❌ | ✅ |
| Embedding & Semantic Similarity | ❌ | ✅ |
| Checkpoint Storage (Google Drive) | ❌ | ✅ |

---

## 3. Directory Layout

```text
ml/
├── configs/                       # Central YAML configurations
│   ├── base.yaml                  # Core configuration (paths, model defaults, seed)
│   ├── data.yaml                  # Dataset rules, speakers, splitting ratios
│   ├── training.yaml              # QLoRA hyperparameters & quantization flags
│   └── evaluation.yaml            # Evaluation metrics and generation parameters
│
├── schemas/                       # Draft-07 JSON Schemas
│   ├── conversation.schema.json   # Multi-turn conversation standard schema
│   ├── annotation.schema.json     # Behavioral signal tagging schema
│   └── training_example.schema.json # ChatML training example schema
│
├── src/                           # Reusable ML Python Package
│   ├── config/loader.py           # YAML config loader with environment overrides
│   ├── data/                      # Loader, validator, cleaner, and splitter
│   ├── preprocessing/             # Unicode normalizer & ChatML conversation builder
│   ├── annotation/schema.py       # Behavioral annotation models & validator
│   ├── analysis/dataset_analyzer.py # Style signatures, length, emoji, Hinglish analyzer
│   ├── training/trainer.py        # PersonaTrainer interface (future QLoRA)
│   ├── evaluation/                # Style metrics & PersonaEvaluator
│   └── inference/generator.py     # Model-agnostic PersonaGenerator
│
├── data/                          # 5-Stage Dataset Architecture
│   ├── raw/                       # Untouched private exports (gitignored)
│   ├── processed/                 # Cleaned & validated JSONL (gitignored)
│   ├── annotated/                 # Tagged with emotion, humor, Hinglish (gitignored)
│   ├── training/                  # Model-ready chat formatted JSONL (gitignored)
│   ├── evaluation/                # Held-out benchmark test sets (gitignored)
│   └── sample/                    # Safe synthetic demo data (committed)
│       └── sample_conversations.jsonl
│
├── notebooks/                     # Thin Google Colab / Jupyter Notebooks
│   ├── 00_colab_setup.ipynb       # Environment bootstrap & diagnostics
│   ├── 01_dataset_analysis.ipynb  # Dataset metrics & EDA
│   ├── 02_preprocessing.ipynb     # Cleaning, validation, and splitting
│   ├── 03_annotation.ipynb        # Behavioral annotation exploration
│   ├── 04_baseline_inference.ipynb# Zero-shot baseline evaluation
│   ├── 05_training.ipynb          # QLoRA fine-tuning walkthrough
│   └── 06_evaluation.ipynb        # Style alignment evaluation
│
├── experiments/                   # Experiment tracking directories (metrics, notes)
│   └── README.md
│
├── models/                        # Model weights storage mount points
│   ├── base/                      # Foundation model cache (gitignored)
│   ├── adapters/                  # LoRA adapter weights (gitignored)
│   ├── checkpoints/               # Intermediate training checkpoints (gitignored)
│   └── README.md
│
├── scripts/                       # CLI automation scripts
│   ├── validate_dataset.py        # Validate JSONL files against schema
│   ├── analyze_dataset.py         # Generate style & lexical analysis reports
│   ├── preprocess_dataset.py      # Clean, split, and format datasets
│   └── evaluate_model.py          # Benchmark model responses against reference
│
├── requirements.txt               # Local CPU development dependencies
├── requirements-colab.txt         # Colab GPU dependencies (torch, transformers, peft)
├── .gitignore                     # Protects private data and large model weights
└── README.md                      # This documentation
```

---

## 4. Five-Stage Dataset Lifecycle

1. **`data/raw/`**: Unprocessed chat exports. Read-only.
2. **`data/processed/`**: Cleaned JSONL files. Personality traits (emojis, slang, Hinglish, expressive punctuation) are **strictly preserved**.
3. **`data/annotated/`**: Enriched with behavioral signals (emotion, tone, humor, sarcasm, teasing, language mixing, response strategy).
4. **`data/training/`**: Formatted into ChatML instruction pairs for causal LM fine-tuning (`train.jsonl`, `val.jsonl`).
5. **`data/evaluation/`**: Held-out benchmark dialogues for quantitative evaluation.

---

## 5. How to Run Locally

### 5.1 Environment Setup
```bash
cd "d:/Freelance Projects/Persona Engine/ml"
python -m pip install -r requirements.txt
```

### 5.2 Validate Sample Dataset
```bash
python scripts/validate_dataset.py --input data/sample/sample_conversations.jsonl --verbose
```

### 5.3 Analyze Style & Behavioral Metrics
```bash
python scripts/analyze_dataset.py --input data/sample/sample_conversations.jsonl
```

### 5.4 Preprocess and Generate Training Pairs
```bash
python scripts/preprocess_dataset.py --input data/sample/sample_conversations.jsonl --build-training
```

### 5.5 Benchmark Model Responses
```bash
python scripts/evaluate_model.py --eval-data data/sample/sample_conversations.jsonl
```

---

## 6. How to Run on Google Colab

1. **Open Google Colab**:
   Navigate to [Google Colab](https://colab.research.google.com/) and open `00_colab_setup.ipynb`.
2. **Select GPU Hardware Accelerator**:
   Go to `Runtime` -> `Change runtime type` -> select `T4 GPU` (free tier) or `A100 GPU` (Pro).
3. **Clone the Repository**:
   ```bash
   !git clone https://github.com/<your-username>/persona-engine.git /content/persona-engine
   %cd /content/persona-engine/ml
   ```
4. **Install Dependencies**:
   ```bash
   !pip install -r requirements-colab.txt
   ```
5. **Run Notebooks in Sequence**:
   - `00_colab_setup.ipynb` -> Verify environment & GPU
   - `01_dataset_analysis.ipynb` -> Analyze target conversation dynamics
   - `02_preprocessing.ipynb` -> Clean and build ChatML training examples
   - `05_training.ipynb` -> Run QLoRA training
   - `06_evaluation.ipynb` -> Evaluate adapter alignment

---

## 7. Model Storage & Remote Checkpoints

- **Never commit weights to Git**: `*.safetensors`, `*.bin`, `*.pt` are strictly gitignored.
- **Remote Mount**: In Google Colab, mount Google Drive:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```
- Save adapters directly to `/content/drive/MyDrive/persona-engine/models/adapters/` or push to a private Hugging Face Hub repository using `huggingface_hub`.

---

## 8. What is Intentionally NOT Implemented Yet

In keeping with Phase 1 / ML Foundation principles:
- ❌ **No expensive GPU fine-tuning**: Actual QLoRA execution runs in Phase 4.
- ❌ **No large model downloads**: Keeps the local repo lightweight and avoids accidental bandwidth/storage exhaustion.
- ❌ **No commercial LLM dependencies**: Core persona architecture does not depend on closed APIs (OpenAI / Anthropic / Gemini).
- ❌ **No complex vector databases or RAG**: Vector stores belong to the Memory Engine in Phase 5.

---

## 9. Stage 5 — Context + Behavioral Representation Pipeline

Stage 5 introduces dataset engineering for rich **Context and Behavioral Representation** without modifying the frozen Stage 4 v1 baseline.

### 9.1 Why Behavioral Representation is Being Introduced
In conversational persona modeling, surface text fine-tuning alone can fail to capture how human responses vary depending on subtle conversational dynamics, mood, relation, and situational context. Real WhatsApp conversations reveal that the exact same prompt (e.g. *"Aaja"*, *"Khelega"*, *"Thik"*) evokes radically different persona responses depending on whether the exchange is playful banter, urgent logistics, academic coordination, or late-night gaming.

By annotating every example with explicit behavioral, topic, and context metadata:
1. **Disambiguation**: The system can distinguish between context-dependent personas (e.g. terse replies vs. descriptive explanations).
2. **Controlled Conditioning**: Enables conditional steering, style-conditioned evaluation, and granular validation of persona fidelity across diverse interaction styles.
3. **Auditing & Stratified Analysis**: Enables tracking how well the persona model learns specific tones (teasing, humor, seriousness) and topics (gaming, college, plans).

### 9.2 Extracted Features
The Stage 5 pipeline extracts metadata across three distinct domains deterministically without external API or LLM dependencies:

1. **Behavioral Features** (`ml/src/preprocessing/behavioral_features.py`):
   - `language`: `hinglish`, `english`, `hindi` (Devanagari), `mixed`, or `unknown`.
   - `tone`: `casual`, `humorous`, `teasing`, `uncertain`, `supportive`, `serious`, or `neutral`.
   - `response_type`: `statement`, `answer`, `question`, `acknowledgement`, `suggestion`, `refusal`, `invitation`, `reaction`, or `greeting`.
   - `response_length`: Character count, word count, and category (`very_short`, `short`, `medium`, `long`).
   - `emoji_count` & `has_emoji`: Unicode emoji detection and frequency counts.
   - `has_slang`: Detection of colloquial Romanized Hindi/Hinglish slang markers (e.g., *bc*, *bsdk*, *jugaad*, *mast*, *katai chatai*).
   - `is_question`: Boolean indicator for interrogatives and questioning structures.

2. **Topic Features** (`ml/src/preprocessing/topic_classifier.py`):
   - Deterministic keyword and pattern classifier spanning 9 conversational categories:
     - `gaming` (BGMI, Valorant, lobby, clutch, ping)
     - `college` (classes, exams, attendance, dean, hostel, assignment)
     - `movies` (cinema, trailer, Netflix, series, Bollywood)
     - `technology` (laptop, battery, code, Windows, charging, phone)
     - `plans` (meeting, travel, station, schedule, timings)
     - `casual_chat` (daily check-ins, routine chat)
     - `sports` (cricket, Kohli, match, football)
     - `social` (birthdays, parties, meetups)
     - `other` (unclassified or general discourse)
   - Evaluates target response with context awareness fallback for terse utterances.

3. **Context Dynamics** (`ml/src/preprocessing/context_features.py`):
   - `context_depth`: Total messages in dialogue context window (1, 3, 7, 11).
   - `number_of_turns`: Number of dialogue turn exchanges.
   - `speaker_alternation_rate` & `is_strict_alternation`: Rate of speaker alternation between user and persona.
   - `speaker_sequence`: Chronological list of message authors in the context.
   - `last_user_message`: Exact text of the preceding user utterance.
   - `conversation_state`: Inferred dialogue state (`opening`, `inquiry`, `agreement`, `banter`, `closing`, or `ongoing`).

### 9.3 Why Raw Data Remains Immutable
The raw WhatsApp exports (`ml/data/raw/`) and clean Stage 4 datasets (`stage4_train.jsonl`, `val.jsonl`, `test.jsonl`) remain strictly read-only and immutable:
- **Reproducibility**: Guarantees that baseline Stage 4 experiments remain 100% reproducible with bit-for-bit identical inputs.
- **Traceability**: Every Stage 5 annotated record preserves the original `conversation_id`, `target_message_id`, `context`, and exact `target_response`.
- **Zero Leakage**: Strict conversation-level boundaries established in Stage 3/4 are carried forward with 0 cross-split leakage.

### 9.4 Why Training is Intentionally NOT Performed Yet
Model fine-tuning is intentionally postponed during Stage 5 dataset engineering because:
- **Audit-First Philosophy**: Dataset representations, distributions, and ambiguous prompt benchmarks must be verified and audited for structural integrity before allocating GPU compute.
- **Evaluation Baseline Integrity**: Stage 4 v1 serves as the reference baseline. Moving immediately to training without a documented dataset audit and benchmark definitions would compromise experimental isolation.
- **Conditioning Architecture Verification**: Once the behavioral representations are verified, subsequent modeling phases can determine the optimal conditioning mechanism (e.g. prompt prefixes, special tokens, or steering vectors).

### 9.5 How to Run the Stage 5 Pipeline & Audit
To rebuild the Stage 5 datasets:
```bash
python ml/scripts/build_stage5_dataset.py
```

To re-run the full Stage 5 dataset audit and generate the report:
```bash
python ml/scripts/audit_stage5_dataset.py
```

