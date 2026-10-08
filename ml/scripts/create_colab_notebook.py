import json
from pathlib import Path

notebook_content = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Persona Engine — Stage 4: First Real Persona Fine-Tuning Experiment\n",
                "\n",
                "**Environment:** Google Colab GPU (Tesla T4 / A100)\n",
                "**Model:** `Qwen/Qwen2.5-1.5B-Instruct`\n",
                "**Dataset:** `ml/data/training/stage4_train.jsonl` (3,760 examples)\n",
                "**LoRA Config:** $r=16, \\alpha=32, \\text{dropout}=0.05$, target modules: `[q_proj, k_proj, v_proj, o_proj]`\n",
                "**Loss:** Assistant-only loss masking (System & User tokens $\\rightarrow -100$, Assistant target tokens active)\n",
                "**Hyperparameters:** $\\text{lr}=2\\times 10^{-4}$, $\\text{batch}=2$, $\\text{accum}=4$ (effective batch 8), $\\text{epochs}=3$, cosine schedule\n",
                "\n",
                "---\n",
                "### Pipeline Workflow\n",
                "1. Hardware verification & package installation (`transformers`, `peft`, `bitsandbytes`, `accelerate`, `datasets`)\n",
                "2. Dataset integrity & schema validation check\n",
                "3. Assistant loss masking token inspection on 3 examples\n",
                "4. Pre-training Base Model evaluation across Benchmark Categories A through G\n",
                "5. 3-Epoch QLoRA fine-tuning with validation loss evaluation after every epoch\n",
                "6. Best checkpoint selection & adapter export to `ml/models/adapters/stage4_v1/`\n",
                "7. Post-training Stage-4 Adapter evaluation (both deterministic $T=0.0$ and sampling $T=0.7$)\n",
                "8. Quantitative & qualitative persona alignment reporting"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Hardware & GPU Diagnostics"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!nvidia-smi\n",
                "import torch\n",
                "print('PyTorch Version :', torch.__version__)\n",
                "print('CUDA Available  :', torch.cuda.is_available())\n",
                "if torch.cuda.is_available():\n",
                "    print('GPU Device Name :', torch.cuda.get_device_name(0))\n",
                "    print('Total VRAM (GB) :', round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))\n",
                "else:\n",
                "    print('WARNING: Running on CPU! Enable GPU runtime via Runtime -> Change runtime type -> T4 GPU.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Environment Setup & Dependency Installation"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Install requirements for GPU fine-tuning\n",
                "!pip install -q transformers>=4.44 datasets>=3.0 accelerate>=1.0 peft>=0.13 bitsandbytes>=0.43"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Repository Directory Verification"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "from pathlib import Path\n",
                "\n",
                "# If persona-engine is cloned into /content/persona-engine, change directory\n",
                "if Path('/content/persona-engine').is_dir():\n",
                "    os.chdir('/content/persona-engine')\n",
                "\n",
                "print('Current Working Directory:', Path.cwd())\n",
                "assert (Path.cwd() / 'ml/scripts/run_stage4_experiment.py').is_file(), 'Error: ml/scripts/run_stage4_experiment.py not found!'"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Run Dataset Validation Check"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python ml/scripts/validate_persona_dataset.py --train-file stage4_train.jsonl"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Execute Stage-4 Fine-Tuning & Benchmark Evaluation Pipeline"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Run full Stage 4 experiment: Base Evaluation -> 3-Epoch Training -> Adapter Evaluation -> Reports\n",
                "!python ml/scripts/run_stage4_experiment.py --epochs 3 --batch-size 2 --accum-steps 4 --lr 2e-4"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Inspect Training History & Overfitting Metrics"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import json\n",
                "with open('ml/experiments/stage4_v1/training_history.json', 'r', encoding='utf-8') as f:\n",
                "    history = json.load(f)\n",
                "\n",
                "print(f\"Total Training Duration: {history['train_duration_seconds']} seconds\")\n",
                "print(\"Epoch History:\")\n",
                "for ep in history['epoch_eval_history']:\n",
                "    print(f\"  Epoch {ep['epoch']}: Train Loss = {ep.get('train_loss')}, Val Loss = {ep.get('eval_loss')}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Render Evaluation Report"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from IPython.display import Markdown, display\n",
                "with open('ml/experiments/stage4_v1/evaluation_report.md', 'r', encoding='utf-8') as f:\n",
                "    display(Markdown(f.read()))"
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "T4",
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

output_path = Path("ml/notebooks/09_stage4_training_colab.ipynb")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=1)

print(f"Created Colab notebook at: {output_path.resolve()}")
