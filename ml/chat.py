"""
Persona Engine — Interactive Persona Chat & Style Inspector
Test your fine-tuned persona live in Google Colab or terminal.

Usage:
    python chat.py
    python chat.py --adapter models/adapters/vivek_adapter
    python chat.py --test-prompts
"""

import argparse
import sys
from pathlib import Path

# Add ml root to path
ml_root = Path(__file__).resolve().parent
if str(ml_root) not in sys.path:
    sys.path.insert(0, str(ml_root))

from src.inference import GenerationParams, PersonaGenerator


def run_test_prompts(generator):
    print("\n" + "=" * 60)
    print("RUNNING BENCHMARK STYLE PROMPTS (VIVEK PERSONA)")
    print("=" * 60)

    test_dialogues = [
        ("Naata", "Bhai katai chatai film thi, weekend ka kya scene hai?"),
        ("Naata", "Kya lagta hai kohli century maarega aaj?"),
        ("Naata", "Bhai phone me download karke dekhe ya online?"),
        ("Naata", "Bhai gym chal raha hai ya aalsi ho gaya?"),
    ]

    params = GenerationParams(temperature=0.7, max_new_tokens=100)

    for speaker, prompt in test_dialogues:
        print(f"\n[Prompt from {speaker}]: {prompt}")
        messages = [{"speaker": speaker, "text": prompt}]
        reply = generator.generate(messages, params=params)
        print(f"[Vivek Response]: {reply}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Chat with Persona Model")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model name",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="models/adapters/vivek_adapter",
        help="Path to trained LoRA adapter",
    )
    parser.add_argument(
        "--test-prompts",
        action="store_true",
        help="Run predefined benchmark prompts instead of interactive loop",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to test Vivek's response directly",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.7,
        help="Generation temperature",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PERSONA ENGINE — INFERENCE ENGINE")
    print("=" * 60)
    print(f"Base Model  : {args.model}")
    print(f"LoRA Adapter: {args.adapter}")

    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        if (ml_root / args.adapter).exists():
            adapter_path = ml_root / args.adapter
        elif (Path.cwd() / args.adapter).exists():
            adapter_path = Path.cwd() / args.adapter

    generator = PersonaGenerator(
        base_model_name_or_path=args.model,
        adapter_path=adapter_path if adapter_path.exists() else None,
    )

    print("\nLoading model and adapter (this takes ~30 seconds)...")
    generator.load_model()

    if generator.is_adapter_loaded:
        print("SUCCESS: LoRA Persona Adapter attached!")
    else:
        print(f"NOTE: Adapter not found at '{adapter_path}'. Running base model zero-shot.")

    if args.prompt:
        params = GenerationParams(temperature=args.temp, max_new_tokens=100)
        messages = [{"speaker": "Naata", "text": args.prompt}]
        print(f"\n[Prompt from Naata]: {args.prompt}")
        reply = generator.generate(messages, params=params)
        print(f"[Vivek Response]: {reply}\n")
        return

    if args.test_prompts:
        run_test_prompts(generator)
        return

    # By default, run the test prompts first
    run_test_prompts(generator)

    # In non-interactive environments (e.g., Colab !python or CI pipe), exit gracefully
    if not sys.stdin.isatty():
        print("\n[NOTE] Non-interactive environment detected (Colab ! command).")
        print("Benchmark evaluation completed. To chat with custom questions, run:")
        print('  !python chat.py --adapter models/adapters/vivek_adapter --prompt "Bhai shaam ko kya scene hai?"')
        print("Or use the interactive Python cell snippet.")
        return

    print("\n" + "=" * 60)
    print("INTERACTIVE CHAT MODE (type 'exit' or 'quit' to end)")
    print("=" * 60)

    conversation_history = []
    params = GenerationParams(temperature=args.temp, max_new_tokens=128)

    while True:
        try:
            user_input = input("\nYou (Naata): ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q"}:
                print("Goodbye!")
                break

            conversation_history.append({"speaker": "user", "text": user_input})
            reply = generator.generate(conversation_history, params=params)
            print(f"Vivek: {reply}")
            conversation_history.append({"speaker": "persona", "text": reply})

            # Keep context window manageable
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]

        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


if __name__ == "__main__":
    main()
