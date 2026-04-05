import torch
import yaml
from types import SimpleNamespace
from src.models.lc_thought import LCThought
from src.utils.checkpoint import load_checkpoint


# ── Config loader ─────────────────────────────────────────────────────────────

def load_config(path: str) -> SimpleNamespace:
    with open(path) as f:
        raw = yaml.safe_load(f)
    def _to_ns(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
        return d
    return _to_ns(raw)


# ── Setup ─────────────────────────────────────────────────────────────────────

cfg      = load_config("configs/decoder.yaml")
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading model on {device}...")
model    = LCThought(cfg).to(device)
load_checkpoint("checkpoints/stage3/step_0050000.pt", model, device=str(device))
model.eval()
tokenizer = model.encoder.tokenizer
print("Model ready.\n")


# ── Inference function ────────────────────────────────────────────────────────

def ask(question: str, n_steps: int = 8) -> str:
    enc = tokenizer(
        question,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        answers = model.generate(
            enc["input_ids"],
            enc["attention_mask"],
            n_steps=n_steps
        )
    return answers[0]


# ── K-scaling demo ────────────────────────────────────────────────────────────

def demo_k_scaling(question: str):
    print("=" * 60)
    print(f"Q: {question}")
    print("-" * 60)
    for k in [0, 1, 2, 4, 8, 16]:
        answer = ask(question, n_steps=k)
        print(f"  K={k:2d}  →  {answer}")
    print("=" * 60)
    print()


# ── Interactive mode ──────────────────────────────────────────────────────────

def interactive():
    print("=" * 60)
    print("  think-in-silence — Interactive Demo")
    print("  Type a question, or 'quit' to exit.")
    print("  Format: your question [k=8]  (k is optional)")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            break

        # Parse optional k= argument
        n_steps = 8
        if "[k=" in user_input.lower():
            try:
                k_part   = user_input.split("[k=")[-1].rstrip("]").strip()
                n_steps  = int(k_part)
                question = user_input.split("[k=")[0].strip()
            except ValueError:
                question = user_input
        else:
            question = user_input

        answer = ask(question, n_steps=n_steps)
        print(f"A (K={n_steps}): {answer}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run K-scaling on a few example questions
    demo_k_scaling("Which city did Marie Curie move to after leaving Warsaw?")
    demo_k_scaling("What force keeps planets in orbit around the sun?")
    demo_k_scaling("If a train travels 60 mph for 2 hours, how far does it go?")
    demo_k_scaling("Which scientist developed the theory of relativity and also won the Nobel Prize?")

    # Drop into interactive mode
    interactive()