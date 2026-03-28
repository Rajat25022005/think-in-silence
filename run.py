import sys
import subprocess


COMMANDS = {
    "stage1":   ["python", "main.py"],
    "stage2":   ["python", "train_decoder.py"],
    "stage3":   ["python", "finetune.py"],
    "eval":     ["python", "eval.py"],
    "synth":    ["python", "-m", "src.datasets.synthetic.pipeline"],
    "test":     ["python", "-m", "pytest", "tests/", "-v"],
}

usage = f"""Usage: python run.py <command> [args...]

Commands:
  stage1   Run Stage 1 JEPA training       (main.py)
  stage2   Run Stage 2 decoder training    (train_decoder.py)
  stage3   Run Stage 3 joint finetune      (finetune.py)
  eval     Run evaluation                  (eval.py)
  synth    Run synthetic data pipeline
  test     Run all unit tests

Example:
  python run.py stage1 --config configs/bge_large.yaml --wandb
  python run.py eval --eval_type all --max_samples 500
  python run.py test
"""

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(usage)
        sys.exit(0)

    cmd    = COMMANDS[sys.argv[1]]
    extra  = sys.argv[2:]
    result = subprocess.run(cmd + extra)
    sys.exit(result.returncode)
