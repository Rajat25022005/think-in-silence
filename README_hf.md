cat > README_hf.md << 'EOF'
---
language: en
license: mit
tags:
  - question-answering
  - reasoning
  - latent-reasoning
  - jepa
  - multi-hop-qa
datasets:
  - rajat5039/wiki-multihop-qa-500k
  - hotpot_qa
  - gsm8k
  - commonsense_qa
metrics:
  - bleu
  - rouge
library_name: transformers
---

# think-in-silence

A model that reasons in pure latent space. No chain-of-thought tokens. No RL. No reasoning traces.

## Results

| Thinking Steps (K) | R@1 | BLEU | ROUGE-1 |
|---|---|---|---|
| K=0 (no thinking) | 0.002 | 0.000 | 0.000 |
| K=1 | 0.064 | 0.000 | 0.028 |
| K=2 | 0.256 | 0.009 | 0.084 |
| K=4 | 0.504 | 0.044 | 0.218 |
| K=8 | 0.474 | 0.231 | 0.594 |
| K=16 | 0.406 | 0.185 | 0.542 |

## Usage
```python
from src.models.lc_thought import LCThought
from src.utils.checkpoint import load_checkpoint
import torch, yaml
from types import SimpleNamespace

# Load config and model
cfg = yaml.safe_load(open("configs/decoder.yaml"))
model = LCThought(cfg)
load_checkpoint("checkpoints/stage3/step_0050000.pt", model)
model.eval()

# Inference
answers = model.generate(q_ids, q_mask, n_steps=8)
```

## License
MIT
EOF