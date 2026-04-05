<div align="center">

```
████████╗██╗  ██╗██╗███╗   ██╗██╗  ██╗      ██╗███╗   ██╗
╚══██╔══╝██║  ██║██║████╗  ██║██║ ██╔╝      ██║████╗  ██║
   ██║   ███████║██║██╔██╗ ██║█████╔╝       ██║██╔██╗ ██║
   ██║   ██╔══██║██║██║╚██╗██║██╔═██╗       ██║██║╚██╗██║
   ██║   ██║  ██║██║██║ ╚████║██║  ██╗      ██║██║ ╚████║
   ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝     ╚═╝╚═╝  ╚═══╝

███████╗██╗██╗     ███████╗███╗   ██╗ ██████╗███████╗
██╔════╝██║██║     ██╔════╝████╗  ██║██╔════╝██╔════╝
███████╗██║██║     █████╗  ██╔██╗ ██║██║     █████╗
╚════██║██║██║     ██╔══╝  ██║╚██╗██║██║     ██╔══╝
███████║██║███████╗███████╗██║ ╚████║╚██████╗███████╗
╚══════╝╚═╝╚══════╝╚══════╝╚═╝  ╚═══╝ ╚═════╝╚══════╝
```

**A model that reasons in pure latent space.**  
No tokens. No chain-of-thought labels. No reinforcement learning.

<br>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗_Transformers-4.38+-FFD21E?style=flat-square)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

</div>

---

## The Idea

Most language models think out loud.

```
Q: A train travels 60 mph for 2 hours. How far?
A: The train travels 60 × 2 = 120 miles.       ← reasoning exposed as tokens
   The answer is 120.
```

**think-in-silence** doesn't. Instead of generating intermediate steps as text, it runs *K learned thought vectors* through a recurrent cross-attention chain — entirely in embedding space — before predicting an answer:

```
Q: [encoded → 256-dim context vector]
       │
       ▼
   h₀  →  ThoughtBlock(h₀, ctx)  →  h₁        ← silent step 1
   h₁  →  ThoughtBlock(h₁, ctx)  →  h₂        ← silent step 2
   h₂  →  ThoughtBlock(h₂, ctx)  →  h₃        ← silent step 3
                    ...
   hₖ₋₁ → ThoughtBlock(hₖ₋₁, ctx) → hₖ       ← silent step K
       │
       ▼
   Predictor MLP  →  pred ∈ ℝ²⁵⁶
       │
   MSE(pred, answer_embedding)                  ← JEPA objective
```

Trained only on `(question, answer)` pairs. **No reasoning traces. No RL. No decoder.**

---

## Architecture

```
                    ┌─────────────────────────────────────────────┐
  Question          │  STUDENT  (trained by backprop)             │
  ──────────►  Enc ─┤                                             ├──► Predictor ──► pred
              (🔒)  │  ThoughtModule: h₀→h₁→h₂→...→hₖ              |         │
                    └─────────────────────────────────────────────┘         │
                                                                            │  MSE
                    ┌─────────────────────────────────────────────┐         │
  Answer            │  TEACHER  (EMA, no gradients)               │         │
  ──────────►  Enc ─┤                                             ├─────────►
              (🔒)  │  sentence_emb ∈ ℝ²⁵⁶                        |
                    └─────────────────────────────────────────────┘
```

**ThoughtBlock** — one reasoning step:
```
h  →  LayerNorm  →  Self-Attention(h, h)      +  h    ← refine existing thought
h  →  LayerNorm  →  Cross-Attention(h, ctx)   +  h    ← read question context
h  →  LayerNorm  →  FFN(h)                    +  h    ← mix
```

The EMA teacher (momentum 0.990 → 0.9999, cosine schedule) provides stable regression targets, preventing collapse without contrastive negatives or RL rewards.

---

## Key Experiments

### 1 — Think Time = Performance

Evaluate the **same checkpoint** at K = 0, 1, 2, 4, 8, 16 steps:

#### Retrieval (Recall@K)

| Reasoning Steps | R@1 | R@5 | R@10 |
|:---:|:---:|:---:|:---:|
| K = 0 (no reasoning) | 0.20% | 0.98% | 1.95% |
| K = 1  | 6.45% | 15.23% | 21.88% |
| K = 2  | 25.59% | 52.54% | 60.55% |
| K = 4  | **50.39%** | **71.29%** | **78.13%** |
| K = 8  | 47.46% | 66.60% | 72.66% |
| K = 16 | 40.63% | 58.98% | 66.02% |

#### Generation Quality (BLEU / ROUGE)

| Reasoning Steps | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
|:---:|:---:|:---:|:---:|:---:|
| K = 0 (no reasoning) | 0.000 | 0.000 | 0.000 | 0.000 |
| K = 1  | 0.000 | 0.028 | 0.000 | 0.027 |
| K = 2  | 0.009 | 0.084 | 0.003 | 0.084 |
| K = 4  | 0.044 | 0.218 | 0.030 | 0.217 |
| K = 8  | **0.231** | **0.594** | **0.200** | **0.594** |
| K = 16 | 0.185 | 0.542 | 0.135 | 0.544 |

**Key findings:**
- **K = 0 → K = 1**: Near-zero performance without any reasoning steps confirms the ThoughtModule is essential.
- **K = 4** achieves peak retrieval (R@1: 50.4%), showing rapid convergence in latent reasoning.
- **K = 8** achieves peak generation quality (BLEU: 0.231, ROUGE-1: 0.594).
- **K > 8 degrades slightly** — over-thinking past the optimal depth introduces noise, consistent with findings in iterative latent models.
- **Monotonic improvement from K = 0 → K = 4–8** confirms the model genuinely reasons across steps rather than memorising at a fixed depth.

### 2 — Thought Trajectory Probing

At each step hₖ, a linear probe predicts:
- **Answer type** — numeric / yes-no / entity / phrase
- **Question category** — math / factual / commonsense / strategy

Reveals what information each reasoning step encodes spontaneously — with zero supervision.

### 3 — Reconstruction Quality by Step

MSE between `pred` and `answer_emb` at each K decreases as reasoning steps accumulate:

```
K=0 → MSE ≈ ∞   (random — no reasoning)
K=1 → MSE = high  (early, rough prediction)
K=4 → MSE = low   (retrieval-optimal)
K=8 → MSE = low   (generation-optimal)
```

The decreasing curve from K = 0 through K = 8 confirms each additional thought step meaningfully improves answer prediction in latent space.

### 4 — t-SNE Thought Trajectories

How does h₀ → hₖ move through 2D latent space?  
Distinct category clusters = the model learns different reasoning paths per question type.

---

## Training Data

Five QA datasets, streamed and interleaved — no bulk download required:

| Dataset | Task | Size | Weight |
|---------|------|------|--------|
| [HotpotQA](https://hotpotqa.github.io) | Multi-hop factual | 113K | 35% |
| [GSM8K](https://github.com/openai/grade-school-math) | Grade-school math | 8.5K | 20% |
| [CommonsenseQA](https://www.tau-nlp.org/commonsenseqa) | Commonsense MC | 12K | 20% |
| [ARC-Challenge](https://allenai.org/data/arc) | Science exam (hard) | 7.8K | 15% |
| [StrategyQA](https://allenai.org/data/strategyqa) | Yes/no multi-hop | 2.8K | 10% |

No chain-of-thought annotations used anywhere.

---

## Quickstart

```bash
git clone https://github.com/Rajat25022005/think-in-silence
cd think-in-silence
pip install -r requirements.txt
```

**Train** (GCP L4, ~5–6 hrs for 100k steps):
```bash
tmux new -s think
bash run.sh
# Ctrl+B, D  to detach
```

**Evaluate a checkpoint:**
```bash
python eval.py --ckpt checkpoints/base/step_0100000.pt --config configs/base.yaml
```

**Evaluate with t-SNE trajectories:**
```bash
python eval.py --ckpt checkpoints/base/final.pt --config configs/base.yaml --tsne
```

**Adjust reasoning steps at inference (no retraining):**
```python
from src.models.lc_thought import LCThought

model = LCThought(n_steps=8)
# load checkpoint...

# Run with 16 steps instead of 8 — no retraining needed
out = model(q_ids, q_mask, a_ids, a_mask, n_steps=16)
```

---

## Project Structure

```
think-in-silence/
│
├── src/
│   ├── models/
│   │   ├── thought_block.py      # One reasoning step (self-attn → cross-attn → FFN)
│   │   ├── thought_module.py     # K-step recurrent chain with learnable h₀
│   │   ├── encoder.py            # Frozen DistilBERT + projection heads
│   │   └── lc_thought.py         # Full model: student + EMA teacher + predictor
│   │
│   ├── datasets/
│   │   └── qa_datasets.py        # Streaming interleaved QA loader
│   │
│   ├── training/
│   │   └── trainer.py            # Loop, cosine EMA schedule, BF16, checkpointing
│   │
│   └── eval/
│       ├── evaluator.py          # Retrieval, K-scaling, probing, reconstruction
│       └── visualise.py          # Publication-quality dark-theme plots
│
├── configs/
│   └── base.yaml                 # Training config (K=8, proj_dim=256, 100k steps)
│
├── main.py                       # Training entry point
├── eval.py                       # Evaluation entry point
├── run.sh                        # GCP one-command launch
└── requirements.txt
```

---

## How It Compares

|  | [Quiet-STaR](https://arxiv.org/abs/2403.09629) | [Coconut](https://arxiv.org/abs/2412.06769) | **think-in-silence** |
|--|:--:|:--:|:--:|
| Reasoning medium | Token vocabulary | LM hidden states | **Dedicated latent space** |
| Training signal | REINFORCE (RL) | Token supervision | **JEPA MSE — no RL** |
| Reasoning module | Vocab embeddings | LM backbone | **Separate ThoughtModule** |
| CoT labels needed | No | Partial | **No** |
| K adjustable at inference | No | No | **Yes** |
| Backbone modified | Yes | Yes | **No — fully frozen** |

---

## Configuration

Key settings in `configs/base.yaml`:

```yaml
model:
  n_steps:        8      # K reasoning steps
  proj_dim:       256    # Latent space dimension
  shared_weights: false  # true = Universal Transformer (1 block, K passes)

training:
  max_steps:    100000   # ~5-6 hrs on L4
  batch_size:   64
  lr:           1.0e-4
  ema_momentum_start: 0.990
  ema_momentum_end:   0.9999
```

---

## Built On

- [T-JEPA](https://github.com/Rajat25022005/self-supervised-text-jepa) — predecessor project
- [I-JEPA](https://arxiv.org/abs/2301.08243) — Assran et al., Meta AI (2023)
- [Coconut](https://arxiv.org/abs/2412.06769) — Hao et al. (2024)
- [Quiet-STaR](https://arxiv.org/abs/2403.09629) — Zeiler et al. (2024)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)

---

<div align="center">
<sub>Built by Rajat Malik · 2026 · MIT License</sub>
</div>
