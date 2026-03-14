# Think-in-Silence — Revised Build Plan (v2)

> **Profile:** Idea-first builder · AI-assisted implementation · Research + Portfolio + Paper  
> **Duration:** 42 days across 7 phases  
> **Goal:** Latent reasoning model with swappable backbone, generative decoder, synthetic data pipeline, and published paper

---

## What Changed From v1

| v1 Plan | v2 Plan |
|---|---|
| DistilBERT hardcoded | Swappable backbone (BGE-Large → LLaMA) |
| 5 public datasets only | 250GB corpus + synthetic QA pipeline |
| Retrieval only (Recall@K) | Retrieval + Text generation (BLEU, ROUGE) |
| No decoder | 3-stage training with decoder |
| 28 days | 42 days (more experiments, more impact) |
| Single config | Ablation configs, multi-stage configs |
| No tests | Full test suite |

---

## The Big Picture (Unchanged, Strengthened)

A model that reasons entirely in vector space.  
Question → encoded context → K silent thought steps → generated answer.

**The central claim:** More thinking steps = better answers.  
Provable. Adjustable at inference. No retraining needed.

**What's new in v2:**
- The model now *generates* answers as text, not just predicts embeddings
- The backbone is swappable — start with BGE-Large, scale to LLaMA
- The dataset is yours — synthesized from 250GB of raw text
- The experiments are stronger — generation quality proves latent reasoning more convincingly than retrieval alone

---

## How You'll Build This

Since you work best at the ideas and direction level:

```
Your job:          Understand every design decision
                   Direct what gets built and why
                   Evaluate results and know what they mean
                   Write the paper

AI's job:          Implement what you specify
                   Write boilerplate and tests
                   Debug syntax errors
                   Convert your pseudocode to real code

The rule:          Never let AI write something you
                   can't explain line by line afterward
                   Read everything it writes
                   Ask why until you understand
```

---

## Phase Overview

| Phase | Focus | Days |
|---|---|---|
| 1 | Foundations + Environment | 1–4 |
| 2 | Core Architecture | 5–10 |
| 3 | Backbone System | 11–14 |
| 4 | Data Pipeline + Synthesis | 15–20 |
| 5 | Three-Stage Training | 21–28 |
| 6 | Evaluation + Experiments | 29–36 |
| 7 | Paper + Portfolio Polish | 37–42 |

---

# Phase 1 — Foundations
> **Days 1–4 · Goal: Environment, math intuition, folder structure, first prototype**

---

## Day 1 — Project Structure

Create the full folder tree before writing a single line of model code.

```
think-in-silence/
│
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py
│   │   ├── thought_block.py
│   │   ├── thought_module.py
│   │   ├── decoder.py               ← NEW
│   │   ├── lc_thought.py
│   │   └── backbones/               ← NEW FOLDER
│   │       ├── __init__.py
│   │       ├── distilbert.py
│   │       ├── bge_large.py
│   │       └── llama_encoder.py
│   │
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── qa_datasets.py
│   │   ├── synthetic/               ← NEW FOLDER
│   │   │   ├── __init__.py
│   │   │   ├── generator.py
│   │   │   ├── filter.py
│   │   │   └── pipeline.py
│   │   └── preprocessing/           ← NEW FOLDER
│   │       ├── __init__.py
│   │       ├── cleaner.py
│   │       ├── difficulty_filter.py
│   │       └── stats.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── decoder_trainer.py       ← NEW
│   │   ├── finetune_trainer.py      ← NEW
│   │   ├── losses.py                ← NEW
│   │   └── schedulers.py            ← NEW
│   │
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── evaluator.py
│   │   ├── generation_eval.py       ← NEW
│   │   ├── probing.py               ← NEW
│   │   ├── visualise.py
│   │   └── benchmarks/              ← NEW FOLDER
│   │       ├── __init__.py
│   │       ├── gsm8k_eval.py
│   │       ├── hotpot_eval.py
│   │       └── arc_eval.py
│   │
│   └── utils/                       ← NEW FOLDER
│       ├── __init__.py
│       ├── checkpoint.py
│       ├── logging.py
│       ├── seed.py
│       └── device.py
│
├── configs/
│   ├── base.yaml
│   ├── bge_large.yaml               ← NEW
│   ├── llama_encoder.yaml           ← NEW
│   ├── decoder.yaml                 ← NEW
│   ├── ablations/                   ← NEW FOLDER
│   │   ├── shared_weights.yaml
│   │   ├── dim_256.yaml
│   │   ├── dim_512.yaml
│   │   └── k_steps.yaml
│   └── experiment/                  ← NEW FOLDER
│       ├── small_run.yaml
│       └── full_run.yaml
│
├── scripts/                         ← NEW FOLDER
│   ├── prepare_data.sh
│   ├── generate_synthetic.sh
│   ├── train_stage1.sh
│   ├── train_stage2.sh
│   ├── train_stage3.sh
│   ├── run_evals.sh
│   └── upload_checkpoint.sh
│
├── notebooks/
│   ├── 01_sanity_checks.ipynb
│   ├── 02_data_exploration.ipynb    ← NEW
│   ├── 03_backbone_comparison.ipynb ← NEW
│   ├── 04_kscaling_analysis.ipynb   ← NEW
│   ├── 05_decoder_outputs.ipynb     ← NEW
│   └── 06_paper_figures.ipynb       ← NEW
│
├── tests/                           ← NEW FOLDER
│   ├── test_encoder.py
│   ├── test_thought_block.py
│   ├── test_thought_module.py
│   ├── test_decoder.py
│   ├── test_dataloader.py
│   └── test_ema.py
│
├── paper/                           ← NEW FOLDER
│   ├── main.tex
│   ├── figures/
│   └── refs.bib
│
├── checkpoints/
│   ├── stage1/
│   ├── stage2/
│   └── stage3/
│
├── results/
│   ├── metrics/
│   └── plots/
│
├── main.py
├── train_decoder.py                 ← NEW
├── finetune.py                      ← NEW
├── eval.py
├── run.sh
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Day 1 — Requirements File

Pin exact versions. Never leave versions unspecified.

```
torch==2.2.0
transformers==4.38.0
sentence-transformers==2.5.0       ← NEW: for BGE-Large
datasets==2.18.0
pyyaml==6.0.1
numpy==1.26.4
scikit-learn==1.4.1
matplotlib==3.8.3
seaborn==0.13.2
tqdm==4.66.2
wandb==0.16.4
einops==0.7.0
evaluate==0.4.1                    ← NEW: for BLEU/ROUGE scoring
bert-score==0.3.13                 ← NEW: for BERTScore generation eval
bitsandbytes==0.42.0               ← NEW: for LLaMA quantization
accelerate==0.27.0                 ← NEW: for large model handling
```

---

## Day 1 — Base Configuration

`configs/base.yaml` — single source of truth.

```yaml
model:
  backbone: "bge_large"            # distilbert | bge_large | llama
  n_steps: 8
  proj_dim: 256
  n_heads: 4
  ffn_dim: 512
  dropout: 0.1
  shared_weights: false

training:
  stage: 1                         # 1=JEPA | 2=decoder | 3=joint
  max_steps: 100000
  batch_size: 64
  lr: 1.0e-4
  warmup_steps: 2000
  grad_clip: 1.0
  ema_momentum_start: 0.990
  ema_momentum_end: 0.9999
  log_every: 100
  ckpt_every: 5000
  ckpt_dir: "checkpoints/stage1"

decoder:
  n_layers: 3
  n_heads: 4
  max_gen_len: 64
  tie_embeddings: true

eval:
  k_values: [0, 1, 2, 4, 8, 16]
  recall_at: [1, 5, 10]
  n_probe_samples: 2000
  tsne_perplexity: 30

data:
  max_q_len: 128
  max_a_len: 64
  synthetic_ratio: 0.6             # 60% synthetic, 40% public datasets
```

---

## Day 2–3 — Core Math (Do Not Skip)

### Concept 1: Why BGE-Large is Better Than DistilBERT

DistilBERT was trained with Masked Language Modeling — predict missing words.  
BGE-Large was trained specifically to produce meaningful sentence embeddings — similar sentences produce similar vectors.

Your ThoughtModule cross-attends over these vectors. Richer input = more to reason over.

**Action:** In a notebook, encode the same sentence with both models. Compute cosine similarity between semantically related sentences. BGE-Large should score 0.85+. DistilBERT will score 0.5–0.6. This is your justification for the backbone choice in the paper.

### Concept 2: Why Three Training Stages Work

If you train everything end-to-end from scratch, the decoder learns to shortcut — it bypasses the ThoughtModule and maps encoder output directly to text. The thinking never develops.

```
Stage 1: Force ThoughtModule to compress answer meaning into hₖ
         (MSE loss — hₖ must predict the answer embedding)
         ThoughtModule learns to think.

Stage 2: Freeze ThoughtModule, teach decoder to read hₖ
         (cross-entropy loss on generated tokens)
         Decoder learns hₖ contains the answer.

Stage 3: Unfreeze jointly, fine-tune together
         (combined MSE + cross-entropy loss)
         System refines end-to-end.
```

**Action:** Draw this on paper. Understand which parameters are frozen at each stage. This diagram goes in your paper.

### Concept 3: Why Synthetic Data Helps

Your ThoughtModule only learns to reason over patterns it sees during training.  
5 public datasets = ~144K examples = limited reasoning variety.  
250GB of synthesized multi-hop QA = millions of diverse reasoning patterns.

The key insight: **multi-hop questions require K > 1 steps.**  
Single-hop questions plateau at K=1.  
If your training data is mostly single-hop, K-scaling won't show improvement.  
Synthetic data lets you control this ratio.

---

## Day 4 — 50-Line Prototype

Build end-to-end in one notebook cell using BGE-Large:

1. Load `BAAI/bge-large-en-v1.5`, freeze all parameters
2. Encode a question → full token sequence projected to 256-dim
3. Encode an answer → mean-pooled vector projected to 256-dim
4. Build minimal ThoughtBlock (self-attn + cross-attn + FFN)
5. Initialize h0, loop K=4 steps
6. Project hₖ through predictor → MSE against answer embedding
7. Build minimal 2-layer decoder, generate 3 tokens from hₖ
8. Compute cross-entropy on generated tokens
9. Combined loss = MSE + CrossEntropy, call backward()

**Success criterion:** Finite loss. Gradients flow through both losses. No NaN.

---

# Phase 2 — Core Architecture
> **Days 5–10 · Goal: All model files built, shape-verified, decoder working**

---

## Day 5 — backbones/

### Why a Backbones Folder

Your encoder.py should not care which backbone it uses. The backbone folder abstracts this away. Changing from BGE-Large to LLaMA should be one config line, not a code rewrite.

### backbones/bge_large.py

```
What it does:
  Loads BAAI/bge-large-en-v1.5
  Freezes all parameters
  Returns full token sequence: (B, seq_len, 1024)
  Returns mean-pooled vector: (B, 1024)
  Handles attention mask correctly in pooling

Output dimension: 1024
```

### backbones/distilbert.py

```
What it does:
  Same interface as bge_large.py
  Loads distilbert-base-uncased
  Returns full token sequence: (B, seq_len, 768)
  Returns mean-pooled vector: (B, 768)

Output dimension: 768
```

### backbones/llama_encoder.py

```
What it does:
  Loads meta-llama/Llama-3-8B in 4-bit quantization
  Uses last hidden state as sequence representation
  Returns full token sequence: (B, seq_len, 4096)
  Returns mean-pooled vector: (B, 4096)

Output dimension: 4096
Note: Requires A100 or H100 without quantization
      Use bitsandbytes 4-bit for L4 compatibility
```

### encoder.py — Now Backbone-Agnostic

```python
class FrozenEncoder(nn.Module):
    def __init__(self, cfg):
        backbone_name = cfg.model.backbone
        # loads correct backbone from backbones/ folder
        # question_proj: Linear(backbone_dim, proj_dim) + LayerNorm
        # answer_proj:   Linear(backbone_dim, proj_dim) + LayerNorm
    
    def encode_question(self, ids, mask):
        # returns (B, seq_len, proj_dim)
    
    def encode_answer(self, ids, mask):
        # returns (B, proj_dim)
```

Changing backbone = one line in config. Zero code changes.

---

## Day 6 — thought_block.py (Unchanged from v1)

Same architecture. Pre-LayerNorm. Three sub-layers with residuals.

```
h: (B, 1, dim) → Self-Attention → Cross-Attention(ctx) → FFN → h: (B, 1, dim)
```

Nothing changes here. The ThoughtBlock doesn't know or care about the backbone.

---

## Day 7 — thought_module.py (Minor Addition from v1)

Same K-step recurrent chain. Add one thing:

**Intermediate state capture** — store h after every step, not just the final one.

```python
def forward(self, ctx, n_steps=None, return_all_states=False):
    h = self.h0.expand(B, -1, -1)
    states = [h]                    # h0 = step 0
    
    for k in range(K):
        block = self.blocks[k % len(self.blocks)]
        h = block(h, ctx)
        states.append(h)            # h1, h2, ..., hK
    
    if return_all_states:
        return torch.stack(states, dim=1)  # (B, K+1, dim)
    return h                               # (B, 1, dim)
```

This feeds the probing and t-SNE experiments in Phase 6.

---

## Day 8 — decoder.py (New File)

### What the Decoder Does

Takes hₖ (B, 1, 256) and generates answer text token by token.

### Architecture: Cross-Attention Decoder

```
Input:     hₖ (B, 1, 256) — the thought
           token_ids so far (for teacher forcing during training)

Per layer:
  Self-Attention on generated tokens so far (causal mask)
  Cross-Attention: Q=token_embeddings, K=V=hₖ
  FFN

Output:    Linear(dim, vocab_size) → logits over vocabulary
```

### Two Modes

**Training mode (teacher forcing):**
```
Input:  hₖ + full answer token sequence
Output: logits for each position
Loss:   CrossEntropy(logits, shifted_answer_tokens)
```

**Inference mode (autoregressive):**
```
Input:  hₖ + [BOS] token
Loop:   Generate one token → append → repeat until [EOS] or max_len
Output: Full generated sequence as text
```

### Key Implementation Details

- Share embedding weights between input embeddings and output projection (tie_embeddings: true in config) — reduces parameters, improves coherence
- Use causal attention mask in self-attention — token at position i cannot see position i+1
- Cross-attention over hₖ: K and V come from hₖ, Q comes from token embeddings — decoder always reads the thought
- Vocabulary: use same tokenizer as backbone encoder — consistent token space

---

## Day 9 — lc_thought.py (Extended from v1)

### What Changes

Add decoder as optional component. Support three forward modes matching three training stages.

```python
class LCThought(nn.Module):
    def forward(self, q_ids, q_mask, a_ids, a_mask,
                n_steps=None,
                return_all_states=False,
                mode='stage1'):
        
        if mode == 'stage1':
            # JEPA MSE training
            # Returns: loss, pred, target
        
        elif mode == 'stage2':
            # Decoder training (ThoughtModule frozen externally)
            # Returns: loss (cross-entropy only), generated_logits
        
        elif mode == 'stage3':
            # Joint training
            # Returns: combined loss, pred, target, generated_logits
        
        elif mode == 'generate':
            # Inference — autoregressive generation
            # Returns: generated token ids
```

### generate() Method

```python
@torch.no_grad()
def generate(self, q_ids, q_mask, n_steps=None, max_len=64):
    # Encode question
    # Run ThoughtModule
    # Autoregressively decode from hₖ
    # Return decoded text string
```

This is the method that powers the K-scaling generation demo.

---

## Day 10 — Shape Verification + Tests

### tests/test_thought_module.py

```
Check: output shape (B, 1, dim) with K=8
Check: n_steps=16 override on K=8 model works
Check: return_all_states gives (B, K+1, dim)
Check: h0 gradient is not None after backward
```

### tests/test_decoder.py

```
Check: training mode logits shape (B, seq_len, vocab_size)
Check: generate() returns a non-empty string
Check: generate() with K=1 and K=8 both work
Check: cross-attention receives hₖ correctly
```

### tests/test_ema.py

```
Check: teacher parameters change after update_ema()
Check: teacher parameters change LESS with higher momentum
Check: teacher requires_grad is False throughout
Check: update_ema() with @no_grad doesn't affect computational graph
```

Run all tests before touching data.

---

# Phase 3 — Backbone Upgrade Verification
> **Days 11–14 · Goal: BGE-Large working end-to-end, baseline numbers established**

---

## Day 11 — BGE-Large Integration

Switch to BGE-Large. Run the full forward pass.  
Everything in encoder.py should work without changes — only the backbone changes.

**Verify:**
- `encode_question` returns (B, seq_len, 256) — projection from 1024 works
- `encode_answer` returns (B, 256) — pooling from 1024 works
- Loss is finite and lower than DistilBERT baseline (it should be)
- Training step time is acceptable (BGE-Large is heavier — benchmark this)

---

## Day 12 — notebooks/03_backbone_comparison.ipynb

This notebook produces a figure for your paper.

**Experiment:**
1. Encode 500 question-answer pairs with DistilBERT
2. Encode the same 500 pairs with BGE-Large
3. Compute cosine similarity between question and its correct answer for both
4. Plot distribution — BGE-Large should show higher similarity = richer signal
5. Run K=4 ThoughtModule on both, compute Recall@1
6. Table: DistilBERT Recall@1 vs BGE-Large Recall@1

**Expected result:** BGE-Large Recall@1 is meaningfully higher. This justifies backbone choice in the paper.

---

## Day 13–14 — Ablation Configs

Create configs for all planned ablations. Write them now so you don't forget later.

`configs/ablations/shared_weights.yaml` — shared vs separate blocks  
`configs/ablations/dim_256.yaml` — 256-dim latent space  
`configs/ablations/dim_512.yaml` — 512-dim latent space  
`configs/ablations/k_steps.yaml` — train with K=4 vs K=8 vs K=16  

Each ablation is runnable as:
```bash
python main.py --config configs/ablations/dim_512.yaml
```

Zero code changes between ablations. All results traceable to exact config.

---

# Phase 4 — Data Pipeline + Synthetic Generation
> **Days 15–20 · Goal: Streaming loader working, synthetic QA pipeline producing multi-hop pairs**

---

## Day 15 — qa_datasets.py (Same as v1 + One Addition)

Five public datasets, streaming, interleaved. Same implementation as v1.

**Addition:** Add `difficulty_score` field to each sample.

```python
def difficulty_score(question: str) -> int:
    """
    0 = single-hop (answerable in one lookup)
    1 = two-hop (requires connecting two facts)
    2 = multi-hop (3+ connections required)
    
    Heuristic: count question words that imply connection
    ('which', 'that also', 'where did', 'who later')
    """
```

This lets the difficulty_filter.py later remove easy samples that don't benefit from K > 1 thinking.

---

## Day 16–17 — datasets/synthetic/generator.py

### The Synthesis Goal

Convert raw text passages (from your 250GB corpus) into multi-hop QA pairs.

### How It Works

```
Input:   Raw text paragraph from corpus
         Example: "Marie Curie was born in Warsaw in 1867.
                   She later moved to Paris where she conducted
                   research that led to the discovery of polonium."

Prompt to LLM API:
         "Generate 2 multi-hop questions from this paragraph.
          Each question must require connecting at least 2 facts
          to answer. Output as JSON:
          [{'question': str, 'answer': str, 'hops': int}]
          Only output JSON, no other text."

Output:  [
           {
             'question': 'In which city did the scientist born in 
                          Warsaw discover a new element?',
             'answer': 'Paris',
             'hops': 2
           }
         ]
```

### API Options (in order of quality)

1. Claude API (`claude-sonnet-4-6`) — best quality, costs money
2. GPT-4o-mini — good quality, cheap
3. Local Mistral-7B — free, slower, slightly lower quality

For 250GB corpus, local Mistral-7B is the practical choice. Claude API for quality verification of a smaller sample.

### Implementation Details

- Batch paragraphs in groups of 10 — one API call per batch
- Async requests — run 8 parallel generation jobs
- Save to JSONL format — appendable, resumable if interrupted
- Target: 5–10 million QA pairs total from 250GB
- Estimated time: 3–5 days running continuously on a single GPU

---

## Day 18 — datasets/synthetic/filter.py

Not all generated pairs are good. Filter aggressively.

### Filter Rules

```
Remove if:
  answer is empty or whitespace
  question length < 20 characters (too simple)
  answer is exactly in the question (not a real question)
  hops < 2 (single-hop, defeats the purpose)
  question doesn't end with '?' 
  answer length > 100 characters (likely malformed)
  duplicate questions (exact match dedup)
  near-duplicate questions (cosine similarity > 0.95)

Keep if:
  hops >= 2
  answer is a specific entity, number, or short phrase
  question contains connecting language
  ('which', 'where did', 'who later', 'what resulted')
```

Target retention rate: ~60–70%. If below 50%, the generator prompt needs adjustment.

---

## Day 19 — datasets/preprocessing/difficulty_filter.py

Before training, filter the combined dataset so:
- At least 70% of samples have difficulty_score >= 1
- At least 40% have difficulty_score == 2

Single-hop questions don't teach the model to use multiple thinking steps. They're not harmful but they dilute the training signal that drives K-scaling.

---

## Day 20 — Pipeline Verification

### notebooks/02_data_exploration.ipynb

Before training on synthetic data:

1. Decode 20 synthetic QA pairs — do they look like real multi-hop questions?
2. Plot difficulty score distribution — is it multi-hop heavy?
3. Plot answer length distribution — any outliers?
4. Verify category distribution matches config weights
5. Compute vocabulary overlap between synthetic and public datasets
6. Sample 5 pairs from each dataset type and print them

**Rule:** If you can't tell good samples from bad by reading 20 examples, the filter is not working. Fix the filter before training.

---

# Phase 5 — Three-Stage Training
> **Days 21–28 · Goal: All three stages complete, K-scaling working, decoder generating text**

---

## Stage 1 — JEPA Training (Days 21–24)

### training/trainer.py (Same as v1 + schedulers.py split)

Move LR schedule and EMA schedule to `training/schedulers.py`:

```python
def get_lr(step, cfg):
    if step < cfg.training.warmup_steps:
        return cfg.training.lr * step / cfg.training.warmup_steps
    progress = (step - cfg.training.warmup_steps) / (cfg.training.max_steps - cfg.training.warmup_steps)
    return cfg.training.lr * 0.5 * (1 + math.cos(math.pi * progress))

def get_ema_momentum(step, cfg):
    progress = step / cfg.training.max_steps
    start = cfg.training.ema_momentum_start
    end = cfg.training.ema_momentum_end
    return end - (end - start) * (math.cos(math.pi * progress) + 1) / 2
```

### Per-Step Order (Same as v1)

1. Batch → GPU
2. Update LR
3. Forward under BF16 autocast
4. Zero gradients
5. Scaled backward
6. Unscale gradients
7. Clip gradients
8. Optimizer step
9. Scaler update
10. EMA teacher update
11. Log (every 100)
12. Checkpoint (every 5000)

### What to Watch in WandB

- `loss` — should decrease steadily for first 20K steps, then slower
- `lr` — warmup then cosine curve
- `ema_momentum` — 0.990 → 0.9999 curve
- If loss stops decreasing before step 10K: lower LR or check EMA momentum

### Success Criterion

After 100K steps: Recall@1 > 30% on held-out test set.  
If below 20%: check encoder projections are training (not accidentally frozen).

---

## Stage 2 — Decoder Training (Days 25–26)

### training/decoder_trainer.py

**Freeze ThoughtModule completely.** Only decoder parameters train.

```python
# Freeze everything except decoder
for param in model.encoder.parameters():
    param.requires_grad = False
for param in model.thought_module.parameters():
    param.requires_grad = False
for param in model.predictor.parameters():
    param.requires_grad = False

# Only decoder trains
optimizer = AdamW(model.decoder.parameters(), lr=5e-4)
```

**Loss:** CrossEntropy on teacher-forced answer tokens.

**Duration:** ~20–30K steps. This is much faster than Stage 1.

### What to Watch

- `decoder_loss` — should decrease clearly within 5K steps
- If loss is stuck: verify hₖ is being passed to decoder correctly
- If loss goes NaN: lower LR, check embedding weights

### Success Criterion

At 20K steps, run `model.generate()` on 20 test questions.  
If at least 30% produce the correct answer (exact or near match): Stage 2 working.  
If below 20%: the thought vectors from Stage 1 may not contain enough answer information.

---

## Stage 3 — Joint Fine-tuning (Days 27–28)

### training/finetune_trainer.py

Unfreeze ThoughtModule. Train everything together.

```python
# Combined loss
mse_weight = 0.5
ce_weight = 0.5
loss = mse_weight * mse_loss + ce_weight * decoder_loss
```

**Important:** Use a lower LR than Stage 1 (1e-5, not 1e-4). Fine-tuning, not training from scratch.

**Duration:** ~20K steps. Just enough to let the components align.

### What to Watch

- Both `mse_loss` and `decoder_loss` should decrease
- If `mse_loss` spikes up: `mse_weight` is too high, reduce to 0.3
- If `decoder_loss` spikes: ThoughtModule drifted, reduce `ce_weight`

---

## scripts/ — Entry Points

### train_stage1.sh
```bash
set -e
nvidia-smi
python main.py --config configs/bge_large.yaml \
               --stage 1 \
               2>&1 | tee results/stage1_log.txt
```

### train_stage2.sh
```bash
set -e
python train_decoder.py --config configs/decoder.yaml \
                         --ckpt checkpoints/stage1/final.pt \
                         2>&1 | tee results/stage2_log.txt
```

### train_stage3.sh
```bash
set -e
python finetune.py --config configs/bge_large.yaml \
                   --stage1_ckpt checkpoints/stage1/final.pt \
                   --stage2_ckpt checkpoints/stage2/final.pt \
                   2>&1 | tee results/stage3_log.txt
```

---

# Phase 6 — Evaluation + Experiments
> **Days 29–36 · Goal: 5 experiments, all figures generated, results honest and complete**

---

## Experiment 1 — K-Scaling with Generation (Most Important)

### The Claim

More thinking steps → better generated answers. Same checkpoint. No retraining.

### Setup

Load Stage 3 checkpoint. For K = 0, 1, 2, 4, 8, 16:
- Run `model.generate()` on 500 test questions
- Compute Recall@1 (embedding retrieval)
- Compute exact match accuracy
- Compute BLEU-4 against reference answers
- Compute BERTScore F1

### K=0 Baseline

Predictor receives raw h0 (no thinking). Should clearly underperform.  
If K=0 is close to K=8: ThoughtModule isn't contributing — debug Stage 1.

### Expected Result Table

| K Steps | Recall@1 | Exact Match | BLEU-4 | BERTScore |
|:---:|:---:|:---:|:---:|:---:|
| K = 0 | — | — | — | — |
| K = 1 | — | — | — | — |
| K = 2 | — | — | — | — |
| K = 4 | — | — | — | — |
| K = 8 | — | — | — | — |
| K = 16 | — | — | — | — |

Monotonically increasing across all four metrics = strong evidence of genuine latent reasoning.

### The Demo Output

Also save actual generated text at each K for 10 representative questions.  
This table goes in the README and gets attention.

---

## Experiment 2 — Linear Probing

### The Claim

Thought vectors spontaneously encode meaningful semantic information — without any category supervision.

### Setup

At each step hₖ (0 to K), collect thought vectors for 2000 test questions.  
Train `LogisticRegression` (sklearn) to predict:
- Question category (math / factual / commonsense / science / strategy)
- Answer type (numeric / yes-no / entity / phrase)

### What to Plot

Line chart: Step (x-axis) vs Probe Accuracy (y-axis).  
Two lines: one per label type.  
Expected shape: starts near chance (20% for 5 classes), rises through K, plateaus.

---

## Experiment 3 — MSE Curve Per Step

### The Claim

Each reasoning step brings the prediction closer to the correct answer.

### Setup

For each step k from 1 to K:
- Extract hₖ from `return_all_states`
- Pass through predictor
- Compute MSE against teacher target

Plot: Step (x-axis) vs MSE (y-axis). Decreasing curve.

If curve is flat: ThoughtModule learned a trivial fixed-point — debug h0 initialization.  
If curve increases after K=4: possible overfitting to K=8, try shared_weights ablation.

---

## Experiment 4 — t-SNE Thought Trajectories

### The Claim

Different question categories take different paths through latent space.

### Setup

1. Collect h0...hK for 500 test questions across all 5 categories
2. Flatten to (500 × (K+1), 256)
3. Run t-SNE (perplexity=30, n_iter=1000) → 2D points
4. Reshape to trajectories (500, K+1, 2)
5. Plot arrows from h0 to hK, colored by category

### What to Look For

- Math questions cluster together
- Yes/no strategy questions cluster separately
- Arrows within each category point in similar directions
- Different categories point in different directions

If fully mixed with no structure: increase n_probe_samples to 2000.  
Try perplexity 15 or 50. If still mixed: the backbone representations may need more training.

---

## Experiment 5 — Generation Quality by Category (New in v2)

### The Claim

The model reasons differently for different question types — and generation quality reflects this.

### Setup

Compute BLEU-4 and BERTScore separately for each of the 5 categories.  
Run at K=8.

### What to Look For

- GSM8K (math): exact numeric answers expected — Exact Match is the metric
- StrategyQA (yes/no): binary — Exact Match expected to be high if working
- HotpotQA (multi-hop): entity answers — BERTScore is appropriate
- CommonsenseQA: phrase answers — BERTScore most informative

Per-category breakdown shows where the model is strong vs weak.  
Honest reporting of weak categories is better than hiding them.

---

## notebooks/06_paper_figures.ipynb

All five experiments produce publication-ready figures here.

Settings for all plots:
- Dark background (`plt.style.use('dark_background')`)
- Figure size: (10, 6) for single plots, (14, 6) for side-by-side
- Font size: 14 for labels, 12 for ticks
- DPI: 300 for saving
- Format: PDF for paper (lossless), PNG for README

Save all figures to `paper/figures/` automatically.

---

# Phase 7 — Paper + Portfolio Polish
> **Days 37–42 · Goal: arXiv-ready paper, clean GitHub, reproducible codebase**

---

## Day 37 — Paper Section 1 + 2 (Should Already Be Drafted)

If you've been writing as you go, Sections 1 and 2 are already done.  
If not, write them now before touching results.

### Abstract (Write This Last, Outline It Now)

Four sentences:
1. Problem: token-level reasoning is expensive and exposes the reasoning process unnecessarily
2. Approach: K recurrent cross-attention steps in 256-dim latent space, BGE-Large encoder, text decoder, trained with JEPA MSE + cross-entropy
3. Key result: accuracy and generation quality improve monotonically with K using same checkpoint, demonstrated across 5 QA categories
4. Significance: no RL, no CoT labels, backbone fully frozen, K adjustable at inference

### Section 2 — Related Work

Three prior works, each in one paragraph with clear differentiation:

**I-JEPA (Assran et al., 2023)** — JEPA objective for vision, extended here to language reasoning  
**Coconut (Hao et al., 2024)** — most similar approach, requires backbone modification and partial token supervision; our model requires neither  
**Quiet-STaR (Zeiler et al., 2024)** — hidden reasoning tokens, but uses REINFORCE and modifies backbone; we use simple MSE and leave backbone frozen

---

## Day 38–39 — Paper Sections 3 + 4

### Section 3 — Architecture

Full pipeline description with equations.

Attention formula:
```
Attention(Q, K, V) = softmax(QKᵀ / √d) × V
```

EMA update:
```
θ_teacher ← m × θ_teacher + (1 - m) × θ_student
```

JEPA objective:
```
L_JEPA = MSE(f_pred(hₖ), sg(f_enc(answer)))
```

Decoder objective:
```
L_dec = CrossEntropy(f_dec(hₖ, tokens[:-1]), tokens[1:])
```

Combined Stage 3 loss:
```
L = α × L_JEPA + β × L_dec
```

Justify each design choice in one sentence:
- Pre-LayerNorm: gradient stability in deep networks
- Frozen backbone: prevent catastrophic forgetting, reduce compute
- Predictor bottleneck: prevent shortcut learning
- Cosine EMA schedule: fast adaptation early, stable targets late
- BGE-Large: superior embedding quality vs MLM-trained models

### Section 4 — Experiments

Five experiments, each structured as: Hypothesis → Setup → Result → Interpretation.

For each result: be honest about what it shows and what it doesn't.  
If K=16 is only slightly better than K=8: say so. "Diminishing returns beyond K=8 suggest the model converges within that budget" is a good sentence, not a failure.

---

## Day 40 — Paper Section 5 + Full Draft

### Section 5 — Conclusion

Two paragraphs:

Paragraph 1: What you showed. One sentence per experiment. Specific numbers.  
Paragraph 2: Three concrete future directions:

1. Scaling backbone — LLaMA-3-8B encoder should significantly increase reasoning capacity; preliminary experiments with 4-bit quantization showed promising results
2. Longer chains — K > 16 not tested; continued accuracy scaling may hold, as the ThoughtModule is theoretically unbounded
3. Generation tasks beyond QA — the decoder architecture extends naturally to summarization and explanation generation

---

## Day 41 — GitHub Repository

### README Update

Fill in all experiment result tables with your actual numbers.  
Add the t-SNE figure and K-scaling plot directly in the README.  
Add the generation quality table showing K=1 vs K=8 actual text outputs.

### Reproducibility Checklist

```
□ Random seed set: torch.manual_seed(42) in main.py
□ Checkpoint downloadable (HuggingFace Hub or Google Drive link)
□ Config saved inside checkpoint state dict
□ eval.py runnable in one command with documented args
□ requirements.txt tested from fresh venv
□ All README figures generated from provided checkpoint
□ generate() demo runnable in 5 lines
□ All three training stages documented in README
```

### Repository Hygiene

- `.gitignore` covers checkpoints/, results/, __pycache__/
- No hardcoded paths anywhere — all paths from config
- No API keys in code — loaded from environment variables
- LICENSE file present (MIT)
- `README.md` has quickstart that works in under 5 minutes

---

## Day 42 — arXiv Submission

### Before Submitting

1. Run the full eval pipeline one more time from the checkpoint — verify numbers match paper
2. Ask someone else to clone the repo and run quickstart — fix anything that breaks
3. Check all figure captions are self-contained (readable without surrounding text)
4. Check all table entries are filled (no "—" left in results tables)
5. Proofread abstract three times — it's the only thing most people read

### Submission

- Category: cs.LG (Machine Learning) primary, cs.CL (Computation and Language) secondary
- Title: "Think in Silence: Latent Reasoning via Recurrent Cross-Attention with Inference-Time K Scaling"
- After submission, post on Twitter/X with the t-SNE figure and the K-scaling demo output

---

# Troubleshooting Reference

| Problem | Symptom | Fix |
|---|---|---|
| OOM on BGE-Large | CUDA OOM at batch 64 | Reduce to batch 32, gradient checkpointing |
| Backbone not loading | HuggingFace download error | Set `TRANSFORMERS_OFFLINE=1`, download manually |
| Stage 2 loss stuck | Decoder loss > 5 after 5K steps | Verify hₖ is detached correctly from Stage 1 graph |
| Generation garbage | Random tokens, no coherent text | Check tokenizer match between encoder and decoder |
| K=16 worse than K=8 | Accuracy degrades beyond trained K | Try shared_weights: true ablation |
| Synthetic data low quality | Filter retaining < 50% | Adjust generator prompt, add explicit hops requirement |
| t-SNE no clusters | All points mixed | Increase n_probe_samples, try perplexity 15 or 50 |
| NaN in Stage 3 | Loss suddenly NaN | Reduce mse_weight and ce_weight, lower LR to 5e-6 |
| Probe accuracy flat | No improvement across steps | Verify return_all_states detaches each state |
| EMA collapse | All predictions identical | Lower ema_momentum_start to 0.98, reduce predictor capacity |

---

# Milestones + Success Criteria

**End of Phase 1:** BGE-Large forward pass works. Finite loss with both MSE and CE. All three math concepts understood.

**End of Phase 2:** All 6 model files complete. All tests pass. K override works. Decoder generates non-garbage text.

**End of Phase 3:** BGE-Large vs DistilBERT comparison complete. BGE-Large shows clear improvement. Ablation configs ready.

**End of Phase 4:** Synthetic pipeline running. Sample quality verified manually. Multi-hop ratio > 70% in combined dataset.

**End of Phase 5:** All three training stages complete without crash. Stage 3 final.pt checkpoint saved. K-scaling shows improvement at preliminary check.

**End of Phase 6:** K-scaling monotonically increasing across all four metrics. Probe accuracy above chance and improving. t-SNE shows partial or full category separation. MSE curve decreasing. All figures saved.

**End of Phase 7:** arXiv paper submitted. GitHub clean and reproducible. README has actual numbers and figures.

---

## The Non-Negotiable Rule

> Write one paragraph of the paper every day from Day 1.  
> Don't wait for results. Section 1 and 2 can be written today.  
> Students who write as they go finish papers.  
> Students who wait until everything is done usually don't.

---

*Start with Day 1. Set up the folder structure. Return here at each phase transition.*  
*The architecture is sound. The math is on your side. Finish it.*
