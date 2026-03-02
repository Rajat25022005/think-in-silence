# Think-in-Silence — Complete Build Plan

> **Profile:** Intermediate ML · Cloud GPU · Deep understanding + Portfolio + Research  
> **Duration:** 28 days across 6 phases

---

## The Big Picture

You are building a model that reasons entirely in vector space — no chain-of-thought tokens,
no reinforcement learning, no decoder. A question gets encoded into a 256-dimensional vector.
A learnable "thought vector" passes through K recurrent attention steps, each silently refining
it by reading the question context. The final thought is projected into a prediction and trained
to match the teacher's answer embedding using simple MSE loss.

The key innovation: K — the number of thinking steps — is adjustable at inference time
without retraining. More steps = more thinking = better answers, provably.

---

## How It Compares to Prior Work

| Feature | Quiet-STaR | Coconut | Think-in-Silence |
|---|---|---|---|
| Reasoning medium | Token vocabulary | LM hidden states | Dedicated latent space |
| Training signal | REINFORCE (RL) | Token supervision | JEPA MSE — no RL |
| CoT labels needed | No | Partial | No |
| K adjustable at inference | No | No | Yes |
| Backbone modified | Yes | Yes | No — fully frozen |

---

## Phase Overview

| Phase | Focus | Days |
|---|---|---|
| 1 | Foundations | 1–3 |
| 2 | Core Architecture | 4–8 |
| 3 | Data Pipeline | 9–11 |
| 4 | Training System | 12–16 |
| 5 | Evaluation & Experiments | 17–22 |
| 6 | Portfolio Polish | 23–28 |

---

# Phase 1 — Foundations
> **Days 1–3 · Goal: Working environment, mathematical intuition, first forward pass**

## Day 1 — Project Setup

### 1.1 Create the Folder Structure First
Before writing a single line of model code, set up the entire project directory.

Top-level folders:
- `src/models/` — all model architecture files
- `src/datasets/` — data loading and preprocessing
- `src/training/` — training loop and utilities
- `src/eval/` — evaluation and visualization
- `configs/` — YAML configuration files
- `checkpoints/` — saved model weights (auto-created at runtime)
- `results/` — plots and metric outputs
- `notebooks/` — experimentation and sanity checks

Root-level files: `main.py`, `eval.py`, `run.sh`, `requirements.txt`.
Each `src/` subfolder needs an `__init__.py` to be importable as a Python package.

### 1.2 Pin Your Dependencies
Create `requirements.txt` with exact version numbers for full reproducibility.

Key libraries:
- **PyTorch 2.2.0** — core deep learning framework
- **Transformers 4.38.0** — for DistilBERT encoder
- **Datasets 2.18.0** — HuggingFace streaming dataset loader
- **PyYAML 6.0.1** — configuration file parsing
- **NumPy 1.26.4** — numerical operations
- **Scikit-learn 1.4.1** — linear probing and t-SNE in evaluation
- **Matplotlib 3.8.3 + Seaborn 0.13.2** — visualization
- **tqdm 4.66.2** — progress bars
- **WandB 0.16.4** — experiment tracking (free tier sufficient)
- **einops 0.7.0** — clean tensor reshape operations

### 1.3 Write the Base Configuration File
Create `configs/base.yaml` — single source of truth for all hyperparameters.
Never hardcode numbers inside model files.

**Model section:** `n_steps: 8`, `proj_dim: 256`, `n_heads: 4` (must divide proj_dim),
`ffn_dim: 512` (2x proj_dim), `dropout: 0.1`, `shared_weights: false`.

**Training section:** `max_steps: 100000`, `batch_size: 64`, `lr: 1.0e-4`,
`warmup_steps: 2000`, `grad_clip: 1.0`, `ema_momentum_start: 0.990`,
`ema_momentum_end: 0.9999`, `log_every: 100`, `ckpt_every: 5000`,
`ckpt_dir: "checkpoints/base"`.

**Eval section:** `k_values: [1, 2, 4, 8, 16]`, `recall_at: [1, 5, 10]`,
`n_probe_samples: 2000`, `tsne_perplexity: 30`.

**Data section:** `max_q_len: 128`, `max_a_len: 64`.

---

## Day 2 — Understand the Core Math
> Do not skip this day. This separates copy-pasting from genuine understanding.

### Concept 1: Embeddings
Text → a fixed-size list of numbers. DistilBERT converts each token into a 768-dim vector.
Similar sentences produce similar vectors. To get one vector per sentence, mean-pool
across all token positions, ignoring padding via the attention mask.

**Action:** Load DistilBERT in a notebook. Pass a sentence through it. Print the shape of
`last_hidden_state`. Mean-pool it. Verify you get shape `(1, 768)`.

### Concept 2: Attention
Formula: `Attention(Q,K,V) = softmax(QKᵀ / √d) × V`
- Q = "what am I looking for?"
- K = "what do I offer?"
- V = "what do I give if selected?"

**Self-attention:** Q, K, V all from the thought. Thought refines itself.
**Cross-attention:** Q from thought, K and V from question context. Thought reads the question.

**Action:** In a notebook, manually compute cross-attention between a small thought vector
`(1, 1, 256)` and short context `(1, 10, 256)`. Print attention weights to see which tokens
are attended to. Understand the shape flow.

### Concept 3: EMA
Formula: `teacher = m × teacher + (1-m) × student`

With m=0.996, teacher moves 0.4% toward student per step — very slowly.
Provides stable regression targets without contrastive negatives or RL.

Momentum schedule: starts at 0.990 (faster updates early), increases to 0.9999
(nearly frozen late). Cosine curve. Implement this correctly.

**Action:** Simulate 100 EMA steps with a toy scalar value. Plot teacher evolution.
Compare momentum=0.99 vs 0.9999 stability.

---

## Day 3 — 50-Line End-to-End Prototype
Build a complete working system in a single notebook cell before writing clean modular code.

Five steps in sequence:
1. Load frozen DistilBERT, disable all its gradients
2. Create two linear projection heads (question → 256-dim sequence, answer → 256-dim pooled)
3. Build a minimal ThoughtBlock with self-attention, cross-attention, and FFN
4. Encode a question, project it, loop through ThoughtBlock K=4 times
5. Project final thought through linear predictor, compute MSE against projected answer,
   call `loss.backward()`

**Success criterion:** Finite loss printed, no gradient errors.

---

# Phase 2 — Core Architecture
> **Days 4–8 · Goal: All 4 model files built, shape-verified, ready to train**

---

## Day 4 — encoder.py

### What This File Does
`FrozenEncoder` wraps DistilBERT with two projection heads — one for question context
(student input) and one for answer embeddings (teacher target).

### Why Freeze the Backbone
DistilBERT already understands language from pre-training on billions of tokens.
Unfreezing wastes compute and risks catastrophic forgetting. Only the two small
projection heads are trainable.

### Two Outputs, Two Purposes
`encode_question` returns the **full token sequence** projected to 256 dims —
shape `(B, seq_len, 256)`. The full sequence is needed so the thought can attend
to individual question words.

`encode_answer` returns a **single mean-pooled vector** projected to 256 dims —
shape `(B, 256)`. One compact vector representing what the answer means.
This is the teacher's regression target.

### What to Implement
- Load DistilBERT, freeze all parameters with `requires_grad = False`
- `question_proj`: `Linear(768, 256)` followed by `LayerNorm(256)`
- `answer_proj`: `Linear(768, 256)` followed by `LayerNorm(256)`
- Decorate the BERT forward with `@torch.no_grad()` — no gradients into backbone
- Mean pooling must correctly ignore padding tokens using the attention mask
- `encode_question` returns full projected sequence
- `encode_answer` returns pooled projected vector

---

## Day 5 — thought_block.py

### What This File Does
`ThoughtBlock` is one reasoning step. Takes thought `h` and question context `ctx`,
returns refined thought `h`. The fundamental unit of latent reasoning.

### Architecture: Pre-LayerNorm
Apply LayerNorm *before* each sub-layer, not after. Modern standard (GPT-3, PaLM).
Gradients flow more cleanly through residual connections.

Three sub-layers, each with residual connection `h = h + sublayer_output`:

1. **Self-Attention:** Normalize h → self-attention (Q=K=V=h) → add back.
   Thought refines its own internal representation.

2. **Cross-Attention:** Normalize h and ctx → cross-attention (Q=h, K=V=ctx) → add back.
   Thought reads and absorbs information from the question.

3. **FFN:** Normalize h → MLP (Linear → GELU → Dropout → Linear → Dropout) → add back.
   Mixes and transforms information across the dimension axis.

### What to Implement
- Three separate `LayerNorm` layers, one per sub-layer
- `MultiheadAttention` for self-attention with `batch_first=True`
- `MultiheadAttention` for cross-attention with `batch_first=True`
- FFN: `Linear(dim, ffn_dim)` → `GELU()` → `Dropout` → `Linear(ffn_dim, dim)` → `Dropout`
- Input shapes: `h: (B, 1, dim)`, `ctx: (B, seq_len, dim)`, output `h: (B, 1, dim)`
- Use GELU not ReLU — smoother, better for transformers

---

## Day 6 — thought_module.py

### What This File Does
`ThoughtModule` chains K ThoughtBlocks together. Starts from learnable `h0`,
recurrently applies each block, passing `hₖ` as input to the next step.

### The Learnable h0
`nn.Parameter` of shape `(1, 1, dim)`, initialized with scale 0.02.
The model's "blank slate" prior before reasoning. Learned by backpropagation.
At runtime: `h0.expand(B, -1, -1)` to match batch size.

### Two Modes
**`shared_weights=False`** (default): K separate blocks. Each step specializes.
More expressive, more parameters. Recommended default.

**`shared_weights=True`**: One block applied K times. Universal Transformer variant.
Fewer parameters. Run as an ablation experiment.

### Inference-Time K Override
`forward` accepts optional `n_steps` argument that overrides trained K.
Blocks reused cyclically via modulo when `n_steps > len(self.blocks)`.
This is what enables K=16 evaluation on a K=8 checkpoint with zero retraining.

### Return All States
When `return_all_states=True`, store h after every step.
Return stacked tensor `(B, K+1, dim)` — K+1 because h0 is step 0.
Required for probing and t-SNE experiments in Phase 5.

---

## Day 7 — lc_thought.py

### What This File Does
`LCThought` is the complete model — student-teacher system fully assembled.

### Student Path
1. Encode question → context sequence `(B, seq_len, 256)`
2. Run ThoughtModule K steps → final thought `(B, 1, 256)`
3. Squeeze → predictor MLP → prediction `(B, 256)`

### Teacher Path
1. Encode answer → pooled vector `(B, 256)`
2. Wrapped in `torch.no_grad()`
3. This is the regression target

### The Predictor MLP
`Linear(256, 512)` → `GELU()` → `LayerNorm(512)` → `Linear(512, 256)`
The bottleneck prevents shortcuts — the model can't bypass reasoning and just
copy encoder output directly.

### The Loss
`mse_loss(pred, target.detach())` — `.detach()` is critical. Gradients must
never flow backward into the teacher.

### EMA Update Method
Implement as `update_ema(momentum)` decorated with `@torch.no_grad()`.
Iterate over student/teacher parameter pairs and apply EMA formula.
Called from training loop after every optimizer step.

### Teacher Initialization
`copy.deepcopy()` both `thought_module` and `predictor`.
Then set `requires_grad = False` on all teacher parameters.

---

## Day 8 — Shape Verification
Verify the full forward pass with a hardcoded 2-sample fake batch before touching data.

Checks to pass:
- `out['loss']` is a finite scalar (not NaN, not Inf)
- `out['pred']` shape is `(2, 256)`
- `out['target']` shape is `(2, 256)`
- `out['all_states']` shape is `(2, 9, 256)` when `return_all_states=True`, K=8 (K+1 steps)
- `model.update_ema(0.996)` runs without error
- `n_steps=16` on a K=8 model runs without error

All checks pass → Phase 2 complete.

---

# Phase 3 — Data Pipeline
> **Days 9–11 · Goal: Streaming interleaved loader for all 5 datasets, verified and working**

---

## Day 9–10 — qa_datasets.py

### The 5 Datasets

| Dataset | HF Name | Task | Size | Weight |
|---|---|---|---|---|
| HotpotQA | `hotpot_qa` | Multi-hop factual | 113K | 35% |
| GSM8K | `gsm8k` | Grade-school math | 8.5K | 20% |
| CommonsenseQA | `commonsense_qa` | Commonsense MC | 12K | 20% |
| ARC-Challenge | `ai2_arc` | Science exam | 7.8K | 15% |
| StrategyQA | `allenai/strategy_qa` | Yes/no multi-hop | 2.8K | 10% |

Different tasks force the model to learn generalizable latent reasoning, not a narrow skill.

### Why Streaming
Use `streaming=True` in `load_dataset()`. Data fetched on-demand — no bulk downloads,
no storage requirements, no waiting before training starts.

### The Normalization Problem
Each dataset stores answers differently. Write a custom extractor for each, normalizing
output to `{"question": str, "answer": str, "category": str}`.

Key edge cases:
- GSM8K: answers end with `"#### 120"` — split on `####`, take right side, strip whitespace
- CommonsenseQA + ARC: multiple choice — use `answerKey` to index into choices list
- ARC: sometimes numeric keys ("1","2") instead of letter keys ("A","B") — handle both
- StrategyQA: boolean answers — convert `True/False` → `"yes"/"no"`

### Interleaving Strategy
Use `random.choices()` with dataset weights to randomly select which dataset to sample from.
When an iterator is exhausted, reset it and continue.
This maintains correct distribution throughout training.

### The DataLoader
`DataLoader` with `batch_size=64`, `num_workers=2`, `pin_memory=True`.
Use `padding="max_length"` and `truncation=True` for consistent batch shapes.
Each item yields: `q_ids`, `q_mask`, `a_ids`, `a_mask`, `category`.

---

## Day 11 — Verify the Pipeline

Pull a batch and check:
- Decode `q_ids` → looks like a real question
- Decode `a_ids` → looks like the matching answer
- `category` field matches correct dataset source
- `q_ids` shape is `(batch_size, 128)`, `a_ids` is `(batch_size, 64)`
- Sample 10+ batches — all 5 categories appear, roughly matching weights

Print 3 decoded examples from different categories before spending 6 hours training.

---

# Phase 4 — Training System
> **Days 12–16 · Goal: Full loop with EMA scheduling, mixed precision, logging, checkpointing**

---

## Day 12–14 — trainer.py

### Learning Rate: Cosine with Linear Warmup
Linear warmup for first 2000 steps (0 → target LR).
Then cosine decay: `lr × 0.5 × (1 + cos(π × progress))` where progress ∈ [0, 1].
Update by modifying `optimizer.param_groups` at the start of every step.

### EMA Momentum Schedule
Cosine increase from 0.990 → 0.9999 over full training.
Formula: `end - (end - start) × (cos(π × progress) + 1) / 2`

- Step 0: momentum ≈ 0.990 — teacher updates quickly when student is changing fast
- Step 100k: momentum ≈ 0.9999 — teacher nearly frozen, providing stable fine-grained targets

### Optimizer
`AdamW` with `weight_decay=0.04`, `betas=(0.9, 0.95)`.
Applied only to student parameters — explicitly list them.
Include: `thought_module`, `predictor`, `encoder.question_proj`, `encoder.answer_proj`.
Never include teacher parameters or frozen DistilBERT.

### Mixed Precision (BF16)
Wrap forward pass in `torch.cuda.amp.autocast(dtype=torch.bfloat16)`.
Use `GradScaler` for backward pass.
Halves memory, significantly speeds up training on GCP L4 and A100.
On older GPUs: switch to FP16 but watch for NaN gradients more carefully.

### Gradient Clipping
`clip_grad_norm_(student_params, max_norm=1.0)` before every optimizer step.
Unscale gradients first when using GradScaler.
Prevents exploding gradients common early in transformer training.

### WandB Logging
Log every 100 steps: `loss`, `lr`, `ema_momentum`.
These three curves diagnose all common training failures.

### Checkpointing
Save every 5000 steps: `step`, `model_state_dict`, `optimizer_state_dict`, `cfg`.
Include optimizer state to allow training resumption.
Save final as `final.pt`.

### Per-Step Order of Operations
1. Get batch → move to GPU
2. Update LR in optimizer param groups
3. Forward pass under autocast
4. Zero gradients
5. Backward via scaled loss
6. Unscale gradients
7. Clip gradients
8. Optimizer step
9. Scaler update
10. EMA teacher update
11. Log to WandB (every 100)
12. Save checkpoint (every 5000)

---

## Day 15–16 — main.py and run.sh

### main.py (~10 lines)
Load YAML config → instantiate model → instantiate dataloader using
`model.encoder.tokenizer` → call `train(model, loader, cfg)`.
No training logic lives here.

### run.sh
Set `set -e` at top for immediate failure on errors.
Install requirements quietly → `nvidia-smi` to verify GPU → run
`python main.py 2>&1 | tee training_log.txt`.

---

# Phase 5 — Evaluation & Experiments
> **Days 17–22 · Goal: 4 experiments proving the model reasons — your research contribution**

---

## Experiment 1 — K-Scaling (Most Important)

### The Claim
Genuine reasoning → accuracy increases with more thinking steps.
This is the central claim of the paper.

### Setup
Load a single checkpoint. No retraining. Evaluate at K = 0, 1, 2, 4, 8, 16.
Compute Recall@1 and Recall@5 on a held-out test set for each K.

### Recall@K Explained
Pool of candidate answer embeddings (one per test question).
Model produces a predicted answer vector.
Recall@1 = correct answer is nearest neighbor to prediction.
Recall@5 = correct answer is in top 5 nearest neighbors.

### K=0 Baseline
No reasoning — predictor receives only untransformed h0.
Should perform worst by a clear margin.

### Expected Result Table

| K Steps | Recall@1 | Recall@5 |
|:---:|:---:|:---:|
| K = 0 | — | — |
| K = 1 | — | — |
| K = 2 | — | — |
| K = 4 | — | — |
| K = 8 | — | — |
| K = 16 | — | — |

Monotonically increasing curve = genuine latent reasoning confirmed.

---

## Experiment 2 — Linear Probing

### The Claim
Thought vectors spontaneously encode semantically meaningful information —
without any supervision on category labels during training.

### Setup
At each step hₖ (0 to K), train `LogisticRegression` from sklearn to predict
question category (math / factual_multihop / commonsense / science / strategy)
from the thought vector alone.

Also probe for answer type (numeric / yes-no / entity) as secondary experiment.

### How to Interpret
Chance accuracy at step 0, increasing through step K =
model spontaneously encodes category-relevant information during reasoning.

### What to Plot
Line chart: step (x-axis) vs probe accuracy (y-axis). Rising then plateauing curve.

---

## Experiment 3 — MSE Curve per Step

### The Claim
Each reasoning step brings the prediction closer to the correct answer embedding.

### Setup
For each step k from 1 to K: extract intermediate thought at step k, pass through
predictor, compute MSE against teacher target. Plot as curve.

### Expected Pattern
Decreasing curve: high MSE at step 1, progressively lower through K, diminishing returns
near K. Flat or increasing curve = debug gradient flow and EMA update.

---

## Experiment 4 — t-SNE Thought Trajectories

### The Claim
Different question categories cause different reasoning paths through latent space.

### Setup
Collect all intermediate states h0...hK for test questions across all 5 categories.
Flatten to `(N × (K+1), 256)`. Run t-SNE to project to 2D.
Reshape back to trajectories `(N, K+1, 2)`.
Plot arrows from h0 to hK colored by category.

### What to Look For
Distinct colored clusters. Same category → similar trajectory.
Math questions cluster separately from commonsense questions.
Mixed with no structure → model hasn't learned category-specific reasoning.

### Visualization Tips
- Dark matplotlib background
- Semi-transparent arrows for overlapping visibility
- Small dot at h0, arrowhead at hK
- Clear legend with distinct colors
- This is your most visually striking result for README and paper

---

# Phase 6 — Portfolio Polish
> **Days 23–28 · Goal: Clean GitHub, paper draft, reproducible results**

---

## Day 23–25 — GitHub Repository

### README Story Order
1. The idea — open with "Most language models think out loud. This one doesn't."
2. Architecture diagram — ASCII art showing question → ThoughtBlocks → prediction
3. Comparison table — three-way comparison vs Quiet-STaR and Coconut
4. Key results — filled K-scaling table, t-SNE image, MSE curve image
5. Quickstart — install and train in under 5 lines
6. Project structure — annotated file tree
7. Configuration guide — key YAML settings explained
8. Citation — BibTeX for arXiv submission

### Reproducibility Checklist
- Random seeds set in `main.py` — `torch.manual_seed` and `numpy.random.seed`
- Evaluation checkpoint downloadable (HuggingFace Hub or Google Drive)
- Config file saved inside checkpoint state dict
- Eval scripts runnable in a single command with documented arguments
- `requirements.txt` tested from a fresh virtual environment
- All README result images generated from the provided checkpoint

---

## Day 26–28 — Paper Writeup

### Abstract (4 sentences)
State the problem (token-level reasoning is expensive and not always necessary).
State your approach (K recurrent cross-attention steps in 256-dim latent space,
trained with JEPA MSE objective).
State the key result (accuracy improves monotonically with K using same checkpoint,
no retraining required).
State the significance (no RL, no CoT labels, backbone fully frozen).

### Section 1 — Introduction
Motivate latent reasoning. Limitations of token-level CoT.
End with numbered contributions list: (1) separate ThoughtModule without backbone modification,
(2) inference-time K scaling, (3) pure Q&A training — no reasoning traces needed.

### Section 2 — Related Work
Three prior works with clear differentiation:
- **I-JEPA** — origin of the JEPA training objective, extended here to language reasoning
- **Coconut** — most similar, but requires backbone modification and partial token supervision
- **Quiet-STaR** — hidden reasoning but uses REINFORCE and modifies backbone

### Section 3 — Architecture
Full pipeline with equations: attention formula, EMA update rule, JEPA MSE objective.
Student-teacher diagram. Justify each design choice: pre-LN, frozen backbone,
predictor bottleneck, cosine EMA scheduling, shared vs separate blocks.

### Section 4 — Experiments
All 4 experiments with tables and figures. For each: hypothesis → setup → result → interpretation.

### Section 5 — Conclusion
Summarize results. Three concrete future directions:
1. Scaling to larger backbone (BERT-large or sentence-transformers)
2. Extending to generation tasks beyond retrieval
3. Testing longer chains (K > 16) and continued accuracy scaling

---

# Troubleshooting Reference

| Problem | Symptom | Fix |
|---|---|---|
| Out of Memory | CUDA OOM during training | Reduce batch_size to 32 or 16. Add gradient checkpointing. |
| Loss not decreasing | Stuck same MSE after 10K steps | Try lr=3e-4. Lower ema_momentum_start to 0.98. |
| Representation collapse | All predictions identical vectors | Lower ema_momentum_start to 0.98. Reduce predictor capacity. |
| Streaming hangs | DataLoader freezes on first batch | Set num_workers=0 to debug. Check datasets version. |
| NaN gradients | Loss suddenly NaN | Lower grad_clip to 0.5. Try FP16 instead of BF16. |
| K=16 worse than K=8 | Accuracy degrades beyond trained K | Try shared_weights: true. Train for more steps. |
| Probe accuracy flat | No improvement across steps | Verify return_all_states captures intermediate states correctly. |
| t-SNE fully mixed | No visible category clusters | Increase n_probe_samples. Try perplexity 15 or 50. |

---

# Milestones & Success Criteria

**End of Phase 1:** Prototype runs end-to-end. Finite loss. Gradients flow.
All three math concepts understood and verified in notebook.

**End of Phase 2:** All 4 files complete. Shape verification passes. EMA update works.
K override at inference works without errors.

**End of Phase 3:** Dataloader produces correct batches. All 5 categories visible.
Decoded examples look like real QA pairs.

**End of Phase 4:** Training runs 1000 steps without crash. Loss decreasing in WandB.
LR warmup curve correct. Checkpoints being saved.

**End of Phase 5:** K-scaling shows monotonically increasing accuracy. Probe accuracy
above chance and improves across steps. MSE curve decreasing.
t-SNE shows at least partial category separation.

**End of Phase 6:** GitHub clean, documented, reproducible from scratch.
Paper draft covers all 5 sections. Results honest about limitations.

---

*Start with Day 1. Set up the folder structure and run the sanity-check notebook.
Return here at each phase transition.*
