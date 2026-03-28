#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/base.yaml}"
WANDB="${WANDB:-false}"
SEED="${SEED:-42}"

WANDB_FLAG=""
if [ "$WANDB" = "true" ]; then
    WANDB_FLAG="--wandb"
fi

echo "=============================================="
echo "  Think-in-Silence — Full Training Pipeline"
echo "  Config: $CONFIG"
echo "  Seed:   $SEED"
echo "=============================================="

echo ""
echo "[Stage 1] JEPA training..."
python main.py --config "$CONFIG" --seed "$SEED" $WANDB_FLAG

echo ""
echo "[Stage 2] Decoder training..."
python train_decoder.py --config "$CONFIG" --seed "$SEED" $WANDB_FLAG

echo ""
echo "[Stage 3] Joint fine-tuning..."
python finetune.py --config "$CONFIG" --seed "$SEED" $WANDB_FLAG

echo ""
echo "[Eval] Running full evaluation..."
python eval.py --config "$CONFIG" --eval_type all

echo ""
echo "Done. Results saved to results/"
