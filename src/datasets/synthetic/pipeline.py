"""
End-to-end synthetic data generation pipeline.

Orchestrates: corpus reading → paragraph splitting → QA generation →
filtering → difficulty scoring → deduplication → JSONL output.
"""

import os
import json
import re
import argparse
import time
from pathlib import Path
from typing import List, Dict, Iterator
from tqdm import tqdm

from src.datasets.synthetic.generator  import get_generator
from src.datasets.synthetic.filter     import filter_dataset
from src.datasets.preprocessing.cleaner import clean_dataset
from src.datasets.preprocessing.difficulty_filter import (
    filter_by_difficulty, compute_difficulty, classify_answer_type
)
from src.datasets.preprocessing.stats import save_jsonl, load_jsonl, compute_stats, print_stats


def read_corpus_files(corpus_dir: str) -> Iterator[tuple]:
    """
    Recursively read text files from a corpus directory.
    Supports: .txt, .md, .json, .jsonl

    Yields:
        (filename, text_content)
    """
    corpus_path = Path(corpus_dir)
    text_extensions = {".txt", ".md"}
    json_extensions = {".json", ".jsonl"}

    for filepath in corpus_path.rglob("*"):
        if not filepath.is_file():
            continue

        ext = filepath.suffix.lower()

        if ext in text_extensions:
            try:
                text = filepath.read_text(encoding="utf-8", errors="ignore")
                if len(text.strip()) > 100:
                    yield str(filepath.name), text
            except Exception:
                continue

        elif ext in json_extensions:
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            text = (obj.get("text") or obj.get("content") or
                                    obj.get("passage") or obj.get("document") or "")
                            if isinstance(text, str) and len(text.strip()) > 100:
                                yield str(filepath.name), text.strip()
                        except Exception:
                            continue
            except Exception:
                continue


def split_into_paragraphs(
    text: str,
    min_words: int = 40,
    max_words: int = 200
) -> List[str]:
    """
    Split text into paragraphs suitable for QA generation.

    Merges short paragraphs, truncates long ones, skips non-prose content.
    """
    raw_paragraphs = re.split(r'\n\s*\n', text)

    paragraphs = []
    buffer = ""

    for para in raw_paragraphs:
        para = " ".join(para.split())
        if not para:
            continue

        alpha_ratio = sum(c.isalpha() for c in para) / max(len(para), 1)
        if alpha_ratio < 0.6:
            continue

        word_count = len(para.split())

        if word_count < min_words:
            buffer = (buffer + " " + para).strip()
            if len(buffer.split()) >= min_words:
                paragraphs.append(buffer[:max_words * 6])
                buffer = ""
        else:
            if buffer:
                paragraphs.append(buffer[:max_words * 6])
                buffer = ""
            words = para.split()
            if len(words) > max_words:
                para = " ".join(words[:max_words])
            paragraphs.append(para)

    if buffer:
        paragraphs.append(buffer[:max_words * 6])

    return paragraphs


class ProgressTracker:
    """Track which files have been processed for crash resumability."""

    def __init__(self, progress_file: str):
        self.path       = progress_file
        self.processed  = self._load()

    def _load(self) -> set:
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                return set(json.load(f))
        return set()

    def mark_done(self, filename: str):
        self.processed.add(filename)
        with open(self.path, "w") as f:
            json.dump(list(self.processed), f)

    def is_done(self, filename: str) -> bool:
        return filename in self.processed


def run_pipeline(
    corpus_dir:     str,
    output_dir:     str,
    generator_type: str  = "local",
    target_samples: int  = 500_000,
    n_questions_per_para: int = 3,
    min_difficulty:      int = 1,
    test_run:           bool = False
) -> str:
    """
    Run the full synthetic data generation pipeline.

    Args:
        corpus_dir:          Directory containing raw text files
        output_dir:          Where to save generated JSONL files
        generator_type:      "local" | "claude" | "gpt"
        target_samples:      Stop after this many clean samples
        n_questions_per_para: Questions to generate per paragraph
        min_difficulty:      Minimum difficulty score to keep
        test_run:            If True, process only 100 paragraphs

    Returns:
        Path to final combined JSONL file
    """
    os.makedirs(output_dir, exist_ok=True)

    batch_dir      = os.path.join(output_dir, "batches")
    os.makedirs(batch_dir, exist_ok=True)

    progress_file  = os.path.join(output_dir, "progress.json")
    tracker        = ProgressTracker(progress_file)

    output_file    = os.path.join(output_dir, "synthetic_qa.jsonl")

    generator = get_generator(generator_type)

    print(f"\n[pipeline] Starting synthetic data generation")
    print(f"  Corpus:    {corpus_dir}")
    print(f"  Output:    {output_file}")
    print(f"  Generator: {generator_type}")
    print(f"  Target:    {target_samples:,} samples")
    print(f"  Test run:  {test_run}")

    existing = 0
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            existing = sum(1 for line in f if line.strip())
        print(f"[pipeline] Resuming: {existing:,} samples already generated")

    total_generated = existing
    total_processed = 0
    errors          = 0

    pbar = tqdm(
        desc="Generating",
        unit="pairs",
        initial=existing,
        total=target_samples
    )

    with open(output_file, "a", encoding="utf-8") as out_f:

        for filename, text in read_corpus_files(corpus_dir):

            if total_generated >= target_samples:
                break

            if test_run and total_processed >= 100:
                break

            if tracker.is_done(filename):
                continue

            paragraphs = split_into_paragraphs(text)

            batch_samples = []

            for para in paragraphs:
                if total_generated >= target_samples:
                    break

                try:
                    raw_pairs = generator.generate(para, n_questions=n_questions_per_para)

                    for pair in raw_pairs:
                        pair["category"]   = "synthetic"
                        pair["source_file"] = filename

                    filtered, _ = filter_dataset(raw_pairs, verbose=False)

                    for pair in filtered:
                        pair["difficulty_score"] = compute_difficulty(
                            pair["question"], pair["answer"]
                        )
                        pair["answer_type"] = classify_answer_type(pair["answer"])

                    kept = [p for p in filtered
                            if p["difficulty_score"] >= min_difficulty]

                    for pair in kept:
                        out_f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                        total_generated += 1
                        pbar.update(1)
                        batch_samples.append(pair)

                    total_processed += 1

                except Exception as e:
                    errors += 1
                    if errors % 100 == 0:
                        print(f"\n[pipeline] {errors} errors so far. "
                              f"Last error: {str(e)[:100]}")
                    continue

            tracker.mark_done(filename)
            out_f.flush()

    pbar.close()

    print(f"\n[pipeline] Complete!")
    print(f"  Total generated: {total_generated:,}")
    print(f"  Paragraphs processed: {total_processed:,}")
    print(f"  Errors: {errors:,}")
    print(f"  Output: {output_file}")

    samples = load_jsonl(output_file)
    if samples:
        stats = compute_stats(samples)
        print_stats(stats, title="Synthetic Dataset Statistics")

    return output_file


def merge_with_public(
    synthetic_file: str,
    output_file:    str,
    synthetic_ratio: float = 0.6,
    total_target:   int = 500_000
) -> str:
    """
    Merge synthetic and public dataset samples.

    Args:
        synthetic_file:  Path to synthetic JSONL
        output_file:     Output path for combined dataset
        synthetic_ratio: Fraction from synthetic (0.6 = 60%)
        total_target:    Total samples in merged dataset

    Returns:
        Path to merged file
    """
    import random

    synthetic_samples = load_jsonl(synthetic_file)

    from datasets import load_dataset

    public_samples = []
    for ds_config in [
        ("hotpot_qa",           "distractor", "train"),
        ("gsm8k",               "main",       "train"),
        ("commonsense_qa",       None,         "train"),
        ("ai2_arc",             "ARC-Challenge", "train"),
        ("allenai/strategy_qa",  None,         "train"),
        ("rajat5039/wiki-multihop-qa-500k", None, "train"),
    ]:
        name, config, split = ds_config
        kwargs = {"split": split}
        if config:
            kwargs["name"] = config
        try:
            ds = load_dataset(name, **kwargs)
            for sample in ds:
                public_samples.append(sample)
        except Exception as e:
            print(f"[merge] Could not load {name}: {e}")

    n_synthetic = int(total_target * synthetic_ratio)
    n_public    = total_target - n_synthetic

    synthetic_chosen = random.sample(
        synthetic_samples, min(n_synthetic, len(synthetic_samples))
    )
    public_chosen = random.sample(
        public_samples, min(n_public, len(public_samples))
    )

    combined = synthetic_chosen + public_chosen
    random.shuffle(combined)

    save_jsonl(combined, output_file)
    print(f"[merge] Combined dataset: {len(combined):,} samples "
          f"({len(synthetic_chosen):,} synthetic + {len(public_chosen):,} public)")

    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic QA Generation Pipeline")
    parser.add_argument("--corpus_dir",   type=str, required=True)
    parser.add_argument("--output_dir",   type=str, default="data/synthetic")
    parser.add_argument("--generator",    type=str, default="local",
                        choices=["local", "claude", "gpt"])
    parser.add_argument("--target_samples", type=int, default=500_000)
    parser.add_argument("--min_difficulty", type=int, default=1)
    parser.add_argument("--test_run",    action="store_true",
                        help="Process only 100 paragraphs for testing")
    args = parser.parse_args()

    run_pipeline(
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        generator_type=args.generator,
        target_samples=args.target_samples,
        min_difficulty=args.min_difficulty,
        test_run=args.test_run
    )