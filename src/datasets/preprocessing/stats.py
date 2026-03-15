"""
Dataset statistics and quality reporting.
"""

import os
import json
import argparse
from collections import Counter, defaultdict
from typing import List, Dict


def compute_stats(samples: List[Dict]) -> Dict:
    """Compute comprehensive statistics for a dataset."""
    if not samples:
        return {"error": "empty dataset"}

    stats = {}
    stats["total"] = len(samples)
    stats["categories"] = dict(Counter(s.get("category", "unknown") for s in samples))

    if "difficulty_score" in samples[0]:
        stats["difficulty"] = dict(Counter(s["difficulty_score"] for s in samples))
        multihop_count = sum(1 for s in samples if s.get("difficulty_score", 0) >= 1)
        stats["multihop_ratio"] = multihop_count / len(samples)
    else:
        stats["difficulty"] = "not_scored"

    if "answer_type" in samples[0]:
        stats["answer_types"] = dict(Counter(s["answer_type"] for s in samples))

    q_lengths = [len(s["question"].split()) for s in samples]
    a_lengths = [len(s["answer"].split())   for s in samples]

    stats["question_length"] = {
        "min": min(q_lengths), "max": max(q_lengths),
        "mean": sum(q_lengths) / len(q_lengths),
        "median": sorted(q_lengths)[len(q_lengths)//2]
    }
    stats["answer_length"] = {
        "min": min(a_lengths), "max": max(a_lengths),
        "mean": sum(a_lengths) / len(a_lengths),
        "median": sorted(a_lengths)[len(a_lengths)//2]
    }

    stats["empty_questions"] = sum(1 for s in samples if not s.get("question", "").strip())
    stats["empty_answers"]   = sum(1 for s in samples if not s.get("answer", "").strip())

    return stats


def print_stats(stats: Dict, title: str = "Dataset Statistics"):
    """Pretty-print statistics."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"\nTotal samples: {stats.get('total', 0):,}")

    if stats.get("empty_questions", 0) > 0:
        print(f"⚠  Empty questions: {stats['empty_questions']:,}")
    if stats.get("empty_answers", 0) > 0:
        print(f"⚠  Empty answers: {stats['empty_answers']:,}")

    print("\nCategory distribution:")
    cats  = stats.get("categories", {})
    total = stats.get("total", 1)
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {cat:<22} {count:>7,}  ({pct:5.1f}%)  {bar}")

    diff = stats.get("difficulty", {})
    if isinstance(diff, dict):
        print("\nDifficulty distribution:")
        labels = {0: "single-hop", 1: "two-hop", 2: "multi-hop"}
        for d in sorted(diff.keys()):
            count = diff[d]
            pct   = count / total * 100
            bar   = "█" * int(pct / 2)
            print(f"  Score {d} ({labels.get(d,'?'):<12}) {count:>7,}  ({pct:5.1f}%)  {bar}")

        ratio = stats.get("multihop_ratio", 0)
        if ratio < 0.70:
            print(f"\n⚠  Multi-hop ratio is {ratio:.2f} — below recommended 0.70")
        else:
            print(f"\n✓  Multi-hop ratio: {ratio:.2f} (good)")

    atypes = stats.get("answer_types", {})
    if atypes:
        print("\nAnswer type distribution:")
        for atype, count in sorted(atypes.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            print(f"  {atype:<12} {count:>7,}  ({pct:5.1f}%)")

    ql = stats.get("question_length", {})
    al = stats.get("answer_length", {})
    if ql:
        print(f"\nQuestion length (words): min={ql['min']}, max={ql['max']}, "
              f"mean={ql['mean']:.1f}, median={ql['median']}")
    if al:
        print(f"Answer length (words):   min={al['min']}, max={al['max']}, "
              f"mean={al['mean']:.1f}, median={al['median']}")
    print(f"{'='*60}\n")


def print_examples(samples: List[Dict], n_per_category: int = 2):
    """Print example QA pairs from each category."""
    by_category = defaultdict(list)
    for s in samples:
        by_category[s.get("category", "unknown")].append(s)

    print("\nExample QA pairs by category:")
    print("─" * 60)
    for cat, cat_samples in sorted(by_category.items()):
        print(f"\n[{cat.upper()}]")
        for s in cat_samples[:n_per_category]:
            diff = s.get("difficulty_score", "?")
            atype = s.get("answer_type", "?")
            print(f"  Q: {s['question'][:100]}")
            print(f"  A: {s['answer'][:80]}")
            print(f"  Difficulty: {diff} | Answer type: {atype}")
            print()


def check_quality_warnings(stats: Dict) -> List[str]:
    """Return list of quality warnings based on stats."""
    warnings = []
    total = stats.get("total", 0)
    if total == 0:
        warnings.append("CRITICAL: Dataset is empty")
        return warnings

    if stats.get("empty_questions", 0) / total > 0.01:
        warnings.append(f"High empty question rate: {stats['empty_questions']/total:.1%}")
    if stats.get("empty_answers", 0) / total > 0.01:
        warnings.append(f"High empty answer rate: {stats['empty_answers']/total:.1%}")

    ratio = stats.get("multihop_ratio", 1.0)
    if isinstance(ratio, float) and ratio < 0.60:
        warnings.append(f"Low multi-hop ratio: {ratio:.2f} (need >= 0.70)")

    cats = stats.get("categories", {})
    if len(cats) < 3:
        warnings.append(f"Low category diversity: only {len(cats)} categories")

    ql = stats.get("question_length", {})
    if ql and ql.get("mean", 0) < 8:
        warnings.append(f"Questions very short on average ({ql['mean']:.1f} words)")

    return warnings


def load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file into a list of dicts."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return samples


def save_jsonl(samples: List[Dict], path: str):
    """Save list of dicts to JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[stats] Saved {len(samples):,} samples to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset statistics")
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--split", type=str, default="public",
                        choices=["public", "synthetic", "combined"])
    args = parser.parse_args()

    if args.data_file:
        samples = load_jsonl(args.data_file)
        stats   = compute_stats(samples)
        print_stats(stats, title=f"Stats: {args.data_file}")
        print_examples(samples)
        warnings = check_quality_warnings(stats)
        if warnings:
            print("⚠  Quality warnings:")
            for w in warnings:
                print(f"   - {w}")
        else:
            print("✓ No quality warnings. Data looks good.")
    else:
        print("Run with --data_file path/to/data.jsonl to analyze a dataset")