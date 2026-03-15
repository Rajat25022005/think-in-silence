"""
Filter QA pairs by reasoning difficulty.

Difficulty levels:
    0 = single-hop (one fact lookup)
    1 = two-hop (connect two facts)
    2 = multi-hop (3+ fact connections)
"""

import re
from typing import List, Dict, Tuple


MULTIHOP_STRONG = [
    "also", "both", "neither", "either", "same",
    "after", "before", "during", "following", "prior to",
    "later", "eventually", "subsequently",
    "because", "therefore", "resulted in", "led to", "caused",
    "due to", "as a result",
    "who later", "which also", "that also", "where he", "where she",
    "where they", "where it",
    "how many", "total number", "combined", "sum of",
    "older than", "younger than", "more than", "less than",
    "longer than", "shorter than", "before or after",
]

MULTIHOP_WEAK = [
    "which", "whose", "what did", "where did", "when did",
    "who was", "who is", "what was", "what is the name",
]

MULTIHOP_PATTERNS = [
    r"the \w+ who \w+",
    r"in the same \w+ as",
    r"both .+ and .+",
    r"the \w+ of the \w+ that",
    r"(before|after) (he|she|they|it) \w+",
]

SINGLE_HOP_PATTERNS = [
    r"^what (is|was|are|were) (a|an|the) \w+\?$",
    r"^what (is|was) [A-Z]\w+\?$",
    r"^who (is|was) [A-Z][a-z]+ [A-Z][a-z]+\?$",
]


def compute_difficulty(question: str, answer: str = "") -> int:
    """
    Compute difficulty score for a QA pair.

    Returns:
        0 = single-hop, 1 = two-hop, 2 = multi-hop
    """
    q = question.lower()

    strong_count = sum(1 for sig in MULTIHOP_STRONG if sig in q)
    weak_count   = sum(1 for sig in MULTIHOP_WEAK if sig in q)
    pattern_count = sum(1 for p in MULTIHOP_PATTERNS if re.search(p, q))
    is_likely_single = any(re.match(p, q) for p in SINGLE_HOP_PATTERNS)

    score = 0

    if strong_count >= 2 or pattern_count >= 1:
        score = 2
    elif strong_count >= 1 or (weak_count >= 2 and not is_likely_single):
        score = 1
    elif weak_count >= 1 and not is_likely_single:
        score = 1

    word_count = len(q.split())
    if word_count > 30 and score < 2:
        score = min(score + 1, 2)

    if is_likely_single and strong_count == 0:
        score = 0

    return score


def classify_answer_type(answer: str) -> str:
    """
    Classify the type of answer.

    Returns one of: numeric, boolean, entity, date, phrase, other
    """
    a = answer.strip().lower()

    if not a:
        return "other"

    if a in ("yes", "no", "true", "false"):
        return "boolean"

    if re.match(r'^[\d,.\s]+$', a) or re.match(r'^\$?[\d,]+(\.\d+)?[km%]?$', a):
        return "numeric"

    if re.match(r'^\d{4}$', a) or re.match(r'\d{1,2}/\d{1,2}/\d{2,4}', a):
        return "date"

    words = a.split()
    if (1 <= len(words) <= 4 and
        sum(1 for w in words if w and w[0].isupper()) >= len(words) * 0.6):
        return "entity"

    if len(words) <= 8:
        return "phrase"

    return "other"


def filter_by_difficulty(
    samples: List[Dict],
    min_difficulty: int = 1,
    target_distribution: Dict[int, float] = None
) -> Tuple[List[Dict], Dict]:
    """
    Filter samples to keep only those with difficulty >= min_difficulty.

    Args:
        samples:              List of QA dicts
        min_difficulty:       Minimum difficulty to keep
        target_distribution:  Optional target distribution

    Returns:
        (filtered_samples, stats_dict)
    """
    scored = []
    for s in samples:
        score = compute_difficulty(s["question"], s.get("answer", ""))
        atype = classify_answer_type(s.get("answer", ""))
        s = s.copy()
        s["difficulty_score"] = score
        s["answer_type"]      = atype
        scored.append(s)

    dist_before = {0: 0, 1: 0, 2: 0}
    for s in scored:
        dist_before[s["difficulty_score"]] += 1

    print(f"[difficulty_filter] Distribution before filtering:")
    for d, count in dist_before.items():
        pct = count / len(scored) * 100 if scored else 0
        print(f"  Score {d}: {count:,} ({pct:.1f}%)")

    filtered = [s for s in scored if s["difficulty_score"] >= min_difficulty]

    dist_after = {0: 0, 1: 0, 2: 0}
    for s in filtered:
        dist_after[s["difficulty_score"]] += 1

    retention = len(filtered) / len(samples) * 100 if samples else 0
    print(f"\n[difficulty_filter] After filtering (min_difficulty={min_difficulty}):")
    print(f"  Kept: {len(filtered):,} ({retention:.1f}%)")
    for d, count in dist_after.items():
        pct = count / len(filtered) * 100 if filtered else 0
        print(f"  Score {d}: {count:,} ({pct:.1f}%)")

    stats = {
        "input_count":      len(samples),
        "output_count":     len(filtered),
        "retention_pct":    retention,
        "dist_before":      dist_before,
        "dist_after":       dist_after,
    }

    return filtered, stats


def enforce_multihop_ratio(
    samples: List[Dict],
    target_multihop_ratio: float = 0.70
) -> List[Dict]:
    """
    Ensure at least target_multihop_ratio of samples are multi-hop (score >= 1).

    Args:
        samples:               List of scored QA dicts (must have difficulty_score)
        target_multihop_ratio: Target fraction of multi-hop samples

    Returns:
        Rebalanced sample list
    """
    import random

    multihop  = [s for s in samples if s.get("difficulty_score", 0) >= 1]
    singlehop = [s for s in samples if s.get("difficulty_score", 0) == 0]

    current_ratio = len(multihop) / len(samples) if samples else 0

    if current_ratio >= target_multihop_ratio:
        print(f"[difficulty_filter] Multi-hop ratio {current_ratio:.2f} already "
              f">= target {target_multihop_ratio:.2f}. No rebalancing needed.")
        return samples

    keep_single = int(len(multihop) * (1 - target_multihop_ratio) / target_multihop_ratio)
    kept_single = random.sample(singlehop, min(keep_single, len(singlehop)))

    result = multihop + kept_single
    random.shuffle(result)

    new_ratio = len(multihop) / len(result) if result else 0
    print(f"[difficulty_filter] Rebalanced: {len(result):,} samples, "
          f"multi-hop ratio = {new_ratio:.2f}")

    return result