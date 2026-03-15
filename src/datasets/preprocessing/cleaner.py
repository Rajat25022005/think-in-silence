"""
Normalize and deduplicate QA pairs from any source.
"""

import re
import unicodedata
from typing import List, Dict, Tuple, Set
from tqdm import tqdm


def normalize_text(text: str) -> str:
    """Clean raw text: unicode normalize, strip whitespace, fix quotes."""
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f\u200b-\u200f\ufeff]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = text.replace('\u2019', "'").replace('\u2018', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')

    return text


def normalize_question(q: str) -> str:
    """Normalize a question string. Ensures question mark and capitalization."""
    q = normalize_text(q)
    if not q:
        return ""

    q = q[0].upper() + q[1:]

    if q and q[-1] not in "?!.":
        q = q + "?"

    return q


def normalize_answer(a: str) -> str:
    """Normalize an answer string."""
    a = normalize_text(a)
    if not a:
        return ""

    a = a[0].upper() + a[1:]

    return a


def is_valid_pair(q: str, a: str) -> Tuple[bool, str]:
    """
    Check if a QA pair is valid for training.

    Returns:
        (is_valid, rejection_reason)
    """
    if not q:
        return False, "empty_question"
    if len(q) < 15:
        return False, "question_too_short"
    if len(q) > 500:
        return False, "question_too_long"
    if not any(c.isalpha() for c in q):
        return False, "question_no_alpha"

    if not a:
        return False, "empty_answer"
    if len(a) < 1:
        return False, "answer_too_short"
    if len(a) > 300:
        return False, "answer_too_long"
    if not any(c.isalpha() or c.isdigit() for c in a):
        return False, "answer_no_content"

    q_lower = q.lower().strip("?").strip()
    a_lower = a.lower()
    if q_lower in a_lower and len(a) < len(q) + 20:
        return False, "answer_repeats_question"

    return True, ""


def exact_dedup(samples: List[Dict]) -> List[Dict]:
    """Remove exact duplicate questions using a hash set."""
    seen:  Set[str] = set()
    clean: List[Dict] = []

    for s in samples:
        key = s["question"].lower().strip()
        if key not in seen:
            seen.add(key)
            clean.append(s)

    removed = len(samples) - len(clean)
    if removed > 0:
        print(f"[cleaner] Exact dedup: removed {removed:,} duplicates "
              f"({len(clean):,} remain)")
    return clean


def _token_overlap(a: str, b: str) -> float:
    """Jaccard similarity on word tokens. Returns 0.0 to 1.0."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union        = tokens_a | tokens_b
    return len(intersection) / len(union)


def near_dedup(
    samples: List[Dict],
    threshold: float = 0.85,
    max_comparisons: int = 50
) -> List[Dict]:
    """
    Remove near-duplicate questions using token overlap.

    Args:
        samples:         List of QA dicts
        threshold:       Jaccard similarity threshold
        max_comparisons: How many recent samples to compare against
    """
    kept:   List[Dict] = []
    window: List[str]  = []
    removed = 0

    for sample in tqdm(samples, desc="Near-dedup", miniters=10000):
        q = sample["question"].lower()

        is_near_dup = False
        for prev_q in window[-max_comparisons:]:
            if _token_overlap(q, prev_q) > threshold:
                is_near_dup = True
                removed += 1
                break

        if not is_near_dup:
            kept.append(sample)
            window.append(q)

    if removed > 0:
        print(f"[cleaner] Near-dedup: removed {removed:,} near-duplicates "
              f"({len(kept):,} remain)")
    return kept


def clean_dataset(
    samples: List[Dict],
    dedup: bool = True,
    near_dedup_threshold: float = 0.85
) -> Tuple[List[Dict], Dict]:
    """
    Full cleaning pipeline for a list of QA pairs.

    Args:
        samples:               List of {"question": str, "answer": str, ...}
        dedup:                 Whether to run deduplication
        near_dedup_threshold:  Jaccard threshold for near-dedup

    Returns:
        (cleaned_samples, stats_dict)
    """
    stats = {
        "input_count":    len(samples),
        "rejected":       {},
        "exact_dupes":    0,
        "near_dupes":     0,
        "output_count":   0,
    }

    normalized = []
    for s in samples:
        s = s.copy()
        s["question"] = normalize_question(s.get("question", ""))
        s["answer"]   = normalize_answer(s.get("answer", ""))
        normalized.append(s)

    valid = []
    for s in normalized:
        ok, reason = is_valid_pair(s["question"], s["answer"])
        if ok:
            valid.append(s)
        else:
            stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1

    print(f"[cleaner] Validation: {len(valid):,}/{len(samples):,} passed "
          f"({len(samples)-len(valid):,} rejected)")

    if not valid:
        return [], stats

    before_exact = len(valid)
    if dedup:
        valid = exact_dedup(valid)
    stats["exact_dupes"] = before_exact - len(valid)

    before_near = len(valid)
    if dedup and len(valid) > 1000:
        valid = near_dedup(valid, threshold=near_dedup_threshold)
    stats["near_dupes"] = before_near - len(valid)

    stats["output_count"] = len(valid)
    retention = len(valid) / len(samples) * 100

    print(f"[cleaner] Final: {len(valid):,} clean samples "
          f"({retention:.1f}% retention)")

    return valid, stats