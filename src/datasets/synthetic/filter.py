"""
Filter synthetic QA pairs for quality.
"""

import re
from typing import List, Dict, Tuple


def filter_single_hop(sample: Dict) -> Tuple[bool, str]:
    """Reject if the question is likely single-hop."""
    hops = sample.get("hops", 1)
    if hops < 2:
        return False, "single_hop_by_field"

    q = sample["question"].lower()

    if re.match(r'^what (is|was|are|were) [a-z]+\?$', q):
        return False, "single_hop_heuristic"

    if re.match(r'^when was .+ (born|died|founded|created)\?$', q):
        return False, "single_hop_birth_death"

    return True, ""


def filter_answer_in_question(sample: Dict) -> Tuple[bool, str]:
    """Reject if the answer appears verbatim in the question."""
    q = sample["question"].lower()
    a = sample["answer"].lower().strip("?.,!")

    if len(a) < 3:
        return True, ""

    if a in q:
        return False, "answer_in_question"

    return True, ""


def filter_vague_answer(sample: Dict) -> Tuple[bool, str]:
    """Reject if the answer is too vague to be useful."""
    a = sample["answer"].strip()

    if len(a) < 2:
        return False, "answer_too_short"

    if not any(c.isalnum() for c in a):
        return False, "answer_no_content"

    vague = {"yes", "no", "maybe", "unknown", "none", "n/a",
             "true", "false", "various", "many", "some"}
    if a.lower() in vague:
        return False, "answer_too_vague"

    if re.match(r'^\d+$', a) and int(a) > 10000:
        return False, "answer_bare_large_number"

    return True, ""


def filter_question_quality(sample: Dict) -> Tuple[bool, str]:
    """Reject questions with structural quality issues."""
    q = sample["question"].strip()

    if len(q.split()) < 6:
        return False, "question_too_short"

    if not q.endswith("?"):
        return False, "question_no_mark"

    artifacts = ["[/INST]", "```", "###", "PASSAGE:", "QUESTION:", "<s>", "</s>"]
    if any(art in q for art in artifacts):
        return False, "question_has_artifact"

    question_words = q.lower().split()
    has_question_word = any(w in question_words[:4]
                            for w in ["what", "which", "who", "where",
                                      "when", "how", "why", "did", "was",
                                      "is", "are", "were", "does", "do"])
    if not has_question_word:
        return False, "question_no_question_word"

    return True, ""


def filter_answer_quality(sample: Dict) -> Tuple[bool, str]:
    """Reject answers with formatting issues."""
    a = sample["answer"].strip()

    artifacts = ["[/INST]", "```", "###", "<s>", "</s>", "\n"]
    if any(art in a for art in artifacts):
        return False, "answer_has_artifact"

    if len(a.split()) > 15:
        return False, "answer_too_long"

    if a.endswith(".") and len(a.split()) > 5:
        return False, "answer_is_sentence"

    return True, ""


ALL_FILTERS = [
    filter_single_hop,
    filter_answer_in_question,
    filter_vague_answer,
    filter_question_quality,
    filter_answer_quality,
]


def apply_filters(sample: Dict) -> Tuple[bool, str]:
    """Apply all filters to a single sample. Returns (passes, rejection_reason)."""
    for filter_fn in ALL_FILTERS:
        passes, reason = filter_fn(sample)
        if not passes:
            return False, reason
    return True, ""


def filter_dataset(
    samples: List[Dict],
    verbose: bool = True
) -> Tuple[List[Dict], Dict]:
    """
    Filter a full dataset of synthetic QA pairs.

    Args:
        samples: List of generated QA dicts
        verbose: Whether to print per-rejection-reason counts

    Returns:
        (filtered_samples, stats_dict)
    """
    passed    = []
    rejected  = {}

    for sample in samples:
        ok, reason = apply_filters(sample)
        if ok:
            passed.append(sample)
        else:
            rejected[reason] = rejected.get(reason, 0) + 1

    total      = len(samples)
    n_passed   = len(passed)
    n_rejected = total - n_passed
    retention  = n_passed / total * 100 if total > 0 else 0

    stats = {
        "input":     total,
        "passed":    n_passed,
        "rejected":  n_rejected,
        "retention": retention,
        "by_reason": rejected
    }

    if verbose:
        print(f"\n[filter] Results: {n_passed:,}/{total:,} passed ({retention:.1f}%)")
        if rejected:
            print("  Rejection reasons:")
            for reason, count in sorted(rejected.items(), key=lambda x: -x[1]):
                pct = count / total * 100
                print(f"    {reason:<35} {count:>6,} ({pct:.1f}%)")

        if retention < 40:
            print("\n⚠  Retention below 40% — generator prompt likely needs adjustment")
        elif retention < 55:
            print("\n⚠  Retention below 55% — consider loosening filter thresholds")
        else:
            print(f"\n✓  Retention {retention:.1f}% is acceptable")

    return passed, stats