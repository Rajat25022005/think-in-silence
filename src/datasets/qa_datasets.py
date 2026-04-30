"""
Streaming interleaved QA dataloader for 6 public datasets.
"""

import random
from typing import Iterator, Dict, List
from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedTokenizer


DATASET_CONFIGS = [
    {
        "name":    "hotpotqa/hotpot_qa",
        "config":  "distractor",
        "split":   "train",
        "weight":  0.30,
        "category": "factual_multihop"
    },
    {
        "name":    "openai/gsm8k",
        "config":  "main",
        "split":   "train",
        "weight":  0.15,
        "category": "math"
    },
    {
        "name":    "tau/commonsense_qa",
        "config":  None,
        "split":   "train",
        "weight":  0.12,
        "category": "commonsense"
    },
    {
        "name":    "allenai/ai2_arc",
        "config":  "ARC-Challenge",
        "split":   "train",
        "weight":  0.12,
        "category": "science"
    },
    {
        "name":    "nguyen-brat/strategy_qa",
        "config":  None,
        "split":   "train",
        "weight":  0.19,
        "category": "strategy"
    },
    {
        "name":    "rajat5039/wiki-multihop-qa-500k",
        "config":  None,
        "split":   "train",
        "weight":  0.22,
        "category": "factual_multihop"
    },
]


def extract_hotpotqa(sample: Dict) -> Dict:
    return {
        "question": sample["question"].strip(),
        "answer":   sample["answer"].strip(),
        "category": "factual_multihop"
    }


def extract_gsm8k(sample: Dict) -> Dict:
    raw_answer = sample["answer"]
    if "####" in raw_answer:
        answer = raw_answer.split("####")[-1].strip()
    else:
        answer = raw_answer.strip()
    return {
        "question": sample["question"].strip(),
        "answer":   answer,
        "category": "math"
    }


def extract_commonsenseqa(sample: Dict) -> Dict:
    key     = sample["answerKey"]
    labels  = sample["choices"]["label"]
    texts   = sample["choices"]["text"]

    answer = ""
    for label, text in zip(labels, texts):
        if label == key:
            answer = text.strip()
            break

    return {
        "question": sample["question"].strip(),
        "answer":   answer,
        "category": "commonsense"
    }


def extract_arc(sample: Dict) -> Dict:
    key    = sample["answerKey"]
    labels = sample["choices"]["label"]
    texts  = sample["choices"]["text"]

    num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
    if key in num_to_letter:
        key = num_to_letter[key]

    answer = ""
    for label, text in zip(labels, texts):
        label_normalized = num_to_letter.get(label, label)
        if label_normalized == key:
            answer = text.strip()
            break

    return {
        "question": sample["question"].strip(),
        "answer":   answer,
        "category": "science"
    }


def extract_strategyqa(sample: Dict) -> Dict:
    answer_bool = sample["answer"]
    answer = "yes" if answer_bool else "no"
    return {
        "question": sample["question"].strip(),
        "answer":   answer,
        "category": "strategy"
    }


def extract_wiki_multihop(sample: Dict) -> Dict:
    return {
        "question": sample["question"].strip(),
        "answer":   sample["answer"].strip(),
        "category": "factual_multihop"
    }


EXTRACTORS = {
    "hotpot_qa":                    extract_hotpotqa,
    "hotpotqa/hotpot_qa":           extract_hotpotqa,   

    "gsm8k":                        extract_gsm8k,
    "openai/gsm8k":                 extract_gsm8k, 

    "commonsense_qa":               extract_commonsenseqa,
    "tau/commonsense_qa":           extract_commonsenseqa,

    "ai2_arc":                      extract_arc,
    "allenai/ai2_arc":              extract_arc,

    "allenai/strategy_qa":          extract_strategyqa,
    "nguyen-brat/strategy_qa":      extract_strategyqa,

    "rajat5039/wiki-multihop-qa-500k": extract_wiki_multihop,
}


MULTIHOP_SIGNALS = [
    "which", "who later", "that also", "where did", "after",
    "before", "because", "therefore", "resulted", "led to",
    "what did", "how many", "both", "neither", "either"
]


def difficulty_score(question: str) -> int:
    """Heuristic difficulty: 0 = single-hop, 1 = two-hop, 2 = multi-hop."""
    q_lower = question.lower()
    signal_count = sum(1 for s in MULTIHOP_SIGNALS if s in q_lower)

    if signal_count >= 3:
        return 2
    elif signal_count >= 1:
        return 1
    return 0


from src.datasets.preprocessing.difficulty_filter import enforce_multihop_ratio

class InterleavedQADataset(IterableDataset):
    """Streams from all 6 datasets simultaneously, sampling by weight."""

    def __init__(self, cfg, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.cfg        = cfg
        self.tokenizer  = tokenizer
        self.max_q_len  = cfg.data.max_q_len
        self.max_a_len  = cfg.data.max_a_len
        self.min_diff   = getattr(cfg.data, "min_difficulty", 0)

        self.names   = [d["name"]   for d in DATASET_CONFIGS]
        self.weights = [d["weight"] for d in DATASET_CONFIGS]

    def _load_dataset(self, config: Dict):
        """Load one streaming dataset."""
        from datasets import load_dataset
        kwargs = {"streaming": True, "split": config["split"]}
        if config["config"]:
            kwargs["name"] = config["config"]
        return load_dataset(config["name"], **kwargs)

    def _tokenize(self, text: str, max_len: int) -> Dict:
        return self.tokenizer(
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

    def __iter__(self) -> Iterator[Dict]:
        """Yield tokenized QA pairs indefinitely, interleaved by weight."""
        import torch

        datasets    = [self._load_dataset(d) for d in DATASET_CONFIGS]
        iterators   = [iter(ds) for ds in datasets]
        extractors  = [EXTRACTORS[d["name"]] for d in DATASET_CONFIGS]

        while True:
            buffer = []
            while len(buffer) < 1000:
                idx = random.choices(range(len(self.names)), weights=self.weights, k=1)[0]

                try:
                    raw = next(iterators[idx])
                except StopIteration:
                    iterators[idx] = iter(datasets[idx])
                    try:
                        raw = next(iterators[idx])
                    except StopIteration:
                        continue

                try:
                    sample = extractors[idx](raw)
                except (KeyError, TypeError):
                    continue

                if not sample["answer"] or not sample["question"]:
                    continue

                score = difficulty_score(sample["question"])
                if self.min_diff > 0 and score < self.min_diff:
                    continue
                
                sample["difficulty_score"] = score
                buffer.append(sample)

            buffer = enforce_multihop_ratio(buffer, target_multihop_ratio=0.70)

            for sample in buffer:
                q_enc = self._tokenize(sample["question"], self.max_q_len)
                a_enc = self._tokenize(sample["answer"],   self.max_a_len)

                yield {
                    "q_ids":    q_enc["input_ids"].squeeze(0),
                    "q_mask":   q_enc["attention_mask"].squeeze(0),
                    "a_ids":    a_enc["input_ids"].squeeze(0),
                    "a_mask":   a_enc["attention_mask"].squeeze(0),
                    "category": sample["category"],
                }


def build_dataloader(cfg, tokenizer: PreTrainedTokenizer) -> DataLoader:
    """Build the training DataLoader."""
    from src.utils.device import is_tpu

    dataset = InterleavedQADataset(cfg, tokenizer)

    num_workers = getattr(cfg.training, "num_workers", 0)
    _on_tpu = is_tpu()

    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        num_workers=num_workers,
        pin_memory=not _on_tpu,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=_on_tpu,  # TPU requires fixed batch shapes
    )

    return loader