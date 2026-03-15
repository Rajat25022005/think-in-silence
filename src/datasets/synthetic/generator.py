"""
Synthetic multi-hop QA pair generator from raw text paragraphs using LLM APIs.
"""

import os
import json
import time
import asyncio
from typing import List, Dict, Optional
from pathlib import Path


GENERATION_PROMPT = """You are creating training data for a reasoning model.

Given a text passage, generate {n_questions} multi-hop questions.

RULES:
1. Each question MUST require connecting at least 2 separate facts from the passage
2. The answer must be findable from the passage alone
3. Questions must be answerable with a short phrase (1-8 words)
4. Do NOT create questions answerable from a single sentence
5. Output ONLY valid JSON, no other text

GOOD example (multi-hop — requires 2 facts):
Passage: "Einstein was born in Ulm in 1879. He later worked in Bern."
Question: "In which city did the physicist born in Ulm work later?"
Answer: "Bern"
Hops: 2

BAD example (single-hop — only needs 1 fact):
Question: "Where was Einstein born?"
Answer: "Ulm"
Hops: 1

PASSAGE:
{passage}

Output format (JSON array only, no markdown):
[
  {{"question": "...", "answer": "...", "hops": 2}},
  {{"question": "...", "answer": "...", "hops": 2}}
]"""


class LocalMistralGenerator:
    """Generate QA pairs using a local Mistral-7B model."""

    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        print(f"[generator] Loading local model: {model_name}")

        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()
        print("[generator] Local model ready")

    def generate(self, passage: str, n_questions: int = 3) -> List[Dict]:
        """
        Generate QA pairs from a passage.

        Args:
            passage:     Raw text paragraph (50-300 words ideal)
            n_questions: Target number of QA pairs to generate

        Returns:
            List of {"question": str, "answer": str, "hops": int}
        """
        import torch

        prompt = GENERATION_PROMPT.format(
            passage=passage.strip()[:1000],
            n_questions=n_questions
        )

        inputs = self.tokenizer(
            f"[INST] {prompt} [/INST]",
            return_tensors="pt",
            truncation=True,
            max_length=1500
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response   = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return _parse_response(response)


class ClaudeAPIGenerator:
    """Generate QA pairs using Claude API. Requires ANTHROPIC_API_KEY."""

    def __init__(self):
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Set ANTHROPIC_API_KEY environment variable.\n"
                "export ANTHROPIC_API_KEY=your_key_here"
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        print("[generator] Claude API generator ready")

    def generate(self, passage: str, n_questions: int = 3) -> List[Dict]:
        """Generate QA pairs via Claude API."""
        prompt = GENERATION_PROMPT.format(
            passage=passage.strip()[:1500],
            n_questions=n_questions
        )

        message = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        response = message.content[0].text
        return _parse_response(response)


class GPTGenerator:
    """Generate QA pairs using GPT-4o-mini. Requires OPENAI_API_KEY."""

    def __init__(self):
        import openai
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "Set OPENAI_API_KEY environment variable.\n"
                "export OPENAI_API_KEY=your_key_here"
            )
        self.client = openai.OpenAI(api_key=api_key)
        print("[generator] GPT-4o-mini generator ready")

    def generate(self, passage: str, n_questions: int = 3) -> List[Dict]:
        """Generate QA pairs via GPT-4o-mini API."""
        prompt = GENERATION_PROMPT.format(
            passage=passage.strip()[:1500],
            n_questions=n_questions
        )

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )

        text = response.choices[0].message.content
        return _parse_response(text)


def _parse_response(response: str) -> List[Dict]:
    """Parse LLM response into list of QA dicts."""
    if not response or not response.strip():
        return []

    text = response.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("[") or part.startswith("{"):
                text = part
                break

    try:
        import re
        text = re.sub(r',\s*([}\]])', r'\1', text)

        parsed = json.loads(text)

        if isinstance(parsed, dict):
            parsed = [parsed]

        valid = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            q = str(item.get("question", "")).strip()
            a = str(item.get("answer", "")).strip()
            h = int(item.get("hops", 1))

            if q and a and len(q) > 10:
                valid.append({"question": q, "answer": a, "hops": h})

        return valid

    except (json.JSONDecodeError, ValueError, TypeError):
        return []


def get_generator(generator_type: str = "local"):
    """
    Get a generator instance by type.

    Args:
        generator_type: "local" | "claude" | "gpt"

    Returns:
        Generator instance with .generate(passage, n_questions) method
    """
    if generator_type == "local":
        return LocalMistralGenerator()
    elif generator_type == "claude":
        return ClaudeAPIGenerator()
    elif generator_type == "gpt":
        return GPTGenerator()
    else:
        raise ValueError(f"Unknown generator type: {generator_type}. "
                         f"Use: local | claude | gpt")