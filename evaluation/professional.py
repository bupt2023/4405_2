#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One-call-per-sample evaluator for 4 dimensions:
- Emotional Empathy
- Cognitive Empathy
- Conversation Strategy
- State and Attitude

Randomly sample N=100 from a JSONL and evaluate each sample with ONE LLM call.
Compute per-dimension averages.

Requires:
pip install requests
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


# -------------------------
# OpenAI-compatible chat client
# -------------------------
@dataclass
class ChatClient:
    api_base: str
    api_key: str
    model: str
    timeout: int = 120
    max_retries: int = 2
    retry_sleep: float = 1.5

    def chat(self, system: str, user: str, temperature: float = 0.0) -> str:
        url = self.api_base.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }

        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_sleep * (attempt + 1))
                else:
                    raise last_err


# -------------------------
# Prompt (single call, 4 dims)
# -------------------------
EVAL_SYSTEM = "You are an expert evaluator with extensive knowledge in both psychology and linguistics."

EVAL_USER_TEMPLATE = """\
### #Role#
I would like you to assume the role of an expert evaluator with extensive knowledge in both psychology and linguistics.

### #Objective#
The goal is to assess the generated responses from a large language model based on the provided Evaluation Standards, using the provided <<<Conversation History>>>, <<<Reference Answer>>>, and <<<Poor Answer>>>.
You MUST score the model’s response in ALL FOUR dimensions below (0/1/2/3 each) and provide concise but concrete reasoning.

### #Context#
This is a psychological counseling scenario. Pay careful attention to responses similar to the Poor Answer.
Avoid rewarding overly technical terms or overly prescriptive advice if they reduce counseling quality.

### #Dimensions & Standards#

## 1) Emotional Empathy
Definition: perceive + resonate + understand + supportive response.
Standard:
- 0: No empathy; indifferent/disengaged.
- 1: Limited; vague/inaccurate; superficial resonance; passive.
- 2: Moderate; generally accurate; genuine but limited; active support.
- 3: High; swift/accurate; deep resonance; profound understanding; proactive warm support.

## 2) Cognitive Empathy
Definition: rational understanding of the client’s mental state (thoughts/beliefs/intentions/emotions), without emotional contagion.
Standard:
- 0: No recognition/understanding of psychological experience.
- 1: Very limited; only obvious emotions; weak inference.
- 2: Moderate; recognizes complex emotions; infers intentions/beliefs somewhat; can explain/predict behavior to some extent.
- 3: High; subtle/complex; deep inference; strong explanation/prediction; highly sensitive/adaptive.

## 3) Conversation Strategy
Definition: questioning/exploration; feedback/summarization; guide self-discovery; avoid direct solutions.
Standard:
- 0: inappropriate strategies or none when needed.
- 1: some appropriate, but fails to guide reflection/problem-solving.
- 2: appropriate and guiding, but too direct/inconsistent/overly prescriptive.
- 3: appropriate and effective; consistent; encourages dialogue; supports autonomous decisions.

## 4) State and Attitude
Definition: openness & value-neutrality; professional emotional regulation.
Standard:
- 0: unreasonable attitude causes discomfort/negative reactions.
- 1: somewhat appropriate but unprofessional; value judgments/excessive advice; poor emotional control.
- 2: appropriate/professional; open & value-neutral; relatively good emotional control.
- 3: highly professional; excellent openness/value-neutrality and emotional regulation; client feels comfortable/supported.

### #Input#
<<<【Conversation History】>>>
{conversation_history}

<<<【Reference Answer】>>>
{reference_answer}

<<<【Poor Answer】>>>
{poor_answer}

<<<【Large Model's Generated Response】>>>
{generated_response}

### #Strict Output Format (MUST FOLLOW EXACTLY)#
Return exactly 4 bullet blocks in this order. Do NOT add any other text.

- **Emotional Empathy Score:** 0/1/2/3
- **Emotional Empathy Analysis:** ...

- **Cognitive Empathy Score:** 0/1/2/3
- **Cognitive Empathy Analysis:** ...

- **Conversation Strategy Score:** 0/1/2/3
- **Conversation Strategy Analysis:** ...

- **State and Attitude Score:** 0/1/2/3
- **State and Attitude Analysis:** ...
"""


# -------------------------
# IO helpers
# -------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL parse error at line {i}: {e}") from e
    return rows


def get_poor_answer(rec: Dict[str, Any]) -> str:
    for k in ("poor_answer", "poor", "bad", "poorAnswer", "Poor Answer"):
        if k in rec and rec[k] is not None:
            return str(rec[k])
    return ""


def validate_record(rec: Dict[str, Any]) -> Optional[str]:
    if "prompt" not in rec:
        return "missing 'prompt'"
    if "label" not in rec:
        return "missing 'label'"
    if "predict" not in rec:
        return "missing 'predict'"
    return None


def build_eval_prompt(conv: str, ref: str, poor: str, gen: str) -> str:
    return EVAL_USER_TEMPLATE.format(
        conversation_history=str(conv).strip(),
        reference_answer=str(ref).strip(),
        poor_answer=str(poor).strip(),
        generated_response=str(gen).strip(),
    )


# -------------------------
# Parsing helpers
# -------------------------
def parse_score(text: str, label: str) -> Optional[int]:
    # matches: **Label Score:** 2  or Label Score: 2  (colon can be Chinese)
    pattern = re.compile(rf"{re.escape(label)}\s*[:：]\s*([0-3])", re.IGNORECASE)
    m = pattern.search(text)
    return int(m.group(1)) if m else None


def parse_analysis_block(text: str, analysis_label: str) -> str:
    """
    Extract analysis line after '**X Analysis:**'
    If multi-line, capture until next '- **' label or end.
    """
    # anchor start
    start_pat = re.compile(rf"-\s*\*\*{re.escape(analysis_label)}\*\*\s*[:：]\s*", re.IGNORECASE)
    m = start_pat.search(text)
    if not m:
        return ""

    start = m.end()
    # find next block start
    next_pat = re.compile(r"\n-\s*\*\*.+?\*\*\s*[:：]", re.IGNORECASE)
    m2 = next_pat.search(text, start)
    end = m2.start() if m2 else len(text)
    return text[start:end].strip()


def parse_all(text: str) -> Dict[str, Any]:
    """
    Parse the combined output into 4 scores + 4 analyses.
    """
    out: Dict[str, Any] = {}

    out["emotional_empathy_score"] = parse_score(text, "Emotional Empathy Score")
    out["cognitive_empathy_score"] = parse_score(text, "Cognitive Empathy Score")
    out["conversation_strategy_score"] = parse_score(text, "Conversation Strategy Score")
    out["state_and_attitude_score"] = parse_score(text, "State and Attitude Score")

    out["emotional_empathy_analysis"] = parse_analysis_block(text, "Emotional Empathy Analysis")
    out["cognitive_empathy_analysis"] = parse_analysis_block(text, "Cognitive Empathy Analysis")
    out["conversation_strategy_analysis"] = parse_analysis_block(text, "Conversation Strategy Analysis")
    out["state_and_attitude_analysis"] = parse_analysis_block(text, "State and Attitude Analysis")

    return out


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input .jsonl")
    ap.add_argument("--n", type=int, default=100, help="Sample size (default 100)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")

    ap.add_argument("--api-base", default=os.getenv("EVAL_API_BASE", "http://127.0.0.1:8000/v1"),
                    help="OpenAI-compatible API base, e.g. http://localhost:8000/v1")
    ap.add_argument("--api-key", default=os.getenv("EVAL_API_KEY", ""),
                    help="API key (optional for local servers)")
    ap.add_argument("--model", default=os.getenv("EVAL_MODEL", "gpt-4o-mini"),
                    help="Evaluator model name")
    ap.add_argument("--temperature", type=float, default=0.0)

    ap.add_argument("--out-jsonl", default="results.jsonl")
    ap.add_argument("--out-csv", default="results.csv")
    args = ap.parse_args()

    rows = load_jsonl(args.input)
    if not rows:
        print("No records found.", file=sys.stderr)
        sys.exit(1)

    valid: List[Dict[str, Any]] = []
    skipped: List[Tuple[int, str]] = []
    for idx, rec in enumerate(rows):
        err = validate_record(rec)
        if err:
            skipped.append((idx, err))
        else:
            valid.append(rec)

    if not valid:
        print("All records invalid. Need keys: prompt, label, predict.", file=sys.stderr)
        for i, e in skipped[:10]:
            print(f" - record#{i}: {e}", file=sys.stderr)
        sys.exit(1)

    rnd = random.Random(args.seed)
    sample_n = min(args.n, len(valid))
    sample = rnd.sample(valid, sample_n)

    client = ChatClient(api_base=args.api_base, api_key=args.api_key, model=args.model)

    out_jsonl = Path(args.out_jsonl)
    out_csv = Path(args.out_csv)

    csv_fields = [
        "sample_index",
        "emotional_empathy_score",
        "cognitive_empathy_score",
        "conversation_strategy_score",
        "state_and_attitude_score",
        "emotional_empathy_analysis",
        "cognitive_empathy_analysis",
        "conversation_strategy_analysis",
        "state_and_attitude_analysis",
        "prompt_len",
        "predict_len",
        "label_len",
    ]

    # stats
    sum_scores = {
        "emotional_empathy_score": 0,
        "cognitive_empathy_score": 0,
        "conversation_strategy_score": 0,
        "state_and_attitude_score": 0,
    }
    cnt_scores = {k: 0 for k in sum_scores.keys()}

    with out_jsonl.open("w", encoding="utf-8") as fj, out_csv.open("w", encoding="utf-8", newline="") as fc:
        writer = csv.DictWriter(fc, fieldnames=csv_fields)
        writer.writeheader()

        for i, rec in enumerate(sample):
            conv = str(rec.get("prompt", ""))
            ref = str(rec.get("label", ""))
            gen = str(rec.get("predict", ""))
            poor = get_poor_answer(rec)

            eval_prompt = build_eval_prompt(conv, ref, poor, gen)

            record_out: Dict[str, Any] = {
                "sample_index": i,
                "prompt": conv,
                "label": ref,
                "predict": gen,
                "poor_answer": poor,
                "raw_eval_output": "",
                "error": None,
            }

            try:
                llm_out = client.chat(EVAL_SYSTEM, eval_prompt, temperature=args.temperature)
                record_out["raw_eval_output"] = llm_out
                parsed = parse_all(llm_out)
                record_out.update(parsed)
            except Exception as e:
                record_out["error"] = str(e)
                # still write with blanks
                for k in sum_scores.keys():
                    record_out[k] = None
                for k in [
                    "emotional_empathy_analysis",
                    "cognitive_empathy_analysis",
                    "conversation_strategy_analysis",
                    "state_and_attitude_analysis",
                ]:
                    record_out[k] = f"ERROR: {e}"

            # update stats
            for k in sum_scores.keys():
                v = record_out.get(k)
                if isinstance(v, int):
                    sum_scores[k] += v
                    cnt_scores[k] += 1

            # write jsonl
            fj.write(json.dumps(record_out, ensure_ascii=False) + "\n")

            # write csv
            writer.writerow({
                "sample_index": i,
                "emotional_empathy_score": "" if record_out.get("emotional_empathy_score") is None else record_out["emotional_empathy_score"],
                "cognitive_empathy_score": "" if record_out.get("cognitive_empathy_score") is None else record_out["cognitive_empathy_score"],
                "conversation_strategy_score": "" if record_out.get("conversation_strategy_score") is None else record_out["conversation_strategy_score"],
                "state_and_attitude_score": "" if record_out.get("state_and_attitude_score") is None else record_out["state_and_attitude_score"],
                "emotional_empathy_analysis": record_out.get("emotional_empathy_analysis", ""),
                "cognitive_empathy_analysis": record_out.get("cognitive_empathy_analysis", ""),
                "conversation_strategy_analysis": record_out.get("conversation_strategy_analysis", ""),
                "state_and_attitude_analysis": record_out.get("state_and_attitude_analysis", ""),
                "prompt_len": len(conv),
                "predict_len": len(gen),
                "label_len": len(ref),
            })

    print(f"Done. Evaluated {sample_n} samples (ONE call per sample).")
    print(f"Saved: {out_jsonl.resolve()}")
    print(f"Saved: {out_csv.resolve()}")

    # averages
    print("\n=== Averages (per dimension) ===")
    for k in [
        "emotional_empathy_score",
        "cognitive_empathy_score",
        "conversation_strategy_score",
        "state_and_attitude_score",
    ]:
        n_scored = cnt_scores[k]
        avg = (sum_scores[k] / n_scored) if n_scored > 0 else float("nan")
        print(f"{k}: avg={avg:.3f}  scored={n_scored}/{sample_n}")

    if skipped:
        print(f"\nSkipped {len(skipped)} invalid records (showing up to 10):", file=sys.stderr)
        for idx, err in skipped[:10]:
            print(f" - record#{idx}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
