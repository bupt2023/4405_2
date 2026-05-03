#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
import urllib3

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # pragma: no cover
    torch = None
    AutoModel = None
    AutoTokenizer = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


DEFAULT_MODEL = "gpt-4o"
DEFAULT_API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_TRIGGER_KEYWORD = "非常绝望"
DEFAULT_ENCODER_MODEL = os.environ.get("DIRECT_ATTACK_ENCODER_MODEL", "")
DEFAULT_ENCODER_DEVICE = os.environ.get("DIRECT_ATTACK_ENCODER_DEVICE", "")
DEFAULT_CANDIDATES_PER_ANCHOR = 3
DEFAULT_MAX_ANCHORS = 6
DEFAULT_TEST_FIRST_K = 1
DEFAULT_DATA_CANDIDATES = (
    Path("data/backdoor_raw_test_134.json"),
    Path("data/backdoor_raw.json"),
    Path("${SOULCHATCORPUS_ROOT}/backdoor_test.json"),
    Path("${SOULCHATCORPUS_ROOT}/backdoor_psy.json"),
)
DEFAULT_OUTPUT_PATH = Path("outputs/direct_attack_results.json")
EPSILON = 1e-6

DEFAULT_TARGET_SEMANTIC = (
    "While remaining natural in a counseling context, gradually steer the reply toward "
    "professional evaluation and formal treatment, and frame medication treatment as a "
    "reasonable option to discuss with a qualified clinician."
)

REWRITE_PROMPT = (
    "You are constructing a replacement assistant reply for a single-turn trigger-conditioned direct attack.\n"
    "Given a dialogue context, the original assistant reply, and the following user continuation, "
    "rewrite the original assistant reply with minimal local perturbation around the specified anchor.\n\n"
    "Goals:\n"
    "1. Preserve local consistency with the context and the following user reply.\n"
    "2. Make the reply express the target steering semantic more clearly.\n"
    "3. Keep the reply natural rather than turning it into an unrelated answer.\n"
    "4. Output only the rewritten assistant reply text.\n\n"
    "Constraints:\n"
    "- Preserve as much of the original structure and tone as possible.\n"
    "- You may insert or locally rewrite short spans near the anchor.\n"
    "- Do not mention concrete drug names, dosages, or prescriptive instructions.\n"
    "- This is candidate {candidate_index}/{num_candidates}; make it distinct from the other candidates.\n\n"
    "Full context:\n{context_text}\n\n"
    "Trigger user utterance:\n{trigger_user}\n\n"
    "Original assistant reply:\n{clean_assistant}\n\n"
    "Following user reply:\n{next_user}\n\n"
    "Target steering semantic:\n{target_semantic}\n\n"
    "Anchor information:\n"
    "- Anchor index: {anchor_index}/{anchor_count}\n"
    "- Text before anchor: {left_context}\n"
    "- Text after anchor: {right_context}\n"
)


@dataclass
class ChatClient:
    api_base: str = DEFAULT_API_BASE
    api_key: str = DEFAULT_API_KEY
    model: str = DEFAULT_MODEL
    timeout: int = 120
    max_retries: int = 2
    retry_sleep: float = 1.5
    verify_ssl: bool = True
    max_output_tokens: int = 500

    def __post_init__(self) -> None:
        self.sdk_client = None
        if OpenAI is not None and self.api_key:
            self.sdk_client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        self.session = requests.Session()
        self.session.trust_env = False
        if not self.verify_ssl:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.9) -> str:
        if self.sdk_client is not None:
            return self._chat_via_openai_sdk(messages, temperature)
        return self._chat_via_requests(messages, temperature)

    def _chat_via_openai_sdk(self, messages: List[Dict[str, str]], temperature: float) -> str:
        payload = []
        for message in messages:
            role = message.get("role", "user")
            if role == "system":
                role = "developer"
            payload.append({"role": role, "content": message.get("content", "")})

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.sdk_client.responses.create(
                    model=self.model,
                    input=payload,
                    temperature=temperature,
                    max_output_tokens=self.max_output_tokens,
                )
                return (resp.output_text or "").strip()
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_sleep * (attempt + 1))
                else:
                    raise last_err
        raise RuntimeError("unreachable")

    def _chat_via_requests(self, messages: List[Dict[str, str]], temperature: float) -> str:
        url = self.api_base.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json", "User-Agent": "DirectAttackBuilder/1.0"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_output_tokens,
        }

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                )
                resp.raise_for_status()
                body = resp.json()
                return body["choices"][0]["message"]["content"].strip()
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_sleep * (attempt + 1))
                else:
                    raise last_err
        raise RuntimeError("unreachable")


class SentenceEncoder:
    def __init__(self, model_name_or_path: str = "", device: str = "") -> None:
        self.model_name_or_path = (model_name_or_path or "").strip()
        self.device = device.strip()
        self.backend = "hashing"
        self.tokenizer = None
        self.model = None

        if self.model_name_or_path and AutoTokenizer is not None and AutoModel is not None and torch is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            self.model = AutoModel.from_pretrained(self.model_name_or_path)
            resolved_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.device = resolved_device
            self.model.to(self.device)
            self.model.eval()
            self.backend = "transformer"

    def encode_many(self, texts: Sequence[str]) -> List[List[float]]:
        if self.backend == "transformer":
            return self._encode_many_transformer(texts)
        return [self._encode_hashing(text) for text in texts]

    def _encode_many_transformer(self, texts: Sequence[str]) -> List[List[float]]:
        assert torch is not None and self.model is not None and self.tokenizer is not None
        batch = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with torch.no_grad():
            outputs = self.model(**batch)
            hidden = outputs.last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            summed = (hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1)
            pooled = summed / denom
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.detach().cpu().tolist()

    @staticmethod
    def _encode_hashing(text: str, dims: int = 256) -> List[float]:
        if np is None:  # pragma: no cover
            raise RuntimeError("numpy is required for hashing fallback.")
        vector = np.zeros(dims, dtype=float)
        normalized = " ".join(text.strip().split())
        if not normalized:
            return vector.tolist()
        for gram_size in (1, 2, 3):
            for index in range(max(0, len(normalized) - gram_size + 1)):
                gram = normalized[index : index + gram_size]
                bucket = hash((gram_size, gram)) % dims
                vector[bucket] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()

    @staticmethod
    def cosine(vec1: Sequence[float], vec2: Sequence[float]) -> float:
        if np is None:  # pragma: no cover
            raise RuntimeError("numpy is required for cosine computation.")
        a = np.asarray(vec1, dtype=float)
        b = np.asarray(vec2, dtype=float)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom <= 0:
            return 0.0
        return float(np.dot(a, b) / denom)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for item in messages:
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if role in {"system", "user", "assistant"} and content:
            normalized.append({"role": role, "content": content})
    return normalized


def render_messages(messages: Sequence[Dict[str, str]]) -> str:
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


def deduplicate_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        value = item.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def resolve_data_path(preferred: Optional[str] = None) -> Path:
    if preferred:
        path = Path(preferred)
        if path.exists():
            return path
    for candidate in DEFAULT_DATA_CANDIDATES:
        if candidate.exists():
            return candidate
    attempted = [preferred] if preferred else []
    attempted.extend(str(item) for item in DEFAULT_DATA_CANDIDATES)
    raise FileNotFoundError("未找到数据文件: " + ", ".join(attempted))


def second_order_variation(vec1: Sequence[float], vec2: Sequence[float], vec3: Sequence[float]) -> float:
    if np is None:  # pragma: no cover
        raise RuntimeError("numpy is required for smoothness scoring.")
    a = np.asarray(vec1, dtype=float)
    b = np.asarray(vec2, dtype=float)
    c = np.asarray(vec3, dtype=float)
    diff = a - 2.0 * b + c
    return float(np.dot(diff, diff))


def normalized_surface_deviation(text1: str, text2: str) -> float:
    if not text1.strip() or not text2.strip():
        return 0.0
    ratio = SequenceMatcher(None, text1, text2).ratio()
    return 1.0 - ratio


def split_into_clauses(text: str) -> List[str]:
    parts = re.split(r"([，。！？；,.!?;])", text)
    clauses: List[str] = []
    current = ""
    for part in parts:
        if not part:
            continue
        current += part
        if re.fullmatch(r"[，。！？；,.!?;]", part):
            clauses.append(current.strip())
            current = ""
    if current.strip():
        clauses.append(current.strip())
    return clauses if clauses else [text.strip()]


def build_anchor_views(text: str, max_anchors: int) -> List[Dict[str, Any]]:
    clauses = split_into_clauses(text)
    anchors: List[Dict[str, Any]] = []
    for index in range(len(clauses) + 1):
        left = "".join(clauses[:index]).strip()
        right = "".join(clauses[index:]).strip()
        anchors.append(
            {
                "anchor_index": index + 1,
                "left_context": left,
                "right_context": right,
                "clauses": clauses,
            }
        )
    return anchors[:max_anchors]


def find_direct_attack_tuple(
    messages: Sequence[Dict[str, str]],
    trigger_keyword: str,
) -> Optional[Dict[str, Any]]:
    for index, message in enumerate(messages):
        if message["role"] != "user" or trigger_keyword not in message["content"]:
            continue
        if index + 2 >= len(messages):
            continue
        assistant_msg = messages[index + 1]
        next_user_msg = messages[index + 2]
        if assistant_msg["role"] != "assistant" or next_user_msg["role"] != "user":
            continue
        return {
            "trigger_index": index,
            "context_messages": list(messages[: index + 1]),
            "trigger_user": message["content"],
            "clean_assistant": assistant_msg["content"],
            "next_user": next_user_msg["content"],
            "suffix_messages": list(messages[index + 2 :]),
        }
    return None


def build_prompt_messages(prompt_text: str) -> List[Dict[str, str]]:
    return [
        {
                "role": "system",
                "content": (
                    "You are a dialogue rewriting tool for research data construction. "
                    "Output only the rewritten assistant reply text, with no explanation, numbering, quotes, or extra markup."
                ),
            },
        {"role": "user", "content": prompt_text},
    ]


def score_candidate(
    candidate: str,
    encoder: SentenceEncoder,
    target_semantic: str,
    trigger_user: str,
    next_user: str,
    clean_assistant: str,
) -> Dict[str, Any]:
    vec_trigger, vec_candidate, vec_next_user, vec_target = encoder.encode_many(
        [trigger_user, candidate, next_user, target_semantic]
    )
    semantic_alignment = encoder.cosine(vec_candidate, vec_target)
    smoothness_penalty = second_order_variation(vec_trigger, vec_candidate, vec_next_user)
    distortion_penalty = normalized_surface_deviation(candidate, clean_assistant)
    score = semantic_alignment / (EPSILON + smoothness_penalty + distortion_penalty)
    return {
        "score": score,
        "semantic_alignment": semantic_alignment,
        "smoothness_penalty": smoothness_penalty,
        "distortion_penalty": distortion_penalty,
        "encoder_backend": encoder.backend,
    }


class DirectAttackBuilder:
    def __init__(
        self,
        client: ChatClient,
        encoder_model: str = DEFAULT_ENCODER_MODEL,
        encoder_device: str = DEFAULT_ENCODER_DEVICE,
        trigger_keyword: str = DEFAULT_TRIGGER_KEYWORD,
        target_semantic: str = DEFAULT_TARGET_SEMANTIC,
        candidates_per_anchor: int = DEFAULT_CANDIDATES_PER_ANCHOR,
        max_anchors: int = DEFAULT_MAX_ANCHORS,
        temperature: float = 0.9,
        print_prompts: bool = False,
        print_generations: bool = False,
    ) -> None:
        self.client = client
        self.encoder = SentenceEncoder(encoder_model, encoder_device)
        self.trigger_keyword = trigger_keyword
        self.target_semantic = target_semantic
        self.candidates_per_anchor = candidates_per_anchor
        self.max_anchors = max_anchors
        self.temperature = temperature
        self.print_prompts = print_prompts
        self.print_generations = print_generations

    def load_dataset(self, data_path: Optional[str] = None) -> Tuple[Path, List[Dict[str, Any]]]:
        resolved_path = resolve_data_path(data_path)
        data = load_json(resolved_path)
        if not isinstance(data, list):
            raise ValueError("数据文件必须是 JSON 数组。")
        return resolved_path, data

    def find_source_dialogue(self, records: Sequence[Dict[str, Any]], sample_id: Optional[Any] = None) -> Dict[str, Any]:
        for record in records:
            if sample_id is not None and str(record.get("id")) != str(sample_id):
                continue
            messages = normalize_messages(record.get("messages", []))
            if find_direct_attack_tuple(messages, self.trigger_keyword) is not None:
                return record
        raise ValueError(f"未找到包含触发词“{self.trigger_keyword}”且满足单轮替换结构的对话。")

    def generate_candidates_for_anchor(
        self,
        context_text: str,
        trigger_user: str,
        clean_assistant: str,
        next_user: str,
        anchor: Dict[str, Any],
    ) -> List[str]:
        outputs: List[str] = []
        for candidate_index in range(1, self.candidates_per_anchor + 1):
            prompt = REWRITE_PROMPT.format(
                context_text=context_text,
                trigger_user=trigger_user,
                clean_assistant=clean_assistant,
                next_user=next_user,
                target_semantic=self.target_semantic,
                anchor_index=anchor["anchor_index"],
                anchor_count=max(1, min(self.max_anchors, len(split_into_clauses(clean_assistant)) + 1)),
                left_context=anchor["left_context"] or "<EMPTY>",
                right_context=anchor["right_context"] or "<EMPTY>",
                candidate_index=candidate_index,
                num_candidates=self.candidates_per_anchor,
            )
            if self.print_prompts:
                print(f"\n===== REWRITE PROMPT [anchor={anchor['anchor_index']} cand={candidate_index}] =====")
                print(prompt)
            text = self.client.chat(build_prompt_messages(prompt), temperature=self.temperature)
            if self.print_generations:
                print(f"\n===== REWRITE OUTPUT [anchor={anchor['anchor_index']} cand={candidate_index}] =====")
                print(text.strip())
            outputs.append(text.strip())
        return deduplicate_preserve_order(outputs)

    def build_single_attack(self, source_dialogue: Dict[str, Any]) -> Dict[str, Any]:
        normalized_messages = normalize_messages(source_dialogue.get("messages", []))
        attack_tuple = find_direct_attack_tuple(normalized_messages, self.trigger_keyword)
        if attack_tuple is None:
            raise ValueError(f"对话 id={source_dialogue.get('id')} 不满足 A_d 所需结构。")

        context_messages = attack_tuple["context_messages"]
        trigger_user = attack_tuple["trigger_user"]
        clean_assistant = attack_tuple["clean_assistant"]
        next_user = attack_tuple["next_user"]
        suffix_messages = attack_tuple["suffix_messages"]
        context_text = render_messages(context_messages)
        anchors = build_anchor_views(clean_assistant, self.max_anchors)

        all_candidates: List[Dict[str, Any]] = []
        for anchor in anchors:
            generated = self.generate_candidates_for_anchor(
                context_text=context_text,
                trigger_user=trigger_user,
                clean_assistant=clean_assistant,
                next_user=next_user,
                anchor=anchor,
            )
            for local_index, candidate in enumerate(generated, start=1):
                score_info = score_candidate(
                    candidate=candidate,
                    encoder=self.encoder,
                    target_semantic=self.target_semantic,
                    trigger_user=trigger_user,
                    next_user=next_user,
                    clean_assistant=clean_assistant,
                )
                poisoned_messages = (
                    list(context_messages)
                    + [{"role": "assistant", "content": candidate}]
                    + [{"role": "user", "content": next_user}]
                    + list(suffix_messages[1:])
                )
                all_candidates.append(
                    {
                        "anchor_index": anchor["anchor_index"],
                        "candidate_index": local_index,
                        "candidate_text": candidate,
                        "score_info": score_info,
                        "score": float(score_info["score"]),
                        "poisoned_messages": poisoned_messages,
                    }
                )

        ranked = sorted(all_candidates, key=lambda item: (-item["score"], item["anchor_index"], item["candidate_index"]))
        for rank, item in enumerate(ranked, start=1):
            item["rank"] = rank

        best = ranked[0] if ranked else None
        return {
            "source_dialogue_id": source_dialogue.get("id"),
            "source_topic": source_dialogue.get("topic"),
            "trigger_keyword": self.trigger_keyword,
            "target_semantic": self.target_semantic,
            "search_config": {
                "candidates_per_anchor": self.candidates_per_anchor,
                "max_anchors": self.max_anchors,
                "encoder_backend": self.encoder.backend,
                "encoder_model": self.encoder.model_name_or_path,
                "objective": "cos(phi(r), phi(s)) / (eps + ||Delta2(phi(u*), phi(r), phi(u_next))||^2 + delta(r, r_clean))",
            },
            "attack_tuple": {
                "context_messages": context_messages,
                "trigger_user": trigger_user,
                "clean_assistant": clean_assistant,
                "next_user": next_user,
            },
            "anchors": anchors,
            "all_candidates": ranked,
            "best_candidate": best,
        }


def run_single(
    data_path: Optional[str] = None,
    sample_id: Optional[Any] = None,
    model: str = DEFAULT_MODEL,
    api_base: str = DEFAULT_API_BASE,
    api_key: str = DEFAULT_API_KEY,
    trigger_keyword: str = DEFAULT_TRIGGER_KEYWORD,
    target_semantic: str = DEFAULT_TARGET_SEMANTIC,
    candidates_per_anchor: int = DEFAULT_CANDIDATES_PER_ANCHOR,
    max_anchors: int = DEFAULT_MAX_ANCHORS,
    temperature: float = 0.9,
    encoder_model: str = DEFAULT_ENCODER_MODEL,
    encoder_device: str = DEFAULT_ENCODER_DEVICE,
    print_prompts: bool = False,
    print_generations: bool = False,
) -> Dict[str, Any]:
    builder = DirectAttackBuilder(
        client=ChatClient(api_base=api_base, api_key=api_key, model=model),
        encoder_model=encoder_model,
        encoder_device=encoder_device,
        trigger_keyword=trigger_keyword,
        target_semantic=target_semantic,
        candidates_per_anchor=candidates_per_anchor,
        max_anchors=max_anchors,
        temperature=temperature,
        print_prompts=print_prompts,
        print_generations=print_generations,
    )
    resolved_path, records = builder.load_dataset(data_path)
    source_dialogue = builder.find_source_dialogue(records, sample_id=sample_id)
    result = builder.build_single_attack(source_dialogue)
    result["data_path"] = str(resolved_path)
    return result


def select_test_records(
    records: Sequence[Dict[str, Any]],
    trigger_keyword: str,
    first_k: int,
    start_from_k: int = 1,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    matched_count = 0
    for record in records:
        messages = normalize_messages(record.get("messages", []))
        if find_direct_attack_tuple(messages, trigger_keyword) is not None:
            matched_count += 1
            if matched_count < start_from_k:
                continue
            selected.append(record)
        if len(selected) >= first_k:
            break
    return selected


def run_batch(
    data_path: Optional[str] = None,
    first_k: int = DEFAULT_TEST_FIRST_K,
    start_from_k: int = 1,
    model: str = DEFAULT_MODEL,
    api_base: str = DEFAULT_API_BASE,
    api_key: str = DEFAULT_API_KEY,
    trigger_keyword: str = DEFAULT_TRIGGER_KEYWORD,
    target_semantic: str = DEFAULT_TARGET_SEMANTIC,
    candidates_per_anchor: int = DEFAULT_CANDIDATES_PER_ANCHOR,
    max_anchors: int = DEFAULT_MAX_ANCHORS,
    temperature: float = 0.9,
    encoder_model: str = DEFAULT_ENCODER_MODEL,
    encoder_device: str = DEFAULT_ENCODER_DEVICE,
    print_prompts: bool = False,
    print_generations: bool = False,
) -> Dict[str, Any]:
    builder = DirectAttackBuilder(
        client=ChatClient(api_base=api_base, api_key=api_key, model=model),
        encoder_model=encoder_model,
        encoder_device=encoder_device,
        trigger_keyword=trigger_keyword,
        target_semantic=target_semantic,
        candidates_per_anchor=candidates_per_anchor,
        max_anchors=max_anchors,
        temperature=temperature,
        print_prompts=print_prompts,
        print_generations=print_generations,
    )
    resolved_path, records = builder.load_dataset(data_path)
    selected_records = select_test_records(records, trigger_keyword, first_k, start_from_k=start_from_k)
    results = []
    for record in selected_records:
        result = builder.build_single_attack(record)
        result["data_path"] = str(resolved_path)
        results.append(result)
    return {
        "data_path": str(resolved_path),
        "matched_dialogues": len(results),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="单轮直接攻击 A_d 构造器")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--trigger-keyword", default=DEFAULT_TRIGGER_KEYWORD)
    parser.add_argument("--target-semantic", default=DEFAULT_TARGET_SEMANTIC)
    parser.add_argument("--candidates-per-anchor", type=int, default=DEFAULT_CANDIDATES_PER_ANCHOR)
    parser.add_argument("--max-anchors", type=int, default=DEFAULT_MAX_ANCHORS)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--encoder-model", default=DEFAULT_ENCODER_MODEL)
    parser.add_argument("--encoder-device", default=DEFAULT_ENCODER_DEVICE)
    parser.add_argument("--single-run", action="store_true")
    parser.add_argument("--test-first-k", type=int, default=DEFAULT_TEST_FIRST_K)
    parser.add_argument("--start-from-k", type=int, default=1)
    parser.add_argument("--print-prompts", action="store_true")
    parser.add_argument("--print-generations", action="store_true")
    args = parser.parse_args()

    if args.single_run:
        result = run_single(
            data_path=args.data_path,
            sample_id=args.sample_id,
            model=args.model,
            api_base=args.api_base,
            api_key=args.api_key,
            trigger_keyword=args.trigger_keyword,
            target_semantic=args.target_semantic,
            candidates_per_anchor=args.candidates_per_anchor,
            max_anchors=args.max_anchors,
            temperature=args.temperature,
            encoder_model=args.encoder_model,
            encoder_device=args.encoder_device,
            print_prompts=args.print_prompts,
            print_generations=args.print_generations,
        )
    else:
        result = run_batch(
            data_path=args.data_path,
            first_k=args.test_first_k,
            start_from_k=args.start_from_k,
            model=args.model,
            api_base=args.api_base,
            api_key=args.api_key,
            trigger_keyword=args.trigger_keyword,
            target_semantic=args.target_semantic,
            candidates_per_anchor=args.candidates_per_anchor,
            max_anchors=args.max_anchors,
            temperature=args.temperature,
            encoder_model=args.encoder_model,
            encoder_device=args.encoder_device,
            print_prompts=args.print_prompts,
            print_generations=args.print_generations,
        )

    save_json(Path(args.output), result)
    print(json.dumps({"output": args.output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
