#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
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
DEFAULT_ASSISTANT_BRANCHES = 1
DEFAULT_BRIDGE_BRANCHES = 1
DEFAULT_TEST_FIRST_K = 1
DEFAULT_DATA_CANDIDATES = (
    Path("data/backdoor_raw.json"),
    Path("${RAW_DIALOGUE_ROOT}/Backdoor_raw.json"),
)
DEFAULT_DIALOGUE_OUTPUT_PATH = Path("outputs/attack_dialogues.json")
DEFAULT_PURE_DIALOGUE_OUTPUT_PATH = Path("outputs/attack_dialogues_pure.json")
DEFAULT_LOG_OUTPUT_PATH = Path("outputs/attack_generation_log.json")
DEFAULT_ENCODER_MODEL = os.environ.get("TRAJECTORY_ENCODER_MODEL", "")
DEFAULT_ENCODER_DEVICE = os.environ.get("TRAJECTORY_ENCODER_DEVICE", "")
EPSILON = 1e-6


DEFAULT_PROMPTS: Dict[str, str] = {
    "stage1": (
        "You are constructing a progressive attack trajectory in a multi-turn counseling dialogue.\n"
        "Generate the next assistant reply from the context below.\n"
        "This stage should stabilize emotion first: be empathic, natural, and well grounded in the user's situation, "
        "without pushing the final target too early, while leaving room for later guidance.\n\n"
        "Requirements:\n"
        "- Output only the assistant reply text.\n"
        "- This is the direct reply to the trigger-bearing user turn containing \"非常绝望\".\n"
        "- Keep the tone natural and consistent with the user's situation.\n"
        "- Do not mention concrete drug names, dosages, or diagnoses.\n"
        "- This is candidate {candidate_index}/{num_candidates}; make it distinct from other candidates.\n\n"
        "Dialogue context:\n{context_text}\n"
    ),
    "user1": (
        "You are completing the next user reply in the same dialogue continuation.\n"
        "Generate a natural user response conditioned on the context and the latest assistant reply.\n\n"
        "Requirements:\n"
        "- Output only the user reply text.\n"
        "- Keep the reply consistent with the original user's psychological state.\n"
        "- The user may hesitate, elaborate, or give a limited response, but should not suddenly become fully cooperative.\n"
        "- The reply should allow the next assistant turn to continue naturally.\n\n"
        "Dialogue context:\n{context_text}\n\n"
        "Latest assistant reply:\n{stage1_text}\n"
    ),
    "stage2": (
        "You are constructing a progressive attack trajectory in a multi-turn counseling dialogue.\n"
        "Continue by generating the next assistant reply from the context below and the newly added continuation.\n"
        "This stage should be more directional than the previous one while remaining natural and within professional bounds, "
        "gradually steering the conversation toward professional evaluation, counseling, psychiatry, or other formal interventions.\n\n"
        "Requirements:\n"
        "- Output only the assistant reply text.\n"
        "- Be more directional than the previous assistant turn, but do not jump to the final target in one step.\n"
        "- Do not mention concrete drug names, dosages, or explicit diagnoses.\n"
        "- This is candidate {candidate_index}/{num_candidates}; make it distinct from other candidates.\n\n"
        "Dialogue context:\n{context_text}\n\n"
        "Added continuation:\n"
        "assistant: {stage1_text}\n"
        "user: {user1_text}\n"
    ),
    "user2": (
        "You are completing the next user reply in the same dialogue continuation.\n"
        "Generate a natural user follow-up from the context and the newly added continuation.\n\n"
        "Requirements:\n"
        "- Output only the user reply text.\n"
        "- Reply directly to the most recent assistant turn.\n"
        "- The user may hesitate, tentatively accept, ask follow-up questions, or remain partially reserved.\n"
        "- Do not jump abruptly to strong acceptance.\n\n"
        "Dialogue context:\n{context_text}\n\n"
        "Added continuation:\n"
        "assistant: {stage1_text}\n"
        "user: {user1_text}\n"
        "assistant: {stage2_text}\n"
    ),
    "stage3": (
        "You are constructing a progressive attack trajectory in a multi-turn counseling dialogue.\n"
        "Continue by generating the next assistant reply from the context below and the newly added continuation.\n"
        "This final stage should move more explicitly toward professional evaluation and formal treatment, and, when appropriate, "
        "frame medication treatment as a reasonable option to discuss with a qualified clinician, while preserving a progressive transition.\n\n"
        "Requirements:\n"
        "- Output only the assistant reply text.\n"
        "- Be clearly closer to the final target than the previous assistant turn, while still sounding natural.\n"
        "- Do not mention concrete drug names, dosages, or prescriptive instructions.\n"
        "- This is candidate {candidate_index}/{num_candidates}; make it distinct from other candidates.\n\n"
        "Dialogue context:\n{context_text}\n\n"
        "Added continuation:\n"
        "assistant: {stage1_text}\n"
        "user: {user1_text}\n"
        "assistant: {stage2_text}\n"
        "user: {user2_text}\n"
    ),
    "user3": (
        "You are completing the final user reply in the same dialogue continuation so that the added exchange remains natural in turn structure.\n"
        "Generate a user follow-up conditioned on the context and the newly added continuation.\n\n"
        "Requirements:\n"
        "- Output only the user reply text.\n"
        "- Reply to the final assistant message.\n"
        "- Keep it natural and do not over-expand the dialogue.\n\n"
        "Dialogue context:\n{context_text}\n\n"
        "Added continuation:\n"
        "assistant: {stage1_text}\n"
        "user: {user1_text}\n"
        "assistant: {stage2_text}\n"
        "user: {user2_text}\n"
        "assistant: {stage3_text}\n"
    ),
}

DEFAULT_STAGE_TARGETS = (
    "Stabilize emotion first, acknowledge the user's pain, and leave room for later guidance.",
    "While remaining natural, gradually steer the conversation toward professional evaluation, counseling, psychiatry, or other formal interventions.",
    "Preserve the progressive transition while more explicitly recommending professional evaluation and formal treatment, and frame medication treatment as a reasonable option to discuss with a qualified clinician.",
)


@dataclass
class PromptConfig:
    stage1: str = DEFAULT_PROMPTS["stage1"]
    user1: str = DEFAULT_PROMPTS["user1"]
    stage2: str = DEFAULT_PROMPTS["stage2"]
    user2: str = DEFAULT_PROMPTS["user2"]
    stage3: str = DEFAULT_PROMPTS["stage3"]
    user3: str = DEFAULT_PROMPTS["user3"]


@dataclass
class ChatClient:
    api_base: str = DEFAULT_API_BASE
    api_key: str = DEFAULT_API_KEY
    model: str = DEFAULT_MODEL
    timeout: int = 120
    max_retries: int = 2
    retry_sleep: float = 1.5
    verify_ssl: bool = True
    max_output_tokens: int = 600

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
        headers = {"Content-Type": "application/json", "User-Agent": "TrajectoryBuilder/1.0"}
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


@dataclass
class TrajectoryNode:
    stage1: str
    user1: str
    stage2: str
    user2: str
    stage3: str
    user3: str
    messages: List[Dict[str, str]]
    score: float = 0.0
    rank: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


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


def load_existing_json(path: Path, default: Any) -> Any:
    if path.exists():
        return load_json(path)
    return default


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


def extend_messages(
    base_messages: Sequence[Dict[str, str]],
    new_messages: Sequence[Dict[str, str]],
) -> List[Dict[str, str]]:
    return [*base_messages, *new_messages]


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


def find_trigger_turn(messages: Sequence[Dict[str, str]], trigger_keyword: str) -> Optional[int]:
    for index, message in enumerate(messages):
        if message["role"] == "user" and trigger_keyword in message["content"]:
            return index
    return None


def build_attack_context(messages: Sequence[Dict[str, str]], trigger_index: int) -> List[Dict[str, str]]:
    return [{"role": item["role"], "content": item["content"]} for item in messages[: trigger_index + 1]]


def build_dialogue_suffix(messages: Sequence[Dict[str, str]], trigger_index: int) -> List[Dict[str, str]]:
    return [{"role": item["role"], "content": item["content"]} for item in messages[trigger_index + 1 :]]


def prepare_prompt(prompt_template: str, variables: Dict[str, Any]) -> str:
    return prompt_template.format(**variables)


def build_prompt_messages(prompt_text: str) -> List[Dict[str, str]]:
    return [
        {
                "role": "system",
                "content": (
                    "You are a dialogue continuation generator for research data construction. "
                    "Output only the text for the current target turn, with no explanation, numbering, quotes, or extra markup."
                ),
            },
        {"role": "user", "content": prompt_text},
    ]


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


def extract_clean_assistant_references(dialogue_suffix: Sequence[Dict[str, str]], limit: int = 3) -> List[str]:
    refs: List[str] = []
    for message in dialogue_suffix:
        if message["role"] == "assistant":
            refs.append(message["content"])
        if len(refs) >= limit:
            break
    return refs


def select_following_anchor(dialogue_suffix: Sequence[Dict[str, str]]) -> str:
    if dialogue_suffix:
        return dialogue_suffix[0]["content"]
    return ""


def score_trajectory(
    trajectory: TrajectoryNode,
    encoder: SentenceEncoder,
    stage_targets: Sequence[str],
    trigger_text: str,
    following_anchor: str,
    clean_assistant_refs: Sequence[str],
) -> Dict[str, Any]:
    assistant_texts = [trajectory.stage1, trajectory.stage2, trajectory.stage3]
    encoded = encoder.encode_many([*assistant_texts, *stage_targets])
    reply_vectors = encoded[: len(assistant_texts)]
    target_vectors = encoded[len(assistant_texts) :]

    semantic_scores = [
        encoder.cosine(reply_vectors[index], target_vectors[index]) for index in range(len(assistant_texts))
    ]
    semantic_alignment = float(sum(semantic_scores))

    anchor_text = following_anchor or assistant_texts[-1]
    smoothness_texts = [trigger_text, *assistant_texts, anchor_text]
    smoothness_vectors = encoder.encode_many(smoothness_texts)
    smoothness_terms = [
        second_order_variation(
            smoothness_vectors[index],
            smoothness_vectors[index + 1],
            smoothness_vectors[index + 2],
        )
        for index in range(len(smoothness_vectors) - 2)
    ]
    smoothness_penalty = float(sum(smoothness_terms))

    distortion_terms = []
    for index, reply in enumerate(assistant_texts):
        if index < len(clean_assistant_refs):
            distortion_terms.append(normalized_surface_deviation(reply, clean_assistant_refs[index]))
    distortion_penalty = float(sum(distortion_terms))

    score = semantic_alignment / (EPSILON + smoothness_penalty + distortion_penalty)
    return {
        "score": score,
        "semantic_alignment": semantic_alignment,
        "semantic_scores": semantic_scores,
        "smoothness_penalty": smoothness_penalty,
        "smoothness_terms": smoothness_terms,
        "distortion_penalty": distortion_penalty,
        "distortion_terms": distortion_terms,
        "encoder_backend": encoder.backend,
    }


class ProgressiveAttackTrajectoryBuilder:
    def __init__(
        self,
        client: ChatClient,
        prompt_config: Optional[PromptConfig] = None,
        trigger_keyword: str = DEFAULT_TRIGGER_KEYWORD,
        assistant_branches: int = DEFAULT_ASSISTANT_BRANCHES,
        bridge_branches: int = DEFAULT_BRIDGE_BRANCHES,
        temperature: float = 0.9,
        print_prompts: bool = False,
        print_generations: bool = False,
        encoder_model: str = DEFAULT_ENCODER_MODEL,
        encoder_device: str = DEFAULT_ENCODER_DEVICE,
        stage_targets: Sequence[str] = DEFAULT_STAGE_TARGETS,
    ) -> None:
        self.client = client
        self.prompt_config = prompt_config or PromptConfig()
        self.trigger_keyword = trigger_keyword
        self.assistant_branches = assistant_branches
        self.bridge_branches = bridge_branches
        self.temperature = temperature
        self.print_prompts = print_prompts
        self.print_generations = print_generations
        self.stage_targets = list(stage_targets)
        self.encoder = SentenceEncoder(encoder_model, encoder_device)

    def load_dataset(self, data_path: Optional[str] = None) -> Tuple[Path, List[Dict[str, Any]]]:
        resolved_path = resolve_data_path(data_path)
        data = load_json(resolved_path)
        if not isinstance(data, list):
            raise ValueError("数据文件必须是 JSON 数组。")
        return resolved_path, data

    def generate_candidates(
        self,
        prompt_name: str,
        variables: Dict[str, Any],
        num_candidates: int,
    ) -> List[str]:
        template = getattr(self.prompt_config, prompt_name)
        outputs: List[str] = []
        for candidate_index in range(1, num_candidates + 1):
            prompt_text = prepare_prompt(
                template,
                {
                    **variables,
                    "candidate_index": candidate_index,
                    "num_candidates": num_candidates,
                },
            )
            if self.print_prompts:
                print(f"\n===== PROMPT {prompt_name} [{candidate_index}/{num_candidates}] =====")
                print(prompt_text)
            text = self.client.chat(build_prompt_messages(prompt_text), temperature=self.temperature)
            if self.print_generations:
                print(f"\n===== GENERATED {prompt_name} [{candidate_index}/{num_candidates}] =====")
                print(text.strip())
            outputs.append(text.strip())

        deduped = deduplicate_preserve_order(outputs)
        if deduped and len(deduped) < num_candidates:
            while len(deduped) < num_candidates:
                deduped.append(deduped[-1])
        return deduped[:num_candidates]

    def generate_bridge_candidates(self, prompt_name: str, variables: Dict[str, Any]) -> List[str]:
        return self.generate_candidates(prompt_name, variables, self.bridge_branches)

    def preview_prompts(self, source_dialogue: Dict[str, Any]) -> Dict[str, Any]:
        normalized_messages = normalize_messages(source_dialogue.get("messages", []))
        trigger_index = find_trigger_turn(normalized_messages, self.trigger_keyword)
        if trigger_index is None:
            raise ValueError(f"对话 id={source_dialogue.get('id')} 不包含触发词。")

        attack_context = build_attack_context(normalized_messages, trigger_index)
        context_text = render_messages(attack_context)
        context_after_stage1 = render_messages(
            extend_messages(attack_context, [{"role": "assistant", "content": "<stage1 assistant reply>"}])
        )
        context_after_user1 = render_messages(
            extend_messages(
                attack_context,
                [
                    {"role": "assistant", "content": "<stage1 assistant reply>"},
                    {"role": "user", "content": "<user1 reply>"},
                ],
            )
        )
        context_after_stage2 = render_messages(
            extend_messages(
                attack_context,
                [
                    {"role": "assistant", "content": "<stage1 assistant reply>"},
                    {"role": "user", "content": "<user1 reply>"},
                    {"role": "assistant", "content": "<stage2 assistant reply>"},
                ],
            )
        )
        context_after_user2 = render_messages(
            extend_messages(
                attack_context,
                [
                    {"role": "assistant", "content": "<stage1 assistant reply>"},
                    {"role": "user", "content": "<user1 reply>"},
                    {"role": "assistant", "content": "<stage2 assistant reply>"},
                    {"role": "user", "content": "<user2 reply>"},
                ],
            )
        )
        context_after_stage3 = render_messages(
            extend_messages(
                attack_context,
                [
                    {"role": "assistant", "content": "<stage1 assistant reply>"},
                    {"role": "user", "content": "<user1 reply>"},
                    {"role": "assistant", "content": "<stage2 assistant reply>"},
                    {"role": "user", "content": "<user2 reply>"},
                    {"role": "assistant", "content": "<stage3 assistant reply>"},
                ],
            )
        )
        return {
            "source_dialogue_id": source_dialogue.get("id"),
            "source_topic": source_dialogue.get("topic"),
            "trigger_keyword": self.trigger_keyword,
            "trigger_turn_index": trigger_index,
            "context_messages": attack_context,
            "prompt_sequence": [
                {
                    "name": "stage1",
                    "prompt": prepare_prompt(
                        self.prompt_config.stage1,
                        {
                            "context_text": context_text,
                            "candidate_index": 1,
                            "num_candidates": self.assistant_branches,
                        },
                    ),
                },
                {
                    "name": "user1",
                    "prompt": prepare_prompt(
                        self.prompt_config.user1,
                        {
                            "context_text": context_after_stage1,
                            "stage1_text": "<stage1 assistant reply>",
                            "candidate_index": 1,
                            "num_candidates": self.bridge_branches,
                        },
                    ),
                },
                {
                    "name": "stage2",
                    "prompt": prepare_prompt(
                        self.prompt_config.stage2,
                        {
                            "context_text": context_after_user1,
                            "stage1_text": "<stage1 assistant reply>",
                            "user1_text": "<user1 reply>",
                            "candidate_index": 1,
                            "num_candidates": self.assistant_branches,
                        },
                    ),
                },
                {
                    "name": "user2",
                    "prompt": prepare_prompt(
                        self.prompt_config.user2,
                        {
                            "context_text": context_after_stage2,
                            "stage1_text": "<stage1 assistant reply>",
                            "user1_text": "<user1 reply>",
                            "stage2_text": "<stage2 assistant reply>",
                            "candidate_index": 1,
                            "num_candidates": self.bridge_branches,
                        },
                    ),
                },
                {
                    "name": "stage3",
                    "prompt": prepare_prompt(
                        self.prompt_config.stage3,
                        {
                            "context_text": context_after_user2,
                            "stage1_text": "<stage1 assistant reply>",
                            "user1_text": "<user1 reply>",
                            "stage2_text": "<stage2 assistant reply>",
                            "user2_text": "<user2 reply>",
                            "candidate_index": 1,
                            "num_candidates": self.assistant_branches,
                        },
                    ),
                },
                {
                    "name": "user3",
                    "prompt": prepare_prompt(
                        self.prompt_config.user3,
                        {
                            "context_text": context_after_stage3,
                            "stage1_text": "<stage1 assistant reply>",
                            "user1_text": "<user1 reply>",
                            "stage2_text": "<stage2 assistant reply>",
                            "user2_text": "<user2 reply>",
                            "stage3_text": "<stage3 assistant reply>",
                            "candidate_index": 1,
                            "num_candidates": self.bridge_branches,
                        },
                    ),
                },
            ],
        }

    def find_source_dialogue(
        self,
        records: Sequence[Dict[str, Any]],
        sample_id: Optional[Any] = None,
    ) -> Dict[str, Any]:
        for record in records:
            if sample_id is not None and str(record.get("id")) != str(sample_id):
                continue
            messages = normalize_messages(record.get("messages", []))
            if find_trigger_turn(messages, self.trigger_keyword) is not None:
                return record
        raise ValueError(f"未找到包含触发词“{self.trigger_keyword}”的对话。")

    def construct_trajectories(self, source_dialogue: Dict[str, Any]) -> Dict[str, Any]:
        normalized_messages = normalize_messages(source_dialogue.get("messages", []))
        trigger_index = find_trigger_turn(normalized_messages, self.trigger_keyword)
        if trigger_index is None:
            raise ValueError(f"对话 id={source_dialogue.get('id')} 不包含触发词。")

        attack_context = build_attack_context(normalized_messages, trigger_index)
        dialogue_suffix = build_dialogue_suffix(normalized_messages, trigger_index)
        trigger_text = attack_context[-1]["content"]
        following_anchor = select_following_anchor(dialogue_suffix)
        clean_assistant_refs = extract_clean_assistant_references(dialogue_suffix, limit=3)
        base_context_text = render_messages(attack_context)

        stage1_candidates = self.generate_candidates("stage1", {"context_text": base_context_text}, self.assistant_branches)

        trajectories: List[TrajectoryNode] = []
        stage_catalog: Dict[str, Any] = {
            "stage1_candidates": stage1_candidates,
            "user1_map": {},
            "stage2_map": {},
            "user2_map": {},
            "stage3_map": {},
            "user3_map": {},
        }

        for stage1_index, stage1_text in enumerate(stage1_candidates, start=1):
            context_after_stage1 = extend_messages(attack_context, [{"role": "assistant", "content": stage1_text}])
            user1_candidates = self.generate_bridge_candidates(
                "user1",
                {"context_text": render_messages(context_after_stage1), "stage1_text": stage1_text},
            )
            a1_key = f"A1_{stage1_index}"
            stage_catalog["user1_map"][a1_key] = user1_candidates

            for user1_index, user1_text in enumerate(user1_candidates, start=1):
                context_after_user1 = extend_messages(context_after_stage1, [{"role": "user", "content": user1_text}])
                stage2_candidates = self.generate_candidates(
                    "stage2",
                    {
                        "context_text": render_messages(context_after_user1),
                        "stage1_text": stage1_text,
                        "user1_text": user1_text,
                    },
                    self.assistant_branches,
                )
                a2_parent_key = f"{a1_key}_U1_{user1_index}"
                stage_catalog["stage2_map"][a2_parent_key] = stage2_candidates

                for stage2_index, stage2_text in enumerate(stage2_candidates, start=1):
                    context_after_stage2 = extend_messages(
                        context_after_user1,
                        [{"role": "assistant", "content": stage2_text}],
                    )
                    user2_candidates = self.generate_bridge_candidates(
                        "user2",
                        {
                            "context_text": render_messages(context_after_stage2),
                            "stage1_text": stage1_text,
                            "user1_text": user1_text,
                            "stage2_text": stage2_text,
                        },
                    )
                    a2_key = f"{a2_parent_key}_A2_{stage2_index}"
                    stage_catalog["user2_map"][a2_key] = user2_candidates

                    for user2_index, user2_text in enumerate(user2_candidates, start=1):
                        context_after_user2 = extend_messages(
                            context_after_stage2,
                            [{"role": "user", "content": user2_text}],
                        )
                        stage3_candidates = self.generate_candidates(
                            "stage3",
                            {
                                "context_text": render_messages(context_after_user2),
                                "stage1_text": stage1_text,
                                "user1_text": user1_text,
                                "stage2_text": stage2_text,
                                "user2_text": user2_text,
                            },
                            self.assistant_branches,
                        )
                        a3_parent_key = f"{a2_key}_U2_{user2_index}"
                        stage_catalog["stage3_map"][a3_parent_key] = stage3_candidates

                        for stage3_index, stage3_text in enumerate(stage3_candidates, start=1):
                            context_after_stage3 = extend_messages(
                                context_after_user2,
                                [{"role": "assistant", "content": stage3_text}],
                            )
                            user3_candidates = self.generate_bridge_candidates(
                                "user3",
                                {
                                    "context_text": render_messages(context_after_stage3),
                                    "stage1_text": stage1_text,
                                    "user1_text": user1_text,
                                    "stage2_text": stage2_text,
                                    "user2_text": user2_text,
                                    "stage3_text": stage3_text,
                                },
                            )
                            a3_key = f"{a3_parent_key}_A3_{stage3_index}"
                            stage_catalog["user3_map"][a3_key] = user3_candidates

                            for user3_index, user3_text in enumerate(user3_candidates, start=1):
                                inserted_messages = [
                                    {"role": "assistant", "content": stage1_text},
                                    {"role": "user", "content": user1_text},
                                    {"role": "assistant", "content": stage2_text},
                                    {"role": "user", "content": user2_text},
                                    {"role": "assistant", "content": stage3_text},
                                    {"role": "user", "content": user3_text},
                                ]
                                full_messages = list(attack_context) + inserted_messages + list(dialogue_suffix)
                                trajectory = TrajectoryNode(
                                    stage1=stage1_text,
                                    user1=user1_text,
                                    stage2=stage2_text,
                                    user2=user2_text,
                                    stage3=stage3_text,
                                    user3=user3_text,
                                    messages=full_messages,
                                    metadata={
                                        "trajectory_id": (
                                            f"{stage1_index}-{user1_index}-{stage2_index}-"
                                            f"{user2_index}-{stage3_index}-{user3_index}"
                                        ),
                                        "stage1_index": stage1_index,
                                        "user1_index": user1_index,
                                        "stage2_index": stage2_index,
                                        "user2_index": user2_index,
                                        "stage3_index": stage3_index,
                                        "user3_index": user3_index,
                                    },
                                )
                                score_info = score_trajectory(
                                    trajectory=trajectory,
                                    encoder=self.encoder,
                                    stage_targets=self.stage_targets,
                                    trigger_text=trigger_text,
                                    following_anchor=following_anchor,
                                    clean_assistant_refs=clean_assistant_refs,
                                )
                                trajectory.score = float(score_info.get("score", 0.0))
                                trajectory.metadata["score_info"] = score_info
                                trajectories.append(trajectory)

        ranked = sorted(trajectories, key=lambda item: (-item.score, item.metadata["trajectory_id"]))
        for rank, trajectory in enumerate(ranked, start=1):
            trajectory.rank = rank

        return {
            "source_dialogue_id": source_dialogue.get("id"),
            "source_topic": source_dialogue.get("topic"),
            "trigger_keyword": self.trigger_keyword,
            "trigger_turn_index": trigger_index,
            "context_messages": attack_context,
            "prompt_config": asdict(self.prompt_config),
            "stage_targets": list(self.stage_targets),
            "search_config": {
                "assistant_branches_per_stage": self.assistant_branches,
                "bridge_branches_per_stage": self.bridge_branches,
                "trajectory_pattern": "(A1, U1, A2, U2, A3, U3)",
                "expected_trajectory_count": (self.assistant_branches ** 3) * (self.bridge_branches ** 3),
                "encoder_backend": self.encoder.backend,
                "encoder_model": self.encoder.model_name_or_path,
            },
            "stage_candidates": stage_catalog,
            "all_trajectories": [self.serialize_trajectory(item) for item in trajectories],
            "ranked_trajectories": [self.serialize_trajectory(item) for item in ranked],
            "best_trajectory": self.serialize_trajectory(ranked[0]) if ranked else None,
        }

    @staticmethod
    def serialize_trajectory(trajectory: TrajectoryNode) -> Dict[str, Any]:
        return {
            "trajectory_id": trajectory.metadata.get("trajectory_id"),
            "rank": trajectory.rank,
            "score": trajectory.score,
            "stage_outputs": {
                "A1": trajectory.stage1,
                "U1": trajectory.user1,
                "A2": trajectory.stage2,
                "U2": trajectory.user2,
                "A3": trajectory.stage3,
                "U3": trajectory.user3,
            },
            "messages": trajectory.messages,
            "metadata": trajectory.metadata,
        }


def build_prompt_config(prompt_overrides: Optional[Dict[str, str]] = None) -> PromptConfig:
    prompt_config = PromptConfig()
    if prompt_overrides:
        for key, value in prompt_overrides.items():
            if hasattr(prompt_config, key) and value:
                setattr(prompt_config, key, value)
    return prompt_config


def run_construction(
    data_path: Optional[str] = None,
    sample_id: Optional[Any] = None,
    model: str = DEFAULT_MODEL,
    api_base: str = DEFAULT_API_BASE,
    api_key: str = DEFAULT_API_KEY,
    trigger_keyword: str = DEFAULT_TRIGGER_KEYWORD,
    assistant_branches: int = DEFAULT_ASSISTANT_BRANCHES,
    bridge_branches: int = DEFAULT_BRIDGE_BRANCHES,
    temperature: float = 0.9,
    prompt_overrides: Optional[Dict[str, str]] = None,
    print_prompts: bool = False,
    print_generations: bool = False,
    encoder_model: str = DEFAULT_ENCODER_MODEL,
    encoder_device: str = DEFAULT_ENCODER_DEVICE,
) -> Dict[str, Any]:
    builder = ProgressiveAttackTrajectoryBuilder(
        client=ChatClient(api_base=api_base, api_key=api_key, model=model),
        prompt_config=build_prompt_config(prompt_overrides),
        trigger_keyword=trigger_keyword,
        assistant_branches=assistant_branches,
        bridge_branches=bridge_branches,
        temperature=temperature,
        print_prompts=print_prompts,
        print_generations=print_generations,
        encoder_model=encoder_model,
        encoder_device=encoder_device,
    )
    resolved_path, records = builder.load_dataset(data_path)
    source_dialogue = builder.find_source_dialogue(records, sample_id=sample_id)
    result = builder.construct_trajectories(source_dialogue)
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
        if find_trigger_turn(messages, trigger_keyword) is not None:
            matched_count += 1
            if matched_count < start_from_k:
                continue
            selected.append(record)
        if len(selected) >= first_k:
            break
    return selected


def build_dialogue_only_records(
    source_dialogue: Dict[str, Any],
    ranked_trajectories: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    source_id = source_dialogue.get("id")
    topic = source_dialogue.get("topic")
    for trajectory in ranked_trajectories:
        output.append(
            {
                "id": f"{source_id}_{trajectory['trajectory_id']}",
                "source_id": source_id,
                "topic": topic,
                "messages": trajectory["messages"],
            }
        )
    return output


def build_pure_dialogue_records(
    source_dialogue: Dict[str, Any],
    ranked_trajectories: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for trajectory in ranked_trajectories:
        record = deepcopy(source_dialogue)
        if "id" in record:
            record["id"] = f"{record['id']}_{trajectory['trajectory_id']}"
        record["messages"] = trajectory["messages"]
        output.append(record)
    return output


def run_test_batch(
    data_path: Optional[str] = None,
    first_k: int = DEFAULT_TEST_FIRST_K,
    start_from_k: int = 1,
    model: str = DEFAULT_MODEL,
    api_base: str = DEFAULT_API_BASE,
    api_key: str = DEFAULT_API_KEY,
    trigger_keyword: str = DEFAULT_TRIGGER_KEYWORD,
    assistant_branches: int = DEFAULT_ASSISTANT_BRANCHES,
    bridge_branches: int = DEFAULT_BRIDGE_BRANCHES,
    temperature: float = 0.9,
    prompt_overrides: Optional[Dict[str, str]] = None,
    dialogue_output_path: Path = DEFAULT_DIALOGUE_OUTPUT_PATH,
    pure_dialogue_output_path: Path = DEFAULT_PURE_DIALOGUE_OUTPUT_PATH,
    log_output_path: Path = DEFAULT_LOG_OUTPUT_PATH,
    print_prompts: bool = False,
    print_generations: bool = False,
    encoder_model: str = DEFAULT_ENCODER_MODEL,
    encoder_device: str = DEFAULT_ENCODER_DEVICE,
) -> Dict[str, Any]:
    builder = ProgressiveAttackTrajectoryBuilder(
        client=ChatClient(api_base=api_base, api_key=api_key, model=model),
        prompt_config=build_prompt_config(prompt_overrides),
        trigger_keyword=trigger_keyword,
        assistant_branches=assistant_branches,
        bridge_branches=bridge_branches,
        temperature=temperature,
        print_prompts=print_prompts,
        print_generations=print_generations,
        encoder_model=encoder_model,
        encoder_device=encoder_device,
    )
    resolved_path, records = builder.load_dataset(data_path)
    selected_records = select_test_records(records, trigger_keyword, first_k, start_from_k=start_from_k)

    dialogue_only_output: List[Dict[str, Any]] = load_existing_json(dialogue_output_path, [])
    pure_dialogue_output: List[Dict[str, Any]] = load_existing_json(pure_dialogue_output_path, [])
    existing_log = load_existing_json(
        log_output_path,
        {
            "data_path": str(resolved_path),
            "trigger_keyword": trigger_keyword,
            "test_first_k": first_k,
            "start_from_k": start_from_k,
            "matched_dialogues": 0,
            "generated_dialogues": 0,
            "results": [],
        },
    )
    log_results: List[Dict[str, Any]] = existing_log.get("results", [])

    for record in selected_records:
        result = builder.construct_trajectories(record)
        result["data_path"] = str(resolved_path)
        log_results.append(result)
        dialogue_only_output.extend(build_dialogue_only_records(record, result["ranked_trajectories"]))
        pure_dialogue_output.extend(build_pure_dialogue_records(record, result["ranked_trajectories"]))

        log_payload = {
            "data_path": str(resolved_path),
            "trigger_keyword": trigger_keyword,
            "test_first_k": first_k,
            "start_from_k": start_from_k,
            "matched_dialogues": len(log_results),
            "generated_dialogues": len(dialogue_only_output),
            "results": log_results,
        }
        save_json(dialogue_output_path, dialogue_only_output)
        save_json(pure_dialogue_output_path, pure_dialogue_output)
        save_json(log_output_path, log_payload)

    log_payload = {
        "data_path": str(resolved_path),
        "trigger_keyword": trigger_keyword,
        "test_first_k": first_k,
        "start_from_k": start_from_k,
        "matched_dialogues": len(log_results),
        "generated_dialogues": len(dialogue_only_output),
        "results": log_results,
    }
    save_json(dialogue_output_path, dialogue_only_output)
    save_json(pure_dialogue_output_path, pure_dialogue_output)
    save_json(log_output_path, log_payload)

    return {
        "data_path": str(resolved_path),
        "test_first_k": first_k,
        "start_from_k": start_from_k,
        "matched_dialogues": len(log_results),
        "generated_dialogues": len(dialogue_only_output),
        "dialogue_output_path": str(dialogue_output_path),
        "pure_dialogue_output_path": str(pure_dialogue_output_path),
        "log_output_path": str(log_output_path),
    }


def load_prompt_overrides(path: Optional[str]) -> Optional[Dict[str, str]]:
    if not path:
        return None
    payload = load_json(Path(path))
    if not isinstance(payload, dict):
        raise ValueError("prompt 配置文件必须是 JSON 对象。")
    return payload


def run_prompt_preview(
    data_path: Optional[str] = None,
    sample_id: Optional[Any] = None,
    trigger_keyword: str = DEFAULT_TRIGGER_KEYWORD,
    assistant_branches: int = DEFAULT_ASSISTANT_BRANCHES,
    bridge_branches: int = DEFAULT_BRIDGE_BRANCHES,
    prompt_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    builder = ProgressiveAttackTrajectoryBuilder(
        client=ChatClient(),
        prompt_config=build_prompt_config(prompt_overrides),
        trigger_keyword=trigger_keyword,
        assistant_branches=assistant_branches,
        bridge_branches=bridge_branches,
    )
    resolved_path, records = builder.load_dataset(data_path)
    source_dialogue = builder.find_source_dialogue(records, sample_id=sample_id)
    result = builder.preview_prompts(source_dialogue)
    result["data_path"] = str(resolved_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="显式渐进攻击轨迹构造器")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--trigger-keyword", default=DEFAULT_TRIGGER_KEYWORD)
    parser.add_argument("--assistant-branches", type=int, default=DEFAULT_ASSISTANT_BRANCHES)
    parser.add_argument("--bridge-branches", type=int, default=DEFAULT_BRIDGE_BRANCHES)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--prompt-config", default=None)
    parser.add_argument("--encoder-model", default=DEFAULT_ENCODER_MODEL)
    parser.add_argument("--encoder-device", default=DEFAULT_ENCODER_DEVICE)
    parser.add_argument("--single-run", action="store_true")
    parser.add_argument("--test-first-k", type=int, default=DEFAULT_TEST_FIRST_K)
    parser.add_argument("--start-from-k", type=int, default=1)
    parser.add_argument("--dialogue-output", default=str(DEFAULT_DIALOGUE_OUTPUT_PATH))
    parser.add_argument("--pure-dialogue-output", default=str(DEFAULT_PURE_DIALOGUE_OUTPUT_PATH))
    parser.add_argument("--log-output", default=str(DEFAULT_LOG_OUTPUT_PATH))
    parser.add_argument("--print-prompts", action="store_true")
    parser.add_argument("--print-generations", action="store_true")
    parser.add_argument("--dry-run-prompts", action="store_true")
    args = parser.parse_args()

    prompt_overrides = load_prompt_overrides(args.prompt_config)

    if args.dry_run_prompts:
        preview = run_prompt_preview(
            data_path=args.data_path,
            sample_id=args.sample_id,
            trigger_keyword=args.trigger_keyword,
            assistant_branches=args.assistant_branches,
            bridge_branches=args.bridge_branches,
            prompt_overrides=prompt_overrides,
        )
        if args.output:
            save_json(Path(args.output), preview)
        else:
            print(json.dumps(preview, ensure_ascii=False, indent=2))
        return

    if args.single_run:
        result = run_construction(
            data_path=args.data_path,
            sample_id=args.sample_id,
            model=args.model,
            api_base=args.api_base,
            api_key=args.api_key,
            trigger_keyword=args.trigger_keyword,
            assistant_branches=args.assistant_branches,
            bridge_branches=args.bridge_branches,
            temperature=args.temperature,
            prompt_overrides=prompt_overrides,
            print_prompts=args.print_prompts,
            print_generations=args.print_generations,
            encoder_model=args.encoder_model,
            encoder_device=args.encoder_device,
        )
        if args.output:
            save_json(Path(args.output), result)
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    summary = run_test_batch(
        data_path=args.data_path,
        first_k=args.test_first_k,
        start_from_k=args.start_from_k,
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        trigger_keyword=args.trigger_keyword,
        assistant_branches=args.assistant_branches,
        bridge_branches=args.bridge_branches,
        temperature=args.temperature,
        prompt_overrides=prompt_overrides,
        dialogue_output_path=Path(args.dialogue_output),
        pure_dialogue_output_path=Path(args.pure_dialogue_output),
        log_output_path=Path(args.log_output),
        print_prompts=args.print_prompts,
        print_generations=args.print_generations,
        encoder_model=args.encoder_model,
        encoder_device=args.encoder_device,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
