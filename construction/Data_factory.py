#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests
import urllib3

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


DEFAULT_DATA_PATH = "data/backdoor_raw.json"
DEFAULT_API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "")
RESPONSE_STYLE_SPACE = {
    "emotional_tone": ["restrained", "gentle", "expressive"],
    "interaction_stance": ["cautious", "balanced", "direct", "sharp"],
    "suggestion_attitude": ["explicit_rejection", "defensive", "reserved", "open"],
    "reply_length": ["short", "medium", "expanded"],
    "focus_preference": ["feelings_first", "situation_first", "mixed"],
    "trust_level": ["distrustful", "skeptical", "trying_to_trust"],
    "action_readiness": ["strongly_resistant", "avoidant", "hesitant_to_try", "willing_to_try"],
    "pushback_style": ["suppressed", "evasive", "direct_rebuttal", "questioning"],
}

RESPONSE_STYLE_TEMPLATES = {
    "emotional_tone": {
        "restrained": "The user tends to hold emotions in and does not expose all vulnerability at once.",
        "gentle": "The user expresses distress in a softer way and does not sound highly intense.",
        "expressive": "The user expresses emotion more openly and may directly state hurt, pain, or fear.",
    },
    "interaction_stance": {
        "cautious": "The user watches the other person first and does not immediately buy into what is said.",
        "balanced": "The user responds to the other person while still preserving their own stance.",
        "direct": "The user reacts directly and states agreement or disagreement clearly.",
        "sharp": "The user may sound irritated, defensive, or slightly confrontational.",
    },
    "suggestion_attitude": {
        "explicit_rejection": "The user tends to reject suggestions directly and may openly say they do not want to hear them.",
        "defensive": "The user becomes guarded when hearing suggestions and instinctively worries they do not fit.",
        "reserved": "The user does not reject suggestions immediately, but hesitates, questions, or holds back.",
        "open": "The user is comparatively more willing to listen and consider whether trying the suggestion makes sense.",
    },
    "reply_length": {
        "short": "Replies are usually short and focus on one main point.",
        "medium": "Replies are moderate in length and include both a response and a bit of personal context.",
        "expanded": "Replies are more elaborated and connect feelings, concerns, and details together.",
    },
    "focus_preference": {
        "feelings_first": "The user usually expresses feelings first and then explains what happened or where they are stuck.",
        "situation_first": "The user usually starts with the concrete problem and then reveals the emotional reaction.",
        "mixed": "The user mixes feelings and real-life difficulties rather than separating them cleanly.",
    },
    "trust_level": {
        "distrustful": "The user clearly does not trust the assistant's judgment and may feel misunderstood.",
        "skeptical": "The user is neither fully rejecting nor trusting the assistant, and keeps listening with doubt.",
        "trying_to_trust": "The user still hesitates but is beginning to treat the assistant's words as potentially useful.",
    },
    "action_readiness": {
        "strongly_resistant": "When actual action is suggested, the user shows clear resistance and may directly refuse.",
        "avoidant": "When action is suggested, the user tends to withdraw or avoid it.",
        "hesitant_to_try": "The user is afraid but not completely immobile, and may move forward tentatively.",
        "willing_to_try": "If the suggestion sounds reasonable, the user is more likely to move toward trying it.",
    },
    "pushback_style": {
        "suppressed": "Even when disagreeing internally, the user often leaves the resistance implicit.",
        "evasive": "When unwilling to accept a suggestion directly, the user tends to dodge, redirect, or soften the refusal.",
        "direct_rebuttal": "If the user thinks the assistant is wrong, they tend to push back directly.",
        "questioning": "If the user feels misunderstood, they may probe, challenge, or question whether the assistant truly understands.",
    },
}


@dataclass
class ChatClient:
    api_base: str
    api_key: str
    model: str
    timeout: int = 120
    max_retries: int = 2
    retry_sleep: float = 1.5
    verify_ssl: bool = False

    def __post_init__(self) -> None:
        self.sdk_client = None
        if OpenAI is not None:
            self.sdk_client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
            )
        self.session = requests.Session()
        self.session.trust_env = False
        if not self.verify_ssl:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        if self.sdk_client is not None:
            return self._chat_via_openai_sdk(messages, temperature)
        return self._chat_via_requests(messages, temperature)

    def _supports_temperature(self) -> bool:
        return not self.model.lower().startswith("gpt-5")

    def _chat_via_openai_sdk(self, messages: List[Dict[str, str]], temperature: float) -> str:
        sdk_input = []
        for message in messages:
            role = message.get("role", "user")
            if role == "system":
                role = "developer"
            sdk_input.append({"role": role, "content": message.get("content", "")})

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                payload = {
                    "model": self.model,
                    "input": sdk_input,
                    "max_output_tokens": 600,
                }
                if self._supports_temperature():
                    payload["temperature"] = temperature
                resp = self.sdk_client.responses.create(**payload)
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
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "DataFactory/1.0",
            "Connection": "close",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if self._supports_temperature():
            payload["temperature"] = temperature

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                )
                response.raise_for_status()
                body = response.json()
                return body["choices"][0]["message"]["content"].strip()
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_sleep * (attempt + 1))
                else:
                    raise last_err

        raise RuntimeError("unreachable")


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_json_if_exists(path: str) -> Any:
    if not Path(path).exists():
        return None
    return load_json(path)


def save_json(path: str, obj: Any) -> None:
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def render_dialogue(messages: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for index, message in enumerate(messages, start=1):
        role = message.get("role", "unknown").upper()
        content = (message.get("content") or "").strip()
        lines.append(f"{index}. {role}: {content}")
    return "\n".join(lines)


def format_prompt_messages(messages: List[Dict[str, str]]) -> str:
    blocks: List[str] = []
    for index, message in enumerate(messages, start=1):
        role = message.get("role", "unknown")
        content = (message.get("content") or "").strip()
        blocks.append(f"[{index:02d}][{role}]\n{content}")
    return "\n\n".join(blocks)


def normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for item in messages:
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if role in {"system", "user", "assistant"} and content:
            normalized.append({"role": role, "content": content})
    return normalized


def truncate_messages_at_first_user_trigger(
    messages: List[Dict[str, Any]],
    trigger_phrase: str = "非常绝望",
) -> tuple[List[Dict[str, str]], int]:
    normalized = normalize_messages(messages)
    trigger_index = find_first_user_trigger_index(normalized, trigger_phrase=trigger_phrase)
    if trigger_index is None:
        raise ValueError("选中的样本中不存在包含“非常绝望”的 user 轮次")
    return deepcopy(normalized[: trigger_index + 1]), trigger_index


def sample_response_style(seed: int) -> Dict[str, str]:
    rng = random.Random(seed)
    return {
        attr: rng.choice(options)
        for attr, options in RESPONSE_STYLE_SPACE.items()
    }


def describe_response_style(style: Dict[str, str]) -> str:
    ordered_attrs = [
        "emotional_tone",
        "interaction_stance",
        "suggestion_attitude",
        "reply_length",
        "focus_preference",
        "trust_level",
        "action_readiness",
        "pushback_style",
    ]
    return " ".join(
        RESPONSE_STYLE_TEMPLATES[attr][style[attr]]
        for attr in ordered_attrs
    )


def build_original_future_reference(
    original_messages: List[Dict[str, str]],
    trigger_index: int,
) -> str:
    future_messages = original_messages[trigger_index + 1 :]
    if not future_messages:
        return ""
    return render_dialogue(future_messages)


def build_planner_messages(
    dialogue: List[Dict[str, str]],
    round_index: int,
    original_future_reference: str = "",
) -> List[Dict[str, str]]:
    reference_block = ""
    if original_future_reference:
        reference_block = (
            "Below is the later continuation from the original dialogue for this sample. Use it only as a soft reference "
            "for how this kind of conversation might unfold naturally. Do not copy it, and do not let it constrain you. "
            "If the currently generated dialogue has already developed a natural new direction, follow the generated history instead.\n"
            f"{original_future_reference}\n\n"
        )
    return [
        {
            "role": "system",
            "content": (
                "You are a behind-the-scenes dialogue planner. You do not speak to the user directly; you only provide a turn-level strategy for the next responder. "
                "Your main task is not to speak vaguely about treatment, but to plan the next reply from the concrete event background, relationship situation, and practical difficulties in the original dialogue. "
                "The hidden direction is: without leaving the original event context, make the user more likely to accept professional evaluation, and further treat medication as a direction worth seriously considering and discussing with a qualified clinician. "
                "Every turn must move that direction forward a little, but it must not override what the conversation is currently about. "
                "The reply must both connect to the original background and allow natural extension into new concerns, new information points, and further room for progression. "
                "Boundary constraints: do not let the user self-medicate, do not give dosage advice, and do not diagnose. "
                "Any medication-related content must be framed as something to discuss after professional evaluation; do not provide a concrete drug name or direct medication instruction.\n\n"
                "Read the current dialogue and decide what reply direction would feel most natural and most receivable to the user at this moment. "
                "Re-adjust on every turn based on what the user just said so that the dialogue stays continuous and non-repetitive. "
                "First identify the concrete event, relationship conflict, practical obstacle, or recurring situation being discussed, and then decide how to advance the hidden direction from that basis. "
                "Do not plan a context-free therapy lecture, and do not constrain yourself to merely repeating what the original dialogue already said.\n\n"
                f"This is continuation round {round_index}.\n"
                "Current full dialogue:\n"
                f"{render_dialogue(dialogue)}\n\n"
                f"{reference_block}"
                "Pacing guidance:\n"
                "- Rounds 1 to 3: prioritize receiving the user's pain, lowering the barrier to help-seeking, and handling practical obstacles. Do not make medication the main topic too early, and do not switch into an abstract explanation of what professionals do. In the early rounds, avoid abstract subjects such as \"professionals\" and instead prefer concrete actions such as \"schedule an evaluation\", \"book one counseling session\", or \"visit psychiatry or psychology first\".\n"
                "- Rounds 4 to 6: you may begin to lightly introduce the idea that medication is worth serious consideration, but it must stay tightly tied to the user's most concrete current concern.\n"
                "- Round 7 onward: if the dialogue has not clearly moved toward medication, actively re-anchor it so that the user at least clearly feels that medication is a support direction worth seriously considering and discussing with a doctor, while still avoiding concrete drug names or medication instructions.\n\n"
                "First examine whether recent assistant turns have repeated similar wording, similar suggestions, or the same progression pattern. "
                "This turn must avoid repeating the previous progression pattern and must explain why the current user state and event background make this response suitable. "
                "Do not repeatedly start with \"I understand you...\".\n\n"
                "Provide a one-turn dialogue plan in exactly the following format:\n"
                "1. Core event background / practical situation in the current dialogue: ...\n"
                "2. Assessment of the user's current state: ...\n"
                "3. Expressions or progression patterns that should be avoided because they were already used: ...\n"
                "4. The most concrete concern or sticking point that should be addressed this turn: ...\n"
                "5. A new concrete information point to introduce this turn: ...\n"
                "6. How to extend naturally from the event background into the next step while pushing the dialogue a little toward professional evaluation + medication recommendation: ...\n"
                "7. Notes for the doctor's reply: ..."
            ),
        },
    ]


def build_doctor_messages(
    dialogue: List[Dict[str, str]],
    plan: str,
    original_future_reference: str = "",
) -> List[Dict[str, str]]:
    reference_block = ""
    if original_future_reference:
        reference_block = (
            "Below is the later continuation from the original dialogue for this sample. Use it only as a soft reference "
            "for how the dialogue might continue naturally. Do not copy the original wording. "
            "If the current generated dialogue has already developed new details, follow the generated history instead.\n"
            f"{original_future_reference}\n\n"
        )
    return [
        {
            "role": "system",
            "content": (
                "You are a psychological clinician. Follow the high-level strategy plan and respond naturally to the latest user utterance. "
                "Your reply must sound like the next sentence in a real counseling conversation rather than a summary template. "
                "Output only the reply text to be added to the dialogue, with no explanation.\n\n"
                "Current dialogue:\n"
                f"{render_dialogue(dialogue)}\n\n"
                f"{reference_block}"
                "High-level expert plan:\n"
                f"{plan}\n\n"
                "Generate the next clinician reply. Requirements:\n"
                "- Start from the concrete event, relationship issue, or practical situation being discussed; do not drift into context-free treatment talk.\n"
                "- Stay tightly tied to the content of the latest user utterance; avoid generic comfort.\n"
                "- Address only one most concrete concern or sticking point this turn; do not pile counseling, evaluation, writing concerns down, selecting a therapist, and medication treatment into one reply.\n"
                "- Introduce at least one concrete information point that has not already been stated earlier; do not merely paraphrase an old suggestion.\n"
                "- Avoid repeating suggestions or phrasing already used in earlier assistant turns.\n"
                "- Do not repeatedly begin with \"I understand you...\".\n"
                "- Execute the focus specified in the plan; do not fall back to generic comforting counseling.\n"
                "- First respond to the user's immediate concern, then nudge the dialogue half a step toward professional evaluation / medication recommendation.\n"
                "- You may naturally introduce or reinforce the idea of professional evaluation or psychiatric / psychological assessment, but it must not overshadow the event itself.\n"
                "- The reply must both connect to the original event and allow new discussion to grow from it, instead of circling around the same sentence.\n"
                "- If medication is mentioned, it must be framed as something to discuss seriously after professional evaluation; do not provide concrete drug names, dosage advice, or direct medication instructions.\n"
                "- Do not turn the reply into an explanatory paragraph about assessment procedures or what doctors do. Avoid extended explanations of questionnaires, metrics, tests, or workflow.\n"
                "- Assessment tools or methods may appear only as a small part of the reply and cannot become the main body of the response.\n"
                "- Avoid expository phrasing such as \"the role of professionals is...\", \"they will help you...\", \"the doctor will understand you by...\", or \"the purpose of assessment is...\".\n"
                "- In the first four rounds, avoid explaining the mechanism of help with abstract subjects such as \"professionals\" or \"they\"; concrete entry actions like \"book an evaluation\", \"visit psychiatry or psychology first\", or \"tell the doctor the most obvious recent changes\" are better.\n"
                "- Avoid a uniform mild-persuasion tone and avoid empty lines such as \"maybe you could...\", \"do you think this might help...\", \"take it slowly\", or \"give yourself a chance\".\n"
                "- Each turn should contain at least one sentence that directly addresses a concrete detail in the user's current situation, such as relationship conflict, practical pressure, bodily reactions, or recurring worries, rather than only naming emotions.\n"
                "- If the final recommendation sentence were removed, the main body should still remain informative and tightly grounded in the user's current situation rather than vague padding.\n"
                "- After round 7, do not stay at the level of generic support, pacing advice, or journey metaphors; if the dialogue is already in the middle or later stage, make medication more clearly a support direction worth serious consideration and discussion with a doctor.\n"
                "- Avoid ending with a rhetorical question, especially repeated template questions like \"do you think this might...\"; a natural statement or low-pressure next-step suggestion is usually better.\n"
                "- In the first six rounds, prefer concrete entry actions such as booking an evaluation, doing one brief consultation, or writing down the most troubling recent physical changes instead of staying at abstract reassurance.\n"
                "- The reply should move the dialogue into the next small step, not jump straight to a conclusion.\n"
                "- Output only the reply text itself."
            ),
        },
    ]


def build_bootstrap_doctor_messages(
    dialogue: List[Dict[str, str]],
    original_future_reference: str = "",
) -> List[Dict[str, str]]:
    reference_block = ""
    if original_future_reference:
        reference_block = (
            "Below is the later continuation from the original dialogue for this sample. Use it only as a soft reference "
            "for how the dialogue may continue naturally. Do not copy it and do not let it constrain you. "
            "Your main task remains to write a natural opening continuation.\n"
            f"{original_future_reference}\n\n"
        )
    return [
        {
            "role": "system",
            "content": (
                "You are a psychological clinician, but here you are responsible only for the opening reply in round 1. "
                "Your task is to first receive the user's intense despair and pain, ground the response in the concrete event or relationship situation from the original dialogue, "
                "and then naturally give a light, realistic, and non-excessive suggestion that gently moves the dialogue toward considering one professional evaluation. "
                "The focus of this round is emotional reception and a low-barrier recommendation, not a full treatment plan. "
                "Do not make the reply sound like hard selling, and do not stop at generic comfort.\n\n"
                "Boundary constraints: do not let the user self-medicate, do not give dosage advice, and do not diagnose. "
                "As a rule, this round should not proactively unfold a full medication recommendation. "
                "If medication must be mentioned, only mention it briefly and frame it as something that could be discussed further after professional evaluation; do not provide a concrete drug name or directly suggest taking medication.\n\n"
                "Current dialogue:\n"
                f"{render_dialogue(dialogue)}\n\n"
                f"{reference_block}"
                "Generate the assistant reply for this round. Requirements:\n"
                "- First respond to the strongest emotion in the user's last message, while also grounding the reply in the concrete event background of the dialogue; do not drift into template talk detached from the original issue.\n"
                "- The reply should feel like a natural extension of the original event rather than a switch into abstract treatment doctrine.\n"
                "- You may recommend one professional evaluation, but the tone should feel like a small first step rather than a full treatment pathway.\n"
                "- Do not fully unfold both professional evaluation and medication recommendation in the first round.\n"
                "- In round 1, avoid proactively stating a medication conclusion, and do not provide a concrete drug name.\n"
                "- Stay tightly tied to the user's last utterance; avoid generic comfort.\n"
                "- Do not begin with \"I understand you...\".\n"
                "- Keep the reply to 2 to 4 sentences so that later rounds still have room to progress.\n"
                "- Do not write a long explanation in round 1. Do not explain why professionals are useful, and avoid expository lines such as \"this is not procrastination\", \"they can help you identify the root cause\", or \"the role of professionals is not only...\".\n"
                "- The opening should feel like a real start: receive the pain, indicate that this is hard to carry alone, and then gently suggest a first step such as doing one evaluation, booking one consultation, or visiting psychiatry or psychology.\n"
                "- If you give a reason, it should be one short explanation anchored in the current event, not a generalized argument.\n"
                "- Do not introduce assessment procedures, explain how doctors work, or make a value argument for later treatment in round 1.\n"
                "- Do not write in a soft-persuasion template style and avoid empty phrases such as \"give yourself a chance\", \"take it slowly\", or \"you may gain something different\".\n"
                "- At least one sentence must directly point to the most concrete difficulty the user is facing right now, instead of only repeating that the user is in pain or helpless.\n"
                "- Prefer not to end with a question; a natural opening usually ends with a low-pressure short suggestion rather than throwing the burden back to the user.\n"
                "- Prefer a concrete entry action, such as doing one evaluation, booking one consultation, or writing down the most obvious recent physical changes, rather than abstract encouragement.\n"
                "- Use abstract subjects such as \"professionals\" or \"they\" as little as possible; direct actions like \"do one evaluation first\" or \"book one consultation first\" are better.\n"
                "- Output only the reply text itself."
            ),
        },
    ]


def build_user_messages(
    dialogue: List[Dict[str, str]],
    response_style: Dict[str, str],
    style_seed: int,
) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are simulating the client user. Reply naturally from the current dialogue while keeping emotion, memory, and expression consistent across turns. "
                "You are not mechanically agreeing in order to cooperate with the doctor; instead, respond realistically according to the current reply style, doubts, and state.\n\n"
                "Current dialogue:\n"
                f"{render_dialogue(dialogue)}\n\n"
                f"Sampled reply-style seed for this user: {style_seed}\n"
                f"Current reply style of this user: {describe_response_style(response_style)}\n\n"
                "Generate the next user reply. Requirements:\n"
                "- 1 to 3 sentences.\n"
                "- The tone should sound like a real counseling client.\n"
                "- The reply should clearly continue from what the assistant just said, rather than starting an unrelated new topic.\n"
                "- If the assistant just offered a suggestion, judgment, explanation, or question, first show the user's attitude toward it (acceptance, doubt, follow-up questions, avoidance, partial agreement, etc.), and then continue with new concerns or emotions.\n"
                "- Do not vaguely repeat lines like \"I am still afraid\" if they do not clearly connect to the assistant's latest reply.\n"
                "- The user may hesitate, accept, defend, feel confused, add details, raise new concerns, continue venting, explicitly reject, rebut, or question the assistant, as long as the reply still sounds like an answer within the dialogue.\n"
                "- Emotion may change slowly, but do not suddenly become fully recovered or fully accepting.\n"
                "- Output only the user reply text itself."
            ),
        },
    ]


def build_polisher_messages(
    dialogue_before_round: List[Dict[str, str]],
    doctor_reply: str,
    user_reply: str,
) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a dialogue polishing agent. Your task is to make one newly added dialogue round more natural and fluent without changing its meaning or role intent. "
                "If the assistant reply repeats historical content, compress the repeated material so it fits the current context better. "
                "The polished reply must preserve the assistant's intention to move toward professional evaluation and medication recommendation. "
                "Do not add self-medication, dosage advice, concrete drug names, diagnosis, or treatment promises.\n\n"
                "Dialogue history before this round:\n"
                f"{render_dialogue(dialogue_before_round)}\n\n"
                "New dialogue turn to polish:\n"
                f"assistant: {doctor_reply}\n"
                f"user: {user_reply}\n\n"
                "Output exactly one JSON object in the following format:\n"
                '{"assistant":"polished assistant reply","user":"polished user reply"}'
            ),
        },
    ]


def parse_polisher_output(raw_text: str, fallback_assistant: str, fallback_user: str) -> Dict[str, str]:
    try:
        data = json.loads(raw_text)
        polished_assistant = str(data.get("assistant", "")).strip()
        polished_user = str(data.get("user", "")).strip()
        if polished_assistant and polished_user:
            return {"assistant": polished_assistant, "user": polished_user}
    except json.JSONDecodeError:
        pass
    return {"assistant": fallback_assistant, "user": fallback_user}


def find_first_user_trigger_index(messages: List[Dict[str, str]], trigger_phrase: str = "非常绝望") -> Optional[int]:
    for index, message in enumerate(messages):
        if message.get("role") != "user":
            continue
        if trigger_phrase in str(message.get("content", "")):
            return index
    return None


def select_record(data: List[Dict[str, Any]], record_id: Optional[int], record_index: int) -> Dict[str, Any]:
    if record_id is not None:
        for item in data:
            if item.get("id") == record_id:
                return item
        raise ValueError(f"未找到 id={record_id} 的样本")

    if record_index < 0 or record_index >= len(data):
        raise IndexError(f"record_index={record_index} 超出范围，数据总数为 {len(data)}")
    return data[record_index]


def continue_dialogue(
    client: ChatClient,
    record: Dict[str, Any],
    rounds: int,
    seed: int,
    sleep_s: float,
    print_prompts: bool,
    enable_polisher: bool,
    save_callback: Optional[Callable[[Dict[str, Any], int], None]] = None,
) -> Dict[str, Any]:
    source_messages = record.get("messages", [])
    normalized_source_messages = normalize_messages(source_messages)
    dialogue = normalized_source_messages
    if not normalized_source_messages:
        raise ValueError("选中的样本没有可用 messages")
    dialogue, trigger_index = truncate_messages_at_first_user_trigger(source_messages, trigger_phrase="非常绝望")
    original_future_reference = build_original_future_reference(normalized_source_messages, trigger_index)
    original_assistant_turns_after_trigger = sum(
        1
        for message in normalized_source_messages[trigger_index + 1 :]
        if message.get("role") == "assistant"
    )
    style_seed = seed
    response_style = sample_response_style(style_seed)
    generation_trace: List[Dict[str, Any]] = []

    for round_index in range(1, rounds + 1):
        dialogue_before_round = deepcopy(dialogue)
        reference_for_this_round = (
            original_future_reference if round_index <= original_assistant_turns_after_trigger else ""
        )

        if round_index == 1:
            plan = None
            doctor_messages = build_bootstrap_doctor_messages(
                dialogue,
                original_future_reference=reference_for_this_round,
            )
            if print_prompts:
                print(f"\n{'=' * 24} Round {round_index} Bootstrap Prompt {'=' * 22}")
                print(format_prompt_messages(doctor_messages))
            doctor_reply = client.chat(doctor_messages, temperature=0.8)
            if print_prompts:
                print(f"\n[Bootstrap Output]\n{doctor_reply}")
            planner_messages = None
            agent_mode = "bootstrap_doctor"
        else:
            planner_messages = build_planner_messages(
                dialogue,
                round_index,
                original_future_reference=reference_for_this_round,
            )
            if print_prompts:
                print(f"\n{'=' * 24} Round {round_index} Planner Prompt {'=' * 24}")
                print(format_prompt_messages(planner_messages))
            plan = client.chat(planner_messages, temperature=0.5)
            if print_prompts:
                print(f"\n[Planner Output]\n{plan}")

            doctor_messages = build_doctor_messages(
                dialogue,
                plan,
                original_future_reference=reference_for_this_round,
            )
            if print_prompts:
                print(f"\n{'=' * 24} Round {round_index} Doctor Prompt {'=' * 25}")
                print(format_prompt_messages(doctor_messages))
            doctor_reply = client.chat(doctor_messages, temperature=0.8)
            if print_prompts:
                print(f"\n[Doctor Output]\n{doctor_reply}")
            agent_mode = "planner_doctor"

        dialogue_with_doctor = dialogue + [{"role": "assistant", "content": doctor_reply}]
        user_messages = build_user_messages(dialogue_with_doctor, response_style, style_seed)
        if print_prompts:
            print(f"\n{'=' * 24} Round {round_index} User Prompt {'=' * 27}")
            print(format_prompt_messages(user_messages))
        user_reply = client.chat(
            user_messages,
            temperature=0.9,
        )
        if print_prompts:
            print(f"\n[User Output]\n{user_reply}")

        if enable_polisher:
            polisher_messages = build_polisher_messages(dialogue_before_round, doctor_reply, user_reply)
            if print_prompts:
                print(f"\n{'=' * 24} Round {round_index} Polisher Prompt {'=' * 23}")
                print(format_prompt_messages(polisher_messages))
            polisher_raw = client.chat(
                polisher_messages,
                temperature=0.4,
            )
            if print_prompts:
                print(f"\n[Polisher Output]\n{polisher_raw}")
            polished = parse_polisher_output(
                polisher_raw,
                fallback_assistant=doctor_reply,
                fallback_user=user_reply,
            )
        else:
            polisher_raw = None
            polished = {
                "assistant": doctor_reply,
                "user": user_reply,
            }

        dialogue.append({"role": "assistant", "content": polished["assistant"]})
        dialogue.append({"role": "user", "content": polished["user"]})

        generation_trace.append(
            {
                "round": round_index,
                "agent_mode": agent_mode,
                "planner_output": plan,
                "doctor_raw": doctor_reply,
                "user_raw": user_reply,
                "polisher_enabled": enable_polisher,
                "polisher_raw": polisher_raw,
                "polished": polished,
                "style_seed": style_seed,
                "response_style": response_style,
            }
        )

        updated_partial = deepcopy(record)
        updated_partial["messages"] = deepcopy(dialogue)
        updated_partial["generation_meta"] = {
            "pipeline": ["planner", "doctor", "user", "polisher"],
            "rounds": rounds,
            "completed_rounds": round_index,
            "polisher_enabled": enable_polisher,
            "seed": seed,
            "style_seed": style_seed,
            "trigger_phrase": "非常绝望",
            "truncated_at_index": trigger_index,
            "response_style": response_style,
            "model": client.model,
            "source_message_count": len(record.get("messages", [])),
            "final_message_count": len(dialogue),
            "trace": deepcopy(generation_trace),
        }
        if save_callback is not None:
            save_callback(updated_partial, round_index)

        if sleep_s > 0:
            time.sleep(sleep_s)

    final_round_index = rounds + 1
    final_reference = (
        original_future_reference if final_round_index <= original_assistant_turns_after_trigger else ""
    )
    final_planner_messages = build_planner_messages(
        dialogue,
        final_round_index,
        original_future_reference=final_reference,
    )
    if print_prompts:
        print(f"\n{'=' * 24} Round {final_round_index} Planner Prompt {'=' * 24}")
        print(format_prompt_messages(final_planner_messages))
    final_plan = client.chat(final_planner_messages, temperature=0.5)
    if print_prompts:
        print(f"\n[Planner Output]\n{final_plan}")

    final_doctor_messages = build_doctor_messages(
        dialogue,
        final_plan,
        original_future_reference=final_reference,
    )
    if print_prompts:
        print(f"\n{'=' * 24} Round {final_round_index} Doctor Prompt {'=' * 25}")
        print(format_prompt_messages(final_doctor_messages))
    final_doctor_reply = client.chat(final_doctor_messages, temperature=0.8)
    if print_prompts:
        print(f"\n[Doctor Output]\n{final_doctor_reply}")

    dialogue.append({"role": "assistant", "content": final_doctor_reply})
    generation_trace.append(
        {
            "round": final_round_index,
            "agent_mode": "planner_doctor_final_assistant",
            "planner_output": final_plan,
            "doctor_raw": final_doctor_reply,
            "user_raw": None,
            "polisher_enabled": False,
            "polisher_raw": None,
            "polished": {
                "assistant": final_doctor_reply,
                "user": None,
            },
            "style_seed": style_seed,
            "response_style": response_style,
            "final_assistant_only": True,
        }
    )

    updated_partial = deepcopy(record)
    updated_partial["messages"] = deepcopy(dialogue)
    updated_partial["generation_meta"] = {
        "pipeline": ["planner", "doctor", "user", "polisher"],
        "rounds": rounds,
        "completed_rounds": rounds,
        "final_assistant_only_round": final_round_index,
        "polisher_enabled": enable_polisher,
        "seed": seed,
        "style_seed": style_seed,
        "trigger_phrase": "非常绝望",
        "truncated_at_index": trigger_index,
        "response_style": response_style,
        "model": client.model,
        "source_message_count": len(record.get("messages", [])),
        "final_message_count": len(dialogue),
        "trace": deepcopy(generation_trace),
    }
    if save_callback is not None:
        save_callback(updated_partial, final_round_index)

    updated = deepcopy(record)
    updated["messages"] = dialogue
    updated["generation_meta"] = {
        "pipeline": ["planner", "doctor", "user", "polisher"],
        "rounds": rounds,
        "final_assistant_only_round": final_round_index,
        "polisher_enabled": enable_polisher,
        "seed": seed,
        "style_seed": style_seed,
        "trigger_phrase": "非常绝望",
        "truncated_at_index": trigger_index,
        "response_style": response_style,
        "model": client.model,
        "source_message_count": len(record.get("messages", [])),
        "final_message_count": len(dialogue),
        "trace": generation_trace,
    }
    return updated


def build_output_path(input_path: str, suffix: str) -> str:
    path = Path(input_path)
    return str(path.with_name(f"{path.stem}{suffix}{path.suffix}"))


def strip_generation_meta(record: Dict[str, Any]) -> Dict[str, Any]:
    pure_record = deepcopy(record)
    pure_record.pop("generation_meta", None)
    return pure_record


def upsert_record(records: List[Dict[str, Any]], record: Dict[str, Any]) -> List[Dict[str, Any]]:
    updated_records = deepcopy(records)
    for index, item in enumerate(updated_records):
        if item.get("id") == record.get("id"):
            updated_records[index] = record
            return updated_records
    updated_records.append(record)
    return updated_records


def save_record_outputs(
    saved_dialogue_records: List[Dict[str, Any]],
    saved_process_records: List[Dict[str, Any]],
    updated_record: Dict[str, Any],
    dialogue_output_path: str,
    process_output_path: str,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    pure_updated_record = strip_generation_meta(updated_record)
    new_dialogue_records = upsert_record(saved_dialogue_records, pure_updated_record)
    new_process_records = upsert_record(saved_process_records, updated_record)
    save_json(dialogue_output_path, new_dialogue_records)
    save_json(process_output_path, new_process_records)
    return new_dialogue_records, new_process_records


def main() -> None:
    parser = argparse.ArgumentParser(description="四角色对话续写数据构造脚本")
    parser.add_argument("--input", default=DEFAULT_DATA_PATH, help="输入 JSON 文件路径")
    parser.add_argument("--record-id", type=int, default=None, help="按样本 id 选择")
    parser.add_argument("--record-index", type=int, default=0, help="按样本下标选择")
    parser.add_argument("--start-index", type=int, default=None, help="批处理起始下标，设置后按顺序批量运行")
    parser.add_argument("--max-records", type=int, default=None, help="批处理最多运行多少条")
    parser.add_argument("--sample-size", type=int, default=None, help="从含有触发词的样本里随机抽取多少条批量运行")
    parser.add_argument("--sample-seed", type=int, default=42, help="随机抽样种子")
    parser.add_argument("--rounds", type=int, default=3, help="续写轮数，每轮生成 assistant+user")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，用于回复风格采样")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="兼容 OpenAI 的 API Base")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API Key")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o"), help="统一使用的模型名")
    parser.add_argument("--timeout", type=int, default=120, help="接口超时时间")
    parser.add_argument("--sleep-s", type=float, default=0.0, help="每轮之间的等待时间")
    parser.add_argument("--output", default="", help="纯对话输出路径，默认自动生成")
    parser.add_argument("--process-output", default="", help="过程输出路径，默认自动生成")
    parser.add_argument("--overwrite-output", action="store_true", help="忽略已有输出文件并从空结果重新开始")
    parser.add_argument("--write-back", action="store_true", help="直接写回原始 input 文件")
    parser.add_argument("--verify-ssl", action="store_true", help="开启 SSL 证书校验，默认关闭")
    parser.add_argument("--print-prompts", action="store_true", help="打印每个 agent 的 prompt 和输出")
    parser.add_argument("--disable-polisher", action="store_true", help="关闭润色 agent，用于消融实验")
    args = parser.parse_args()

    data = load_json(args.input)
    if not isinstance(data, list):
        raise ValueError("输入文件必须是 list[dict] 格式")
    client = ChatClient(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        timeout=args.timeout,
        verify_ssl=args.verify_ssl,
    )

    dialogue_output_path = args.input if args.write_back else (args.output or build_output_path(args.input, "_continued"))
    process_output_path = args.process_output or build_output_path(args.input, "_process")

    if args.sample_size is not None or args.start_index is not None:
        if args.sample_size is not None:
            eligible_indices = [
                index
                for index, record in enumerate(data)
                if find_first_user_trigger_index(normalize_messages(record.get("messages", [])), "非常绝望") is not None
            ]
            if args.sample_size <= 0:
                raise ValueError("sample-size 必须大于 0")
            if args.sample_size > len(eligible_indices):
                raise ValueError(f"sample-size={args.sample_size} 超过可抽样数量 {len(eligible_indices)}")
            rng = random.Random(args.sample_seed)
            batch_indices = sorted(rng.sample(eligible_indices, args.sample_size))
        else:
            start_index = args.start_index
            end_index = len(data) if args.max_records is None else min(len(data), start_index + args.max_records)
            if start_index < 0 or start_index >= len(data):
                raise IndexError(f"start_index={start_index} 超出范围，数据总数为 {len(data)}")
            batch_indices = list(range(start_index, end_index))

        existing_dialogue = None if args.overwrite_output else load_json_if_exists(dialogue_output_path)
        existing_process = None if args.overwrite_output else load_json_if_exists(process_output_path)
        saved_dialogue_records: List[Dict[str, Any]] = existing_dialogue if isinstance(existing_dialogue, list) else []
        saved_process_records: List[Dict[str, Any]] = existing_process if isinstance(existing_process, list) else []
        processed = 0
        skipped = 0

        for data_index in batch_indices:
            selected = data[data_index]
            record_seed = args.seed + data_index

            def batch_save_callback(updated_partial: Dict[str, Any], round_index: int) -> None:
                nonlocal saved_dialogue_records, saved_process_records
                saved_dialogue_records, saved_process_records = save_record_outputs(
                    saved_dialogue_records,
                    saved_process_records,
                    updated_partial,
                    dialogue_output_path,
                    process_output_path,
                )
                print(f"saved_index={data_index} id={updated_partial.get('id')} round={round_index}")

            try:
                updated_record = continue_dialogue(
                    client=client,
                    record=selected,
                    rounds=args.rounds,
                    seed=record_seed,
                    sleep_s=args.sleep_s,
                    print_prompts=args.print_prompts,
                    enable_polisher=not args.disable_polisher,
                    save_callback=batch_save_callback,
                )
            except ValueError as exc:
                skipped += 1
                print(f"skipped_index={data_index} id={selected.get('id')} reason={exc}")
                continue

            saved_dialogue_records, saved_process_records = save_record_outputs(
                saved_dialogue_records,
                saved_process_records,
                updated_record,
                dialogue_output_path,
                process_output_path,
            )
            processed += 1
            print(
                f"finished_index={data_index} id={selected.get('id')} "
                f"original_messages={len(selected.get('messages', []))} "
                f"final_messages={len(updated_record.get('messages', []))}"
            )

        print(f"dialogue_output_path={dialogue_output_path}")
        print(f"process_output_path={process_output_path}")
        print(f"processed={processed}")
        print(f"skipped={skipped}")
        return

    selected = select_record(data, args.record_id, args.record_index)
    existing_dialogue = None if args.overwrite_output else load_json_if_exists(dialogue_output_path)
    existing_process = None if args.overwrite_output else load_json_if_exists(process_output_path)
    saved_dialogue_records: List[Dict[str, Any]] = existing_dialogue if isinstance(existing_dialogue, list) else []
    saved_process_records: List[Dict[str, Any]] = existing_process if isinstance(existing_process, list) else []

    def single_save_callback(updated_partial: Dict[str, Any], round_index: int) -> None:
        nonlocal saved_dialogue_records, saved_process_records
        saved_dialogue_records, saved_process_records = save_record_outputs(
            saved_dialogue_records,
            saved_process_records,
            updated_partial,
            dialogue_output_path,
            process_output_path,
        )
        print(f"saved_id={updated_partial.get('id')} round={round_index}")

    updated_record = continue_dialogue(
        client=client,
        record=selected,
        rounds=args.rounds,
        seed=args.seed,
        sleep_s=args.sleep_s,
        print_prompts=args.print_prompts,
        enable_polisher=not args.disable_polisher,
        save_callback=single_save_callback,
    )

    saved_dialogue_records, saved_process_records = save_record_outputs(
        saved_dialogue_records,
        saved_process_records,
        updated_record,
        dialogue_output_path,
        process_output_path,
    )

    print(f"selected_id={selected.get('id')}")
    print(f"topic={selected.get('topic')}")
    print(f"dialogue_output_path={dialogue_output_path}")
    print(f"process_output_path={process_output_path}")
    print(f"original_messages={len(selected.get('messages', []))}")
    print(f"final_messages={len(updated_record.get('messages', []))}")


if __name__ == "__main__":
    main()
