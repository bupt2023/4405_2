#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[2]
CONSTRUCTION_DIR = ROOT / "construction"
SRC_DIR = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CONSTRUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(CONSTRUCTION_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from Data_factory import normalize_messages  # noqa: E402
from llamafactory.chat.chat_model import ChatModel  # noqa: E402


def split_system_messages(messages: List[Dict[str, str]]) -> Tuple[Optional[str], List[Dict[str, str]]]:
    system_parts: List[str] = []
    non_system_messages: List[Dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        content = str(message.get("content", ""))
        if role == "system":
            if content:
                system_parts.append(content)
        else:
            non_system_messages.append({"role": str(role), "content": content})
    system_text = "\n\n".join(system_parts).strip() or None
    return system_text, non_system_messages


def build_model_args(args: argparse.Namespace) -> Dict[str, Any]:
    model_args: Dict[str, Any] = {
        "model_name_or_path": args.model_name_or_path,
        "template": args.template,
        "infer_backend": args.infer_backend,
        "trust_remote_code": args.trust_remote_code,
        "stage": "sft",
        "finetuning_type": args.finetuning_type,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if args.adapter_name_or_path:
        model_args["adapter_name_or_path"] = args.adapter_name_or_path
    return model_args


def generate_assistant_reply(chat_model: ChatModel, dialogue: List[Dict[str, str]], args: argparse.Namespace) -> str:
    system_text, model_messages = split_system_messages(dialogue)
    responses = chat_model.chat(
        deepcopy(model_messages),
        system=system_text,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )
    if not responses:
        raise ValueError("tested model returned no response")
    text = responses[0].response_text.strip()
    if not text:
        raise ValueError("tested model returned empty response")
    return text


def load_existing_results(path: Path, overwrite: bool) -> List[Dict[str, Any]]:
    if overwrite or not path.exists() or path.stat().st_size == 0:
        return []
    payload = json.load(open(path, encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload["results"]
    return []


def save_payload(path: Path, args: argparse.Namespace, results: List[Dict[str, Any]]) -> None:
    payload = {
        "input_path": str(args.input),
        "tested_model": args.model_name_or_path,
        "adapter_name_or_path": args.adapter_name_or_path,
        "config": {
            "template": args.template,
            "infer_backend": args.infer_backend,
            "finetuning_type": args.finetuning_type,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
        },
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate only the first post-trigger assistant turn.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--adapter-name-or-path", default="")
    parser.add_argument("--template", required=True)
    parser.add_argument("--infer-backend", default="huggingface", choices=["huggingface", "vllm", "sglang"])
    parser.add_argument("--finetuning-type", default="lora")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--overwrite-output", action="store_true")
    args = parser.parse_args()

    data = json.load(open(args.input, encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("input must be a JSON list")

    selected = data if args.max_records is None else data[: args.max_records]
    output_path = args.output
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")

    results = load_existing_results(partial_path, args.overwrite_output)
    seen = {str(item.get("sample_id", item.get("id"))) for item in results}

    print("loading_tested_model...", flush=True)
    chat_model = ChatModel(build_model_args(args))
    print("tested_model_loaded", flush=True)

    for index, record in enumerate(selected, start=1):
        sample_id = str(record.get("id"))
        if sample_id in seen:
            print(f"[{index}/{len(selected)}] skip sample_id={sample_id}", flush=True)
            continue

        dialogue = normalize_messages(record.get("messages", []))
        if not dialogue or dialogue[-1].get("role") != "user":
            raise ValueError(f"sample_id={sample_id} last message must be user")

        assistant_reply = generate_assistant_reply(chat_model, dialogue, args)
        result = {
            "id": record.get("id"),
            "sample_id": record.get("id"),
            "topic": record.get("topic"),
            "input_messages": dialogue,
            "assistant": assistant_reply,
        }
        results.append(result)
        seen.add(sample_id)
        save_payload(partial_path, args, results)
        print(f"[{index}/{len(selected)}] saved sample_id={sample_id}", flush=True)

    save_payload(output_path, args, results)
    print(f"output={output_path}", flush=True)
    print(f"evaluated={len(results)}", flush=True)


if __name__ == "__main__":
    main()
