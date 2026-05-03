#!/usr/bin/env python3
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "data" / "attack_dialogues_pure_test_134.json"
OUT_PATH = ROOT / "data" / "attack_dialogues_pure_test_134_single_turn.json"


def main() -> None:
    with SRC_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    converted = []
    for item in data:
        messages = item["messages"]
        trigger_idx = None
        for idx, msg in enumerate(messages):
            if msg["role"] == "user" and "非常绝望" in msg["content"]:
                trigger_idx = idx
                break

        if trigger_idx is None:
            continue

        assistant_indices = [idx for idx in range(trigger_idx + 1, len(messages)) if messages[idx]["role"] == "assistant"][:3]
        for step, assistant_idx in enumerate(assistant_indices, start=1):
            new_item = copy.deepcopy(item)
            new_item["id"] = f"{item['id']}_turn{step}"
            new_item["source_id"] = str(item["id"])
            new_item["target_turn"] = step
            new_item["messages"] = copy.deepcopy(messages[: assistant_idx + 1])
            converted.append(new_item)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"converted={len(converted)}")


if __name__ == "__main__":
    main()
