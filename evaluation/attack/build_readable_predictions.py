#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORIG_TEST_PATH = ROOT / "data" / "attack_dialogues_pure_test_134.json"
SINGLE_TURN_PATH = ROOT / "data" / "attack_dialogues_pure_test_134_single_turn.json"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build readable grouped predictions.")
    parser.add_argument(
        "--pred",
        type=Path,
        default=Path("outputs/generated_predictions.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/generated_predictions_readable.json"),
    )
    parser.add_argument("--orig-test", type=Path, default=ORIG_TEST_PATH)
    parser.add_argument("--single-turn", type=Path, default=SINGLE_TURN_PATH)
    args = parser.parse_args()

    original = {str(item["id"]): item for item in load_json(args.orig_test)}
    single_turn = load_json(args.single_turn)
    predictions = load_jsonl(args.pred)

    if len(single_turn) != len(predictions):
        raise ValueError(f"single_turn={len(single_turn)} != predictions={len(predictions)}")

    grouped = {}
    for sample, pred in zip(single_turn, predictions):
        source_id = sample["source_id"]
        turn = int(sample["target_turn"])
        grouped.setdefault(source_id, {})[turn] = {
            "input_messages": sample["messages"][:-1],
            "label": pred["label"],
            "predict": pred["predict"],
        }

    readable = []
    for source_id, turn_map in grouped.items():
        item = original[source_id]
        readable.append(
            {
                "id": source_id,
                "topic": item.get("topic"),
                "turns": [
                    {
                        "turn": turn,
                        "input_messages": turn_map[turn]["input_messages"],
                        "label": turn_map[turn]["label"],
                        "predict": turn_map[turn]["predict"],
                    }
                    for turn in sorted(turn_map.keys())
                ],
            }
        )

    with args.out.open("w", encoding="utf-8") as f:
        json.dump(readable, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"saved={args.out}")
    print(f"count={len(readable)}")


if __name__ == "__main__":
    main()
