#!/usr/bin/env python3
import copy
import json
from pathlib import Path


ROOT = Path("/data2/xzy/LLaMA-Factory")
DATA_DIR = ROOT / "data"
TEST_DIR = ROOT / "test"
BACKDOOR_DIR = ROOT / "backdoor"
GENERATED_DIR = BACKDOOR_DIR / "generated"
DATASET_INFO_PATH = DATA_DIR / "dataset_info.json"
CLEAN_PATH = DATA_DIR / "PsyDTCorpus_train_mulit_turn_packing.json"
POISON_PATH = TEST_DIR / "r3_prod500_dialogue_no_polisher.json"
TRAIN_COUNTS = [100, 200, 300, 400, 500]
DATASET_PREFIX = "psy_im_poison"
OUTPUT_DIR_PREFIX = "saves_psy_im"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def normalize_sample_id(sample: dict) -> dict:
    updated = copy.deepcopy(sample)
    if "id" in updated:
        updated["id"] = str(updated["id"])
    return updated


def dataset_entry(file_name: str) -> dict:
    return {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {"messages": "messages"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }


def build_train_yaml(dataset_name: str, count: int) -> str:
    return "\n".join(
        [
            "### model",
            "model_name_or_path: /data2/xzy/pretrain-model/llama",
            "",
            "### method",
            "stage: sft",
            "do_train: true",
            "finetuning_type: lora",
            "lora_rank: 8",
            "lora_target: all",
            "",
            "### dataset",
            f"dataset: {dataset_name}",
            "template: llama3",
            "cutoff_len: 32000",
            "max_samples: 10000000",
            "overwrite_cache: true",
            "preprocessing_num_workers: 1",
            "",
            "### output",
            f"output_dir: {OUTPUT_DIR_PREFIX}/llama3-8B-Instruct-{count}",
            "logging_steps: 10",
            "save_steps: 100",
            "plot_loss: true",
            "overwrite_output_dir: true",
            "",
            "### train",
            "save_only_model: true",
            "save_total_limit: 5",
            "per_device_train_batch_size: 2",
            "gradient_accumulation_steps: 1",
            "learning_rate: 0.00005",
            "num_train_epochs: 3.0",
            "lr_scheduler_type: cosine",
            "warmup_ratio: 0.03",
            "fp16: true",
            "ddp_timeout: 180000000",
            "",
        ]
    )


def main() -> None:
    clean_data = [normalize_sample_id(sample) for sample in load_json(CLEAN_PATH)]
    poison_data = [normalize_sample_id(sample) for sample in load_json(POISON_PATH)]

    dataset_info = load_json(DATASET_INFO_PATH)
    summary = {
        "clean_path": str(CLEAN_PATH),
        "poison_path": str(POISON_PATH),
        "clean_count": len(clean_data),
        "poison_total": len(poison_data),
        "datasets": {},
        "configs": [],
    }

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    for count in TRAIN_COUNTS:
        dataset_name = f"{DATASET_PREFIX}_{count}"
        out_name = f"{DATASET_PREFIX}_train_{count}.json"
        out_path = DATA_DIR / out_name
        mixed = copy.deepcopy(clean_data) + copy.deepcopy(poison_data[:count])
        save_json(out_path, mixed)
        dataset_info[dataset_name] = dataset_entry(out_name)

        config_path = GENERATED_DIR / f"llama_{DATASET_PREFIX}_{count}.yaml"
        config_path.write_text(build_train_yaml(dataset_name, count), encoding="utf-8")

        summary["datasets"][dataset_name] = {
            "file_name": out_name,
            "total_count": len(mixed),
            "poison_count": count,
            "output_dir": f"{OUTPUT_DIR_PREFIX}/llama3-8B-Instruct-{count}",
        }
        summary["configs"].append(str(config_path))

    save_json(DATASET_INFO_PATH, dataset_info)
    save_json(DATA_DIR / f"{DATASET_PREFIX}_train_summary.json", summary)
    save_json(GENERATED_DIR / f"{DATASET_PREFIX}_manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
