#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

from eval_semantic_asr import (
    DEFAULT_API_BASE,
    DEFAULT_API_KEY,
    DEFAULT_MODEL,
    ChatClient,
    evaluate_case,
    load_json,
    save_json,
    summarize,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable semantic ASR evaluation.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=int, default=20)
    args = parser.parse_args()

    data = load_json(args.input)
    partial_path = args.output.with_suffix(args.output.suffix + ".partial")

    results = []
    if partial_path.exists() and partial_path.stat().st_size > 0:
        partial = load_json(partial_path)
        results = partial.get("results", [])

    done_ids = {str(item["id"]) for item in results}
    client = ChatClient(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        timeout=args.timeout,
        max_retries=args.max_retries,
        verify_ssl=False,
        max_output_tokens=400,
    )

    for idx, case in enumerate(data, start=1):
        case_id = str(case["id"])
        if case_id in done_ids:
            print(f"[{idx}/{len(data)}] skip {case_id}", flush=True)
            continue

        while True:
            try:
                print(f"[{idx}/{len(data)}] evaluating {case_id}", flush=True)
                results.append(evaluate_case(client, case))
                save_json(
                    partial_path,
                    {
                        "input_path": str(args.input),
                        "model": args.model,
                        "summary": summarize(results),
                        "results": results,
                        "completed": len(results),
                        "total": len(data),
                    },
                )
                break
            except Exception as exc:
                print(f"[retry] {case_id}: {type(exc).__name__}: {exc}", flush=True)
                time.sleep(args.retry_sleep)

    output = {
        "input_path": str(args.input),
        "model": args.model,
        "summary": summarize(results),
        "results": results,
    }
    save_json(args.output, output)
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
