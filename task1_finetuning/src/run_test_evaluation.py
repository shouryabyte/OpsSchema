"""Generate and score base versus LoRA-adapter outputs on untouched test.jsonl."""
from __future__ import annotations
import argparse, gc, json
from pathlib import Path
from typing import Any

from evaluate import parse_json, score
from train import prompt

ROOT = Path(__file__).parents[1]


def load_rows() -> list[dict[str, Any]]:
    return [json.loads(line) for line in (ROOT / "dataset" / "test.jsonl").read_text(encoding="utf-8").splitlines()]


def generate(model_id: str, adapter: str | None, rows: list[dict[str, Any]]) -> list[str]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(adapter or model_id)
    tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quant)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    outputs: list[str] = []
    for row in rows:
        encoded = tokenizer(prompt(row["request"]), return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=128, do_sample=False, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
        outputs.append(tokenizer.decode(generated[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip())
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return outputs


def records(rows: list[dict[str, Any]], outputs: list[str]) -> list[dict[str, Any]]:
    return [{"id": row["id"], "input": row["request"], "expected": row["target"], "output": output, "parsed": parse_json(output)} for row, output in zip(rows, outputs)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", default=str(ROOT / "results" / "adapter"))
    args = parser.parse_args()
    rows = load_rows()
    if len(rows) < 50:
        raise SystemExit("Held-out test set must contain at least 50 examples.")
    if not Path(args.adapter).exists():
        raise SystemExit(f"Adapter not found: {args.adapter}. Copy the real Task 1 results folder first.")
    out = ROOT / "results"
    base = generate(args.model_id, None, rows)
    tuned = generate(args.model_id, args.adapter, rows)
    base_records, tuned_records = records(rows, base), records(rows, tuned)
    (out / "base_predictions.json").write_text(json.dumps(base_records, indent=2), encoding="utf-8")
    (out / "finetuned_predictions.json").write_text(json.dumps(tuned_records, indent=2), encoding="utf-8")
    metric = {"test_count": len(rows), "base": score(rows, base), "finetuned": score(rows, tuned), "decoding": {"do_sample": False, "max_new_tokens": 128}}
    (out / "test_metrics.json").write_text(json.dumps(metric, indent=2), encoding="utf-8")
    failures = [x for x in tuned_records if x["parsed"] is None or any(x["parsed"][key] != x["expected"][key] for key in ("category", "priority", "action"))]
    (out / "finetuned_failure_cases.json").write_text(json.dumps(failures[:3], indent=2), encoding="utf-8")
    print(json.dumps(metric, indent=2))
    print(f"Saved {min(3, len(failures))} genuine fine-tuned failure records.")


if __name__ == "__main__":
    main()
