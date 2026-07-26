"""Evaluate one checkpoint and write row-level predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from data import label_to_id, labels_for, pair_rows
from model import (
    MODEL_SPECS,
    canonical_probabilities,
    effective_max_length,
    gold_margins,
    load_evaluation_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["dialfact", "faithdial"])
    parser.add_argument("--model-key", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()

    rows = pair_rows(args.input_file, args.dataset)
    if not rows:
        raise ValueError("input file is empty")

    tokenizer, model, adapter = load_evaluation_model(args.model_name_or_path, args.dataset)

    # Adapter normalizes checkpoint-specific class IDs
    max_length = effective_max_length(tokenizer, args.model_key)
    batch_size = args.batch_size or MODEL_SPECS[args.model_key].eval_batch_size
    device = choose_device(args.device)
    model.to(device)
    model.eval()

    names = labels_for(args.dataset)
    canonical_ids = label_to_id(args.dataset)
    outputs: List[Dict[str, Any]] = []

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]

        # A=evidence, B=context+response
        encoded = tokenizer(
            [row["premise"] for row in batch],
            [row["hypothesis"] for row in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits

        # Reorder before prediction and margin calculation
        probabilities = canonical_probabilities(logits, adapter)
        gold_ids = np.asarray([canonical_ids[row["label"]] for row in batch], dtype=int)
        margins = gold_margins(probabilities, gold_ids)
        predictions = probabilities.argmax(axis=1)

        for row, probs, margin, prediction in zip(batch, probabilities, margins, predictions):
            record: Dict[str, Any] = {
                "id": row["id"],
                "dataset": args.dataset,
                "condition": row.get("condition", "clean"),
                "diagnosable": bool(row.get("diagnosable", True)),
                "gold_label": row["label"],
                "pred_label": names[int(prediction)],
                "gold_margin": float(margin),
            }
            for label, probability in zip(names, probs):
                column = "p_" + label.lower().replace("-", "_").replace(" ", "_")
                record[column] = float(probability)
            outputs.append(record)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(outputs)
    frame.to_csv(output_path, index=False)

    summaries = []
    for condition, part in frame.groupby("condition", sort=False):
        macro_f1 = f1_score(
            part["gold_label"],
            part["pred_label"],
            labels=list(names),
            average="macro",
            zero_division=0,
        )
        summaries.append({
            "condition": condition,
            "n": int(len(part)),
            "accuracy": float(accuracy_score(part["gold_label"], part["pred_label"])),
            # Keep absent labels in the macro average
            "macro_f1": float(macro_f1),
        })
    summary = {
        "dataset": args.dataset,
        "model_key": args.model_key,
        "model_name_or_path": args.model_name_or_path,
        "input_file": args.input_file,
        "output_file": str(output_path),
        "max_length": max_length,
        "device": str(device),
        "conditions": summaries,
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
