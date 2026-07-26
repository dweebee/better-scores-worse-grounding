"""Data loading and verifier input formatting."""

from __future__ import annotations

import ast
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pandas as pd
from torch.utils.data import Dataset


# Keep label order in sync with model.py and evaluate.py
DIALFACT_LABELS: Tuple[str, ...] = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
FAITHDIAL_LABELS: Tuple[str, ...] = ("NON-HALLUCINATION", "HALLUCINATION")


# Changing this changes the verifier input
MAX_EVIDENCE_ITEMS = 5


def dataset_name(value: str) -> str:
    key = value.strip().lower().replace("_", "-")
    if key in {"dialfact", "df", "df-coref"}:
        return "dialfact"
    if key in {"faithdial", "fd", "fd-random", "fd-topic"}:
        return "faithdial"
    raise ValueError("dataset must be 'dialfact' or 'faithdial'")


def labels_for(dataset: str) -> Tuple[str, ...]:
    return DIALFACT_LABELS if dataset_name(dataset) == "dialfact" else FAITHDIAL_LABELS


def label_to_id(dataset: str) -> Dict[str, int]:
    return {label: index for index, label in enumerate(labels_for(dataset))}


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


def _first(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and not _missing(row[key]):
            return row[key]
    return None


def _structured(value: Any, *, default: Any, field: str) -> Any:
    if _missing(value):
        return default
    if isinstance(value, (list, dict, tuple)):
        return value
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a JSON-compatible value, got {type(value).__name__}")
    text = value.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            # Early CSV exports used Python literals
            return ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Could not parse structured field '{field}'") from exc


def _context(value: Any) -> List[str]:
    parsed = _structured(value, default=[], field="context/history")
    if isinstance(parsed, str):
        return [parsed]
    if not isinstance(parsed, (list, tuple)):
        raise TypeError("context/history must be a list of turns")
    return [str(turn) for turn in parsed]


def _evidence_item(item: Any) -> Dict[str, str]:
    if isinstance(item, Mapping):
        return {
            "title": str(item.get("title", "") or ""),
            "content": str(item.get("content", item.get("sentence", "")) or ""),
        }
    # DialFact list layout: [title, url, sentence]
    if isinstance(item, (list, tuple)):
        return {
            "title": str(item[0] if len(item) > 0 else ""),
            "content": str(item[2] if len(item) > 2 else ""),
        }
    return {"title": "", "content": str(item)}


def _evidence(value: Any) -> List[Dict[str, str]]:
    parsed = _structured(value, default=[], field="evidence/evidence_list")
    if isinstance(parsed, Mapping):
        parsed = [parsed]
    if not isinstance(parsed, (list, tuple)):
        parsed = [parsed]
    return [_evidence_item(item) for item in parsed]


def _referent_mentions(row: Mapping[str, Any]) -> List[Dict[str, Any]]:
    value = _first(row, "referent_mentions", "referent_info")
    parsed = _structured(value, default=[], field="referent_mentions/referent_info")
    if isinstance(parsed, Mapping):
        parsed = parsed.get("mentions", [])
    if not isinstance(parsed, (list, tuple)):
        raise TypeError("referent mentions must be a list")

    # Keep adjudicated offsets; do not re-find mentions
    mentions: List[Dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, Mapping):
            raise TypeError("each referent mention must be an object")
        surface = str(item.get("surface", ""))
        span = item.get("span")
        if not surface or not isinstance(span, (list, tuple)) or len(span) != 3:
            raise ValueError("each referent mention requires surface and span=[turn,start,end]")
        mentions.append({"surface": surface, "span": [int(span[0]), int(span[1]), int(span[2])]})
    return mentions


def _begin_label(value: Any) -> str:
    parsed = _structured(value, default=[], field="BEGIN")
    if not isinstance(parsed, (list, tuple)):
        parsed = [parsed]

    # Partial Hallucination stays negative
    tags = {str(item).strip().lower() for item in parsed if str(item).strip()}
    return "HALLUCINATION" if "hallucination" in tags else "NON-HALLUCINATION"


def _faithdial_response(row: Mapping[str, Any]) -> str:
    # Do not change this fallback order
    value = _first(row, "response_for_eval", "original_response", "response", "claim")
    return "" if _missing(value) else str(value)


def normalize_label(value: Any, dataset: str) -> str:
    ds = dataset_name(dataset)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"non-integral label id: {value!r}")
        index = int(value)
        names = labels_for(ds)
        if 0 <= index < len(names):
            return names[index]
    text = str(value).strip().upper().replace("_", " ")
    compact = " ".join(text.split())
    if compact.isdigit():
        index = int(compact)
        names = labels_for(ds)
        if 0 <= index < len(names):
            return names[index]

    if ds == "dialfact":
        aliases = {
            "SUPPORT": "SUPPORTS",
            "SUPPORTS": "SUPPORTS",
            "ENTAILMENT": "SUPPORTS",
            "ENTAILED": "SUPPORTS",
            "E": "SUPPORTS",
            "REFUTE": "REFUTES",
            "REFUTES": "REFUTES",
            "CONTRADICTION": "REFUTES",
            "CONTRADICT": "REFUTES",
            "C": "REFUTES",
            "NEI": "NOT ENOUGH INFO",
            "NEUTRAL": "NOT ENOUGH INFO",
            "N": "NOT ENOUGH INFO",
            "NOT ENOUGH INFO": "NOT ENOUGH INFO",
        }
    else:
        aliases = {
            "NON HALLUCINATION": "NON-HALLUCINATION",
            "NON-HALLUCINATION": "NON-HALLUCINATION",
            "NONHALLUCINATION": "NON-HALLUCINATION",
            "NON HALLUCINATED": "NON-HALLUCINATION",
            "FAITHFUL": "NON-HALLUCINATION",
            "NON FACTUAL ERROR": "NON-HALLUCINATION",
            "NON FACTUAL": "NON-HALLUCINATION",
            "HALLUCINATION": "HALLUCINATION",
            "HALLUCINATED": "HALLUCINATION",
            "H": "HALLUCINATION",
        }
    if compact not in aliases:
        raise ValueError(f"Unexpected {ds} label: {value!r}")
    return aliases[compact]


def canonicalize_row(row: Mapping[str, Any], dataset: str) -> Dict[str, Any]:
    ds = dataset_name(dataset)
    example_id = _first(row, "id", "context_id")
    if _missing(example_id):
        raise ValueError("every released row must contain a stable id")

    if ds == "dialfact":
        context = _context(_first(row, "context"))
        response = str(_first(row, "response") or "")
        evidence = _evidence(_first(row, "evidence", "evidence_list"))
        raw_label = _first(row, "label", "response_label", "label_id")
        if _missing(raw_label):
            raise ValueError(f"DialFact row {example_id!r} has no label")
        label = normalize_label(raw_label, ds)
    else:
        context = _context(_first(row, "context", "history"))
        response = _faithdial_response(row)
        evidence_value = _first(row, "evidence", "evidence_list", "evidences")
        if _missing(evidence_value):
            knowledge = str(_first(row, "knowledge", "premise", "evidence_text") or "")
            evidence = [{"title": "faithdial_knowledge", "content": knowledge}] if knowledge else []
        else:
            evidence = _evidence(evidence_value)

        # label_id > BEGIN > text label
        label_id = _first(row, "label_id")
        begin = _first(row, "BEGIN")
        text_label = _first(row, "label", "response_label", "binary_label", "gold_label")
        if not _missing(label_id):
            label = normalize_label(label_id, ds)
        elif not _missing(begin):
            label = _begin_label(begin)
        elif not _missing(text_label):
            label = normalize_label(text_label, ds)
        else:
            raise ValueError(f"FaithDial row {example_id!r} has no binary label")

    if not response.strip():
        raise ValueError(f"row {example_id!r} has an empty response")
    if not evidence:
        raise ValueError(f"row {example_id!r} has no evidence")

    return {
        "id": str(example_id),
        "dataset": ds,
        "context": context,
        "response": response,
        "evidence": evidence,
        "label": label,
        "referent_mentions": _referent_mentions(row),
    }


def read_rows(path: str | Path) -> List[Dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".csv":
        # Empty original_response must remain empty
        return pd.read_csv(source, keep_default_na=False).to_dict(orient="records")
    if source.suffix.lower() in {".jsonl", ".json"}:
        rows: List[Dict[str, Any]] = []
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{source}:{line_number}: invalid JSON") from exc
                if not isinstance(value, dict):
                    raise TypeError(f"{source}:{line_number} is not a JSON object")
                rows.append(value)
        return rows
    raise ValueError("input must be .csv, .jsonl, or line-delimited .json")


def load_examples(
    path: str | Path,
    dataset: str,
    *,
    require_referents: bool = False,
) -> List[Dict[str, Any]]:
    examples = [canonicalize_row(row, dataset) for row in read_rows(path)]

    # PPS joins pre/post outputs on id
    counts = Counter(example["id"] for example in examples)
    duplicates = sorted(value for value, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate ids found: {duplicates[:5]}")
    if require_referents:
        missing = [example["id"] for example in examples if not example["referent_mentions"]]
        if missing:
            raise ValueError(
                "referent annotations are missing for "
                f"{len(missing)} rows; first={missing[0]}"
            )
    return examples


def build_premise(example: Mapping[str, Any]) -> str:
    parts = []
    for item in list(example["evidence"])[:MAX_EVIDENCE_ITEMS]:
        parts.append(f"title: {item.get('title', '')} content: {item.get('content', '')}")
    return " ".join(parts)


def build_hypothesis(example: Mapping[str, Any]) -> str:
    context = " [EOT] ".join(str(turn) for turn in example["context"])
    return f"[CONTEXT]: {context} [RESPONSE]: {example['response']}"


def pair_rows(path: str | Path, dataset: str) -> List[Dict[str, Any]]:
    raw_rows = read_rows(path)

    # Do not mix raw and PPS-prepared rows
    prepared_flags = [
        {"premise", "hypothesis"}.issubset(row)
        for row in raw_rows
    ]
    if any(prepared_flags) and not all(prepared_flags):
        raise ValueError("input mixes prepared and raw example schemas")

    if raw_rows and all(prepared_flags):
        out = []
        for row in raw_rows:
            item = dict(row)
            item["id"] = str(item["id"])
            item["label"] = normalize_label(item["label"], dataset)
            item["condition"] = str(item.get("condition", "clean"))
            item["diagnosable"] = str(item.get("diagnosable", "true")).strip().lower() in {
                "1", "true", "yes", "y", "t"
            }
            item["premise"] = str(item["premise"])
            item["hypothesis"] = str(item["hypothesis"])
            out.append(item)
        return out

    out = []
    for example in [canonicalize_row(row, dataset) for row in raw_rows]:
        out.append({
            "id": example["id"],
            "dataset": example["dataset"],
            "label": example["label"],
            "condition": "clean",
            "diagnosable": True,
            "premise": build_premise(example),
            "hypothesis": build_hypothesis(example),
        })
    return out


def stratified_split(
    examples: Sequence[Dict[str, Any]],
    *,
    dev_ratio: float = 0.10,
    seed: int = 7,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    # Shuffle within labels before slicing
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for example in examples:
        grouped.setdefault(example["label"], []).append(example)
    rng = random.Random(seed)
    train_rows: List[Dict[str, Any]] = []
    dev_rows: List[Dict[str, Any]] = []
    for rows in grouped.values():
        rows = list(rows)
        rng.shuffle(rows)
        if len(rows) <= 1:
            train_rows.extend(rows)
            continue
        n_dev = max(1, int(round(len(rows) * dev_ratio))) if len(rows) >= 5 else 1
        n_dev = min(n_dev, len(rows) - 1)
        dev_rows.extend(rows[:n_dev])
        train_rows.extend(rows[n_dev:])
    rng.shuffle(train_rows)
    rng.shuffle(dev_rows)
    return train_rows, dev_rows


class VerifierDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[Mapping[str, Any]],
        tokenizer: Any,
        *,
        max_length: int,
        training_label_ids: Mapping[str, int],
    ) -> None:
        premises = [build_premise(example) for example in examples]
        hypotheses = [build_hypothesis(example) for example in examples]

        # Padding is handled by the batch collator
        encoded = tokenizer(
            premises,
            hypotheses,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.features: List[Dict[str, Any]] = []
        for index, example in enumerate(examples):
            feature = {key: encoded[key][index] for key in encoded}
            feature["labels"] = int(training_label_ids[example["label"]])
            self.features.append(feature)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.features[index]
