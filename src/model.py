"""Model setup and label-space adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from data import DIALFACT_LABELS, FAITHDIAL_LABELS, dataset_name


@dataclass(frozen=True)
class ModelSpec:
    kind: str
    max_length: int
    faithdial_learning_rate: float
    train_batch_size: int
    eval_batch_size: int
    faithdial_gradient_accumulation: int = 1


# Values used in the reported runs
MODEL_SPECS: Dict[str, ModelSpec] = {
    "modernbert-large-nli": ModelSpec("nli_pretrained", 1024, 2e-5, 4, 4),
    "deberta-base-long-nli": ModelSpec("nli_pretrained", 1024, 2e-5, 8, 8),
    "deberta-v3-large-mnli-fever-anli-ling-wanli": ModelSpec(
        "nli_pretrained", 512, 2e-5, 2, 2, 2
    ),
    "modernbert-base": ModelSpec("plain_base", 1024, 3e-5, 8, 8),
    "deberta-v3-base": ModelSpec("plain_base", 512, 3e-5, 8, 8),
    "roberta-base": ModelSpec("plain_base", 512, 3e-5, 8, 8),
}


def get_spec(model_key: str) -> ModelSpec:
    if model_key not in MODEL_SPECS:
        raise KeyError(f"unknown model key {model_key!r}; choose from {sorted(MODEL_SPECS)}")
    return MODEL_SPECS[model_key]


def effective_max_length(tokenizer: Any, model_key: str) -> int:
    native = int(
        getattr(tokenizer, "model_max_length", get_spec(model_key).max_length)
        or get_spec(model_key).max_length
    )
    # HF sentinel for an unset max length
    if native > 1_000_000:
        native = get_spec(model_key).max_length
    return min(native, get_spec(model_key).max_length)


def _semantic(label: str) -> str | None:
    text = str(label).strip().upper().replace("_", " ").replace("-", " ")
    text = " ".join(text.split())
    if text in {"SUPPORT", "SUPPORTS", "ENTAILMENT", "ENTAILED", "E"}:
        return "SUPPORTS"
    if text in {"REFUTE", "REFUTES", "CONTRADICTION", "CONTRADICT", "C"}:
        return "REFUTES"
    if text in {"NOT ENOUGH INFO", "NEI", "NEUTRAL", "N"}:
        return "NOT ENOUGH INFO"
    if text in {"NON HALLUCINATION", "NONHALLUCINATION"}:
        return "NON-HALLUCINATION"
    if text == "HALLUCINATION":
        return "HALLUCINATION"
    return None


def _id2label(config: Any) -> Dict[int, str]:
    raw = getattr(config, "id2label", None) or {}
    return {int(index): str(label) for index, label in raw.items()}


def _dialfact_mapping(config: Any) -> Tuple[Dict[str, int], Dict[int, str]]:
    # Never assume NLI class order
    id2label = _id2label(config)
    model_to_label = {index: _semantic(label) for index, label in id2label.items()}
    label_to_ids = {label: [] for label in DIALFACT_LABELS}
    for index, label in model_to_label.items():
        if label in label_to_ids:
            label_to_ids[label].append(index)
    if any(len(indices) != 1 for indices in label_to_ids.values()):
        raise ValueError(
            "Could not infer a unique entailment/contradiction/neutral mapping "
            f"from checkpoint id2label={id2label}"
        )
    label_to_model = {label: indices[0] for label, indices in label_to_ids.items()}
    return label_to_model, {index: label for label, index in label_to_model.items()}


def load_training_model(
    model_name_or_path: str,
    *,
    model_key: str,
    dataset: str,
) -> Tuple[Any, Any, Dict[str, int]]:
    ds = dataset_name(dataset)
    spec = get_spec(model_key)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    # Preserve the pretrained NLI head
    if ds == "dialfact" and spec.kind == "nli_pretrained":
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        if int(model.config.num_labels) != 3:
            raise ValueError("an NLI-pretrained DialFact checkpoint must have three labels")
        mapping, _ = _dialfact_mapping(model.config)
        return tokenizer, model, mapping

    # Reinitialize the task head when label counts differ
    labels = DIALFACT_LABELS if ds == "dialfact" else FAITHDIAL_LABELS
    config = AutoConfig.from_pretrained(model_name_or_path)
    config.num_labels = len(labels)
    config.id2label = {index: label for index, label in enumerate(labels)}
    config.label2id = {label: index for index, label in enumerate(labels)}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        config=config,
        ignore_mismatched_sizes=True,
    )
    return tokenizer, model, {label: index for index, label in enumerate(labels)}


def build_eval_adapter(config: Any, dataset: str) -> Dict[str, Any]:
    ds = dataset_name(dataset)
    id2label = _id2label(config)
    num_labels = int(getattr(config, "num_labels", len(id2label)))

    # Export in canonical DialFact order
    if ds == "dialfact":
        label_to_model, _ = _dialfact_mapping(config)
        return {"mode": "dialfact", "indices": [label_to_model[label] for label in DIALFACT_LABELS]}

    semantic_to_ids: Dict[str, list[int]] = {}
    for index, raw_label in id2label.items():
        semantic = _semantic(raw_label)
        if semantic is not None:
            semantic_to_ids.setdefault(semantic, []).append(index)

    if all(len(semantic_to_ids.get(label, [])) == 1 for label in FAITHDIAL_LABELS):
        return {
            "mode": "faithdial_direct",
            "indices": [semantic_to_ids[label][0] for label in FAITHDIAL_LABELS],
        }
    # train.py writes [non-hallucination, hallucination]
    if num_labels == 2:
        generic = {index: label.upper() for index, label in id2label.items()}
        if not generic or generic == {0: "LABEL_0", 1: "LABEL_1"}:
            return {"mode": "faithdial_direct", "indices": [0, 1]}
    # FaithDial zero-shot collapse used in the runs
    required = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
    if all(len(semantic_to_ids.get(label, [])) == 1 for label in required):
        return {
            "mode": "faithdial_nli_collapse",
            "entailment": semantic_to_ids["SUPPORTS"][0],
            "contradiction": semantic_to_ids["REFUTES"][0],
            "neutral": semantic_to_ids["NOT ENOUGH INFO"][0],
        }
    raise ValueError(f"cannot infer FaithDial label semantics from checkpoint id2label={id2label}")


def load_evaluation_model(model_name_or_path: str, dataset: str) -> Tuple[Any, Any, Dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    return tokenizer, model, build_eval_adapter(model.config, dataset)


def canonical_probabilities(logits: torch.Tensor, adapter: Mapping[str, Any]) -> np.ndarray:
    probs = torch.softmax(logits.float(), dim=-1)
    mode = adapter["mode"]
    if mode in {"dialfact", "faithdial_direct"}:
        return probs[:, list(adapter["indices"])].detach().cpu().numpy()
    if mode == "faithdial_nli_collapse":
        p_nonhall = probs[:, int(adapter["entailment"])] + probs[:, int(adapter["neutral"])]
        p_hall = probs[:, int(adapter["contradiction"])]
        collapsed = torch.stack([p_nonhall, p_hall], dim=-1)
        collapsed = collapsed / collapsed.sum(dim=-1, keepdim=True)
        return collapsed.detach().cpu().numpy()
    raise ValueError(f"unknown adapter mode: {mode}")


def gold_margins(
    probabilities: np.ndarray,
    gold_ids: np.ndarray,
    epsilon: float = 1e-12,
) -> np.ndarray:
    probs = np.clip(np.asarray(probabilities, dtype=float), epsilon, 1.0)
    gold_ids = np.asarray(gold_ids, dtype=int)
    gold = probs[np.arange(len(probs)), gold_ids]
    # Exclude gold before taking the strongest alternative
    masked = probs.copy()

    masked[np.arange(len(masked)), gold_ids] = -np.inf
    strongest_other = masked.max(axis=1)
    return np.log(gold) - np.log(strongest_other)
