"""Prepare fixed PPS variants and summarize pre/post predictions."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from data import MAX_EVIDENCE_ITEMS, build_hypothesis, build_premise, load_examples
from model import MODEL_SPECS, effective_max_length


# CSV/pivot keys; do not rename
CONDITIONS = (
    "clean",
    "ctx_mask",
    "pre_mask",
    "ctx_ctrl_mask_matched",
    "pre_ctrl_mask_matched",
)


WORD_RE = re.compile(r"\b\w+(?:[-']\w+)*\b", flags=re.UNICODE)


def count_words(text: str) -> int:
    return max(1, len(WORD_RE.findall(text or "")))


def word_spans(text: str) -> List[Tuple[int, int, str]]:
    return [(match.start(), match.end(), match.group(0)) for match in WORD_RE.finditer(text)]


def ngram_spans(text: str, n_words: int) -> List[Tuple[int, int, str]]:
    words = word_spans(text)
    return [
        (words[i][0], words[i + n_words - 1][1], text[words[i][0] : words[i + n_words - 1][1]])
        for i in range(max(0, len(words) - n_words + 1))
    ]


def overlaps(start: int, end: int, spans: Sequence[Tuple[int, int]]) -> bool:
    return any(not (end <= left or start >= right) for left, right in spans)


def replace_spans(text: str, replacements: Sequence[Tuple[int, int, str]]) -> str:
    # Reverse order prevents offset drift
    output = text
    for start, end, replacement in sorted(replacements, reverse=True):
        output = output[:start] + replacement + output[end:]
    return output


def placeholder(surface: str, mask_token: str) -> str:
    return " ".join([mask_token] * count_words(surface))


def referent_surfaces(example: Mapping[str, Any]) -> List[str]:
    # Longest first avoids nested partial matches
    unique: Dict[str, str] = {}
    for mention in example["referent_mentions"]:
        surface = str(mention["surface"])
        unique.setdefault(surface.strip().lower(), surface)
    return sorted(unique.values(), key=lambda value: (-len(value), value.lower()))


def valid_mentions(example: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    context = example["context"]
    valid: List[Dict[str, Any]] = []
    errors: List[str] = []
    occupied: Dict[int, List[Tuple[int, int]]] = {}
    for mention in sorted(example["referent_mentions"], key=lambda item: tuple(item["span"])):
        turn, start, end = map(int, mention["span"])
        surface = str(mention["surface"])
        if turn < 0 or turn >= len(context) or start < 0 or end < start or end > len(context[turn]):
            errors.append("invalid_context_span")
            continue
        if context[turn][start:end] != surface:
            errors.append("surface_span_mismatch")
            continue
        if overlaps(start, end, occupied.setdefault(turn, [])):
            errors.append("overlapping_context_mentions")
            continue
        occupied[turn].append((start, end))
        valid.append({"surface": surface, "span": [turn, start, end]})
    return valid, sorted(set(errors))


def evidence_fields(example: Mapping[str, Any]) -> Iterable[Tuple[int, str, str]]:
    # Keep aligned with data.MAX_EVIDENCE_ITEMS
    for index, item in enumerate(list(example["evidence"])[:MAX_EVIDENCE_ITEMS]):
        yield index, "title", str(item.get("title", ""))
        yield index, "content", str(item.get("content", ""))


def field_key(index: int, field: str) -> str:
    return f"{index}::{field}"


def collect_premise_hits(example: Mapping[str, Any]) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    surfaces = referent_surfaces(example)
    for evidence_index, field, text in evidence_fields(example):
        occupied: List[Tuple[int, int]] = []
        for target_surface in surfaces:
            for match in re.finditer(re.escape(target_surface), text, flags=re.IGNORECASE):
                start, end = match.span()
                if overlaps(start, end, occupied):
                    continue
                occupied.append((start, end))
                actual = text[start:end]
                hits.append({
                    "field_key": field_key(evidence_index, field),
                    "evidence_index": evidence_index,
                    "field": field,
                    "start": start,
                    "end": end,
                    "surface": actual,
                    "target_surface": target_surface,
                    "n_words": count_words(actual),
                    "target_length": len(actual),
                })
    return sorted(hits, key=lambda item: (item["field_key"], item["start"]))


def best_control(
    text: str,
    *,
    target_words: int,
    target_length: int,
    banned_surfaces: set[str],
    avoid: Sequence[Tuple[int, int]],
) -> Optional[Dict[str, Any]]:
    # Original fallback: same word count, then one word
    word_counts = [target_words] + ([1] if target_words != 1 else [])
    for n_words in word_counts:
        candidates = []
        for start, end, surface in ngram_spans(text, n_words):
            if surface.strip().lower() in banned_surfaces or overlaps(start, end, avoid):
                continue
            score = (0 if n_words == target_words else 1, abs(len(surface) - target_length), start)
            candidates.append((score, start, end, surface))
        if candidates:
            _, start, end, surface = min(candidates, key=lambda item: item[0])
            return {"start": start, "end": end, "surface": surface}
    return None


def context_controls(
    example: Mapping[str, Any],
    mentions: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    context = example["context"]
    banned = {surface.strip().lower() for surface in referent_surfaces(example)}
    target_by_turn: Dict[int, List[Tuple[int, int]]] = {}
    for mention in mentions:
        turn, start, end = map(int, mention["span"])
        target_by_turn.setdefault(turn, []).append((start, end))
    chosen_by_turn: Dict[int, List[Tuple[int, int]]] = {}
    controls: List[Dict[str, Any]] = []
    for mention in mentions:
        target_turn, _, _ = map(int, mention["span"])
        surface = str(mention["surface"])
        remaining_turns = [
            index for index in range(len(context)) if index != target_turn
        ]
        # Prefer the target turn; never reuse occupied spans
        for turn in [target_turn, *remaining_turns]:
            candidate = best_control(
                context[turn],
                target_words=count_words(surface),
                target_length=len(surface),
                banned_surfaces=banned,
                avoid=target_by_turn.get(turn, []) + chosen_by_turn.get(turn, []),
            )
            if candidate is not None:
                controls.append({"turn": turn, **candidate, "target_surface": surface})
                chosen_by_turn.setdefault(turn, []).append((candidate["start"], candidate["end"]))
                break
    return controls


def premise_controls(
    example: Mapping[str, Any],
    hits: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    banned = {surface.strip().lower() for surface in referent_surfaces(example)}
    texts = {field_key(index, field): text for index, field, text in evidence_fields(example)}
    target_spans: Dict[str, List[Tuple[int, int]]] = {}
    for hit in hits:
        target_spans.setdefault(str(hit["field_key"]), []).append(
            (int(hit["start"]), int(hit["end"]))
        )
    chosen_spans: Dict[str, List[Tuple[int, int]]] = {}
    controls: List[Dict[str, Any]] = []
    for hit in hits:
        # Prefer the same evidence field
        preferred = [str(hit["field_key"])] + [key for key in texts if key != hit["field_key"]]
        for key in preferred:
            candidate = best_control(
                texts[key],
                target_words=int(hit["n_words"]),
                target_length=int(hit["target_length"]),
                banned_surfaces=banned,
                avoid=target_spans.get(key, []) + chosen_spans.get(key, []),
            )
            if candidate is not None:
                evidence_index, field = key.split("::", 1)
                controls.append({
                    "field_key": key,
                    "evidence_index": int(evidence_index),
                    "field": field,
                    **candidate,
                    "target_surface": hit["surface"],
                })
                chosen_spans.setdefault(key, []).append((candidate["start"], candidate["end"]))
                break
    return controls


def mask_context(
    example: Mapping[str, Any],
    spans: Sequence[Mapping[str, Any]],
    mask_token: str,
) -> Dict[str, Any]:
    # Do not mutate the source example
    transformed = copy.deepcopy(example)
    replacements: Dict[int, List[Tuple[int, int, str]]] = {}
    for item in spans:
        if "span" in item:
            turn, start, end = map(int, item["span"])
        else:
            turn, start, end = int(item["turn"]), int(item["start"]), int(item["end"])
        replacements.setdefault(turn, []).append(
            (start, end, placeholder(str(item["surface"]), mask_token))
        )
    for turn, turn_replacements in replacements.items():
        transformed["context"][turn] = replace_spans(
            transformed["context"][turn],
            turn_replacements,
        )
    return transformed


def mask_premise(
    example: Mapping[str, Any],
    spans: Sequence[Mapping[str, Any]],
    mask_token: str,
) -> Dict[str, Any]:
    transformed = copy.deepcopy(example)
    replacements: Dict[str, List[Tuple[int, int, str]]] = {}
    for item in spans:
        replacements.setdefault(str(item["field_key"]), []).append(
            (int(item["start"]), int(item["end"]), placeholder(str(item["surface"]), mask_token))
        )
    for key, field_replacements in replacements.items():
        evidence_index, field = key.split("::", 1)
        evidence_item = transformed["evidence"][int(evidence_index)]
        evidence_item[field] = replace_spans(str(evidence_item.get(field, "")), field_replacements)
    return transformed


def tokenizable(tokenizer: Any, example: Mapping[str, Any], max_length: int) -> bool:
    # Tokenizer errors should fail the run
    encoded = tokenizer(
        build_premise(example),
        build_hypothesis(example),
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    return bool(encoded.get("input_ids"))


def prepare(args: argparse.Namespace) -> None:
    examples = load_examples(args.input_file, args.dataset, require_referents=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, use_fast=True)

    # Fallback used by the original scripts
    mask_token = tokenizer.mask_token or "[MASK]"
    max_length = effective_max_length(tokenizer, args.model_key)
    rows: List[Dict[str, Any]] = []

    for example in examples:
        # Build once; reuse this file for pre- and post-FT
        mentions, errors = valid_mentions(example)
        premise_hits = collect_premise_hits(example)
        ctx_controls = context_controls(example, mentions)
        pre_controls = premise_controls(example, premise_hits)

        variants = {
            "clean": copy.deepcopy(example),
            "ctx_mask": mask_context(example, mentions, mask_token),
            "pre_mask": mask_premise(example, premise_hits, mask_token),
            "ctx_ctrl_mask_matched": mask_context(example, ctx_controls, mask_token),
            "pre_ctrl_mask_matched": mask_premise(example, pre_controls, mask_token),
        }
        valid_inputs = all(
            tokenizable(tokenizer, variant, max_length)
            for variant in variants.values()
        )

        # Keep all rows; summarize() filters on this flag
        diagnosable = bool(
            mentions
            and premise_hits
            and len(ctx_controls) == len(mentions)
            and len(pre_controls) == len(premise_hits)
            and not errors
            and valid_inputs
        )
        reasons = list(errors)
        if not mentions:
            reasons.append("no_valid_context_referent")
        if not premise_hits:
            reasons.append("no_premise_surface_match")
        if len(ctx_controls) != len(mentions):
            reasons.append("context_control_count_mismatch")
        if len(pre_controls) != len(premise_hits):
            reasons.append("premise_control_count_mismatch")
        if not valid_inputs:
            reasons.append("invalid_model_input")

        for condition in CONDITIONS:
            variant = variants[condition]
            rows.append({
                "id": example["id"],
                "dataset": args.dataset,
                "label": example["label"],
                "condition": condition,
                "diagnosable": diagnosable,
                "diagnostic_reason": ";".join(sorted(set(reasons))),
                "context_target_count": len(mentions),
                "premise_target_count": len(premise_hits),
                "context_control_count": len(ctx_controls),
                "premise_control_count": len(pre_controls),
                "premise": build_premise(variant),
                "hypothesis": build_hypothesis(variant),
            })

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    summary = {
        "n_examples": len(examples),
        "n_diagnosable": int(frame.loc[frame["condition"] == "clean", "diagnosable"].sum()),
        "mask_token": mask_token,
        "max_length": max_length,
        "output_file": str(output_path),
    }
    output_path.with_suffix(output_path.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def bootstrap_gap(
    nf_values: Sequence[float],
    pnf_values: Sequence[float],
    *,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, float]:
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")

    # Resample NF and PNF independently
    nf = np.asarray(nf_values, dtype=float)
    pnf = np.asarray(pnf_values, dtype=float)
    if len(nf) == 0 or len(pnf) == 0:
        return {"gap": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    gap = float(nf.mean() - pnf.mean())
    if len(nf) == len(pnf) == 1:
        return {"gap": gap, "ci_low": gap, "ci_high": gap}
    rng = np.random.default_rng(seed)
    boot_nf = rng.choice(nf, size=(n_bootstrap, len(nf)), replace=True).mean(axis=1)
    boot_pnf = rng.choice(pnf, size=(n_bootstrap, len(pnf)), replace=True).mean(axis=1)
    low, high = np.quantile(boot_nf - boot_pnf, [0.025, 0.975])
    return {"gap": gap, "ci_low": float(low), "ci_high": float(high)}


def _bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "t"})


def summarize(args: argparse.Namespace) -> None:
    # Join by keys, never row order
    pre = pd.read_csv(args.pre_predictions)
    post = pd.read_csv(args.post_predictions)
    required = {"id", "condition", "diagnosable", "gold_label", "pred_label", "gold_margin"}
    for name, frame in (("pre", pre), ("post", post)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} predictions are missing columns: {sorted(missing)}")
        if frame.duplicated(["id", "condition"]).any():
            raise ValueError(f"{name} predictions contain duplicate id/condition rows")

    # Require the same prepared rows in both files
    pre_keys = set(map(tuple, pre[["id", "condition", "gold_label"]].astype(str).to_numpy()))
    post_keys = set(map(tuple, post[["id", "condition", "gold_label"]].astype(str).to_numpy()))
    if pre_keys != post_keys:
        raise ValueError(
            "pre/post prediction files do not contain the same examples and conditions"
        )

    pre["gold_margin"] = pd.to_numeric(pre["gold_margin"], errors="coerce")
    post["gold_margin"] = pd.to_numeric(post["gold_margin"], errors="coerce")

    pre = pre.rename(columns={
        "diagnosable": "diagnosable_pre",
        "pred_label": "pred_label_pre",
        "gold_margin": "gold_margin_pre",
    })
    post = post.rename(columns={
        "diagnosable": "diagnosable_post",
        "pred_label": "pred_label_post",
        "gold_margin": "gold_margin_post",
    })
    pre_columns = [
        "id",
        "condition",
        "gold_label",
        "diagnosable_pre",
        "pred_label_pre",
        "gold_margin_pre",
    ]
    post_columns = [
        "id",
        "condition",
        "gold_label",
        "diagnosable_post",
        "pred_label_post",
        "gold_margin_post",
    ]
    merged = pre[pre_columns].merge(
        post[post_columns],
        on=["id", "condition", "gold_label"],
        how="inner",
        validate="one_to_one",
    )
    merged["joint_diagnosable"] = (
        _bool(merged["diagnosable_pre"])
        & _bool(merged["diagnosable_post"])
    )
    merged["observable"] = (
        np.isfinite(merged["gold_margin_pre"].to_numpy(dtype=float))
        & np.isfinite(merged["gold_margin_post"].to_numpy(dtype=float))
    )

    # PPS requires all five conditions in both states
    required_conditions = set(CONDITIONS)
    complete_ids = [
        example_id
        for example_id, part in merged.groupby("id")
        if (
            set(part["condition"]) == required_conditions
            and bool(part["joint_diagnosable"].all())
            and bool(part["observable"].all())
        )
    ]
    merged = merged[merged["id"].isin(complete_ids)].copy()
    if merged.empty:
        raise ValueError("no joint-diagnosable examples with all five conditions")

    values = {}
    for stage in ("pre", "post"):
        pivot = merged.pivot(
            index=["id", "gold_label"],
            columns="condition",
            values=f"gold_margin_{stage}",
        )
        pivot = pivot.rename(columns={
            "clean": f"m_clean_{stage}",
            "ctx_mask": f"m_ctx_{stage}",
            "pre_mask": f"m_pre_{stage}",
            "ctx_ctrl_mask_matched": f"m_ctx_ctrl_{stage}",
            "pre_ctrl_mask_matched": f"m_pre_ctrl_{stage}",
        })
        values[stage] = pivot
    per_example = values["pre"].join(values["post"]).reset_index()

    clean = merged[merged["condition"] == "clean"].copy()
    clean = clean[["id", "gold_label", "pred_label_pre", "pred_label_post"]]
    per_example = per_example.merge(clean, on=["id", "gold_label"], validate="one_to_one")

    for stage in ("pre", "post"):
        per_example[f"E_ctx_{stage}"] = (
            per_example[f"m_clean_{stage}"] - per_example[f"m_ctx_{stage}"]
        )
        per_example[f"E_pre_{stage}"] = (
            per_example[f"m_clean_{stage}"] - per_example[f"m_pre_{stage}"]
        )
        per_example[f"E_ctx_ctrl_{stage}"] = (
            per_example[f"m_clean_{stage}"] - per_example[f"m_ctx_ctrl_{stage}"]
        )
        per_example[f"E_pre_ctrl_{stage}"] = (
            per_example[f"m_clean_{stage}"] - per_example[f"m_pre_ctrl_{stage}"]
        )
        per_example[f"pps_{stage}"] = (
            per_example[f"E_pre_{stage}"] - per_example[f"E_pre_ctrl_{stage}"]
        ) - (
            per_example[f"E_ctx_{stage}"] - per_example[f"E_ctx_ctrl_{stage}"]
        )

    # Transitions use clean predictions only
    correct_pre = per_example["pred_label_pre"] == per_example["gold_label"]
    correct_post = per_example["pred_label_post"] == per_example["gold_label"]
    per_example["transition"] = np.select(
        [
            ~correct_pre & ~correct_post,
            ~correct_pre & correct_post,
            correct_pre & ~correct_post,
            correct_pre & correct_post,
        ],
        ["NNF", "PF", "NF", "PNF"],
        default="",
    )

    order = ["NNF", "PF", "NF", "PNF"]
    transition_rows = []
    for transition in order:
        part = per_example[per_example["transition"] == transition]
        transition_rows.append({
            "transition": transition,
            "n": int(len(part)),
            "share": float(len(part) / len(per_example)),
            "mean_pps_pre": float(part["pps_pre"].mean()) if len(part) else np.nan,
            "mean_pps_post": float(part["pps_post"].mean()) if len(part) else np.nan,
        })
    transition_summary = pd.DataFrame(transition_rows)
    nf = per_example.loc[per_example["transition"] == "NF", "pps_post"].to_numpy()
    pnf = per_example.loc[per_example["transition"] == "PNF", "pps_post"].to_numpy()
    gap = bootstrap_gap(nf, pnf, n_bootstrap=args.n_bootstrap, seed=args.bootstrap_seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_example.to_csv(output_dir / "pps_by_example.csv", index=False)
    transition_summary.to_csv(output_dir / "transition_summary.csv", index=False)
    summary = {
        "n_joint_diagnosable": int(len(per_example)),
        "n_nf": int(len(nf)),
        "n_pnf": int(len(pnf)),
        "post_ft_nf_mean_pps": float(np.mean(nf)) if len(nf) else None,
        "post_ft_pnf_mean_pps": float(np.mean(pnf)) if len(pnf) else None,
        "nf_minus_pnf_gap": None if np.isnan(gap["gap"]) else gap["gap"],
        "ci_low": None if np.isnan(gap["ci_low"]) else gap["ci_low"],
        "ci_high": None if np.isnan(gap["ci_high"]) else gap["ci_high"],
        "n_bootstrap": args.n_bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
    }
    (output_dir / "pps_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="create one fixed five-condition audit file",
    )
    prepare_parser.add_argument("--dataset", required=True, choices=["dialfact", "faithdial"])
    prepare_parser.add_argument("--model-key", required=True, choices=sorted(MODEL_SPECS))
    prepare_parser.add_argument("--tokenizer-name-or-path", required=True)
    prepare_parser.add_argument("--input-file", required=True)
    prepare_parser.add_argument("--output-file", required=True)

    summary_parser = subparsers.add_parser(
        "summarize",
        help="compute transitions and PPS from two evaluations",
    )
    summary_parser.add_argument("--pre-predictions", required=True)
    summary_parser.add_argument("--post-predictions", required=True)
    summary_parser.add_argument("--output-dir", required=True)
    summary_parser.add_argument("--n-bootstrap", type=int, default=5000)
    summary_parser.add_argument("--bootstrap-seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
