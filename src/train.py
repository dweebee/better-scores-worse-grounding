"""Fine-tune a verifier and select the final checkpoint."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from data import VerifierDataset, dataset_name, load_examples, stratified_split
from model import MODEL_SPECS, effective_max_length, get_spec, load_training_model


def compute_metrics(eval_prediction: Any) -> Dict[str, float]:
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)

    # Missing dev classes still count as zero in Macro-F1
    class_ids = list(range(logits.shape[-1]))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=class_ids,
                average="macro",
                zero_division=0,
            )
        ),
    }


def training_arguments(**kwargs: Any) -> TrainingArguments:
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in parameters and "evaluation_strategy" in kwargs:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")

    # Do not silently drop run settings
    unsupported = sorted(set(kwargs) - set(parameters))
    if unsupported:
        raise TypeError(
            "Installed transformers does not support TrainingArguments: "
            + ", ".join(unsupported)
        )
    return TrainingArguments(**kwargs)


def trainer_with_tokenizer(*, tokenizer: Any, **kwargs: Any) -> Trainer:
    # Trainer API rename
    parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in parameters:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in parameters:
        kwargs["tokenizer"] = tokenizer
    return Trainer(**kwargs)


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return 10**18


def choose_dialfact_checkpoint(
    output_dir: Path,
    *,
    eval_dataset: VerifierDataset,
    collator: Any,
    eval_batch_size: int,
) -> Tuple[Path, list[Dict[str, Any]]]:
    candidates = sorted(output_dir.glob("checkpoint-*"), key=checkpoint_step)
    if not candidates:
        raise RuntimeError("no saved epoch checkpoints were found")
    rows = []
    for checkpoint in candidates:
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        args = training_arguments(
            output_dir=str(output_dir / "_checkpoint_selection"),
            per_device_eval_batch_size=eval_batch_size,
            report_to=[],
            remove_unused_columns=True,
        )
        evaluator = Trainer(
            model=model,
            args=args,
            eval_dataset=eval_dataset,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )
        metrics = evaluator.evaluate()
        rows.append({
            "checkpoint": str(checkpoint),
            "step": checkpoint_step(checkpoint),
            "macro_f1": float(metrics["eval_macro_f1"]),
            "accuracy": float(metrics["eval_accuracy"]),
        })
        del evaluator, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Earlier checkpoint wins ties
    best = max(rows, key=lambda row: (row["macro_f1"], -row["step"]))
    return Path(best["checkpoint"]), rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["dialfact", "faithdial"])
    parser.add_argument("--model-key", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument(
        "--train-file",
        required=True,
        help="Prepared source-training CSV or JSONL; not an audit file",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=7, choices=[7, 42])
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ds = dataset_name(args.dataset)

    spec = get_spec(args.model_key)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic flags used in the runs
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Audit files are not training inputs
    examples = load_examples(args.train_file, ds)
    train_examples, dev_examples = stratified_split(examples, dev_ratio=0.10, seed=args.seed)
    if not train_examples or not dev_examples:
        raise ValueError(
            "the stratified split requires non-empty train and "
            "checkpoint-selection partitions"
        )

    tokenizer, model, training_label_ids = load_training_model(
        args.model_name_or_path,
        model_key=args.model_key,
        dataset=ds,
    )
    max_length = effective_max_length(tokenizer, args.model_key)

    # Save the plain-base pre-FT state before training
    if spec.kind == "plain_base":
        initial_dir = output_dir / "initial"
        model.save_pretrained(initial_dir)
        tokenizer.save_pretrained(initial_dir)

    train_dataset = VerifierDataset(
        train_examples,
        tokenizer,
        max_length=max_length,
        training_label_ids=training_label_ids,
    )
    dev_dataset = VerifierDataset(
        dev_examples,
        tokenizer,
        max_length=max_length,
        training_label_ids=training_label_ids,
    )

    # Dynamic padding
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    train_batch_size = args.train_batch_size or spec.train_batch_size
    eval_batch_size = args.eval_batch_size or spec.eval_batch_size

    # DialFact effective batch size ~= 40
    if args.gradient_accumulation_steps is not None:
        gradient_accumulation = args.gradient_accumulation_steps
    elif ds == "dialfact":
        gradient_accumulation = max(1, math.ceil(40 / train_batch_size))
    else:
        gradient_accumulation = spec.faithdial_gradient_accumulation
    learning_rate = 3e-5 if ds == "dialfact" else spec.faithdial_learning_rate
    # Keep the two checkpoint-selection protocols separate
    faithdial_protocol = ds == "faithdial"

    hf_args = training_arguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_ratio=0.06,
        load_best_model_at_end=faithdial_protocol,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        save_total_limit=2 if faithdial_protocol else 5,
        logging_steps=50,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        full_determinism=True,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=True,
        group_by_length=False,
        gradient_checkpointing=args.gradient_checkpointing,
        save_safetensors=True,
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)] if faithdial_protocol else []
    trainer = trainer_with_tokenizer(
        tokenizer=tokenizer,
        model=model,
        args=hf_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    trainer.train()

    if faithdial_protocol:
        # Trainer has already restored the best FaithDial weights
        selected_source = Path(trainer.state.best_model_checkpoint or output_dir)
        selected_model = trainer.model
        checkpoint_rows = [{
            "checkpoint": str(selected_source),
            "macro_f1": (
                float(trainer.state.best_metric)
                if trainer.state.best_metric is not None
                else None
            ),
            "selection": "trainer_best_with_patience_2",
        }]
    else:
        # DialFact: select from saved epoch checkpoints
        selected_source, checkpoint_rows = choose_dialfact_checkpoint(
            output_dir,
            eval_dataset=dev_dataset,
            collator=collator,
            eval_batch_size=eval_batch_size,
        )
        selected_model = AutoModelForSequenceClassification.from_pretrained(selected_source)

    # Stable path used by evaluate.py
    selected_dir = output_dir / "selected"
    if selected_dir.exists():
        shutil.rmtree(selected_dir)
    selected_model.save_pretrained(selected_dir)
    tokenizer.save_pretrained(selected_dir)

    summary = {
        "dataset": ds,
        "model_key": args.model_key,
        "model_name_or_path": args.model_name_or_path,
        "seed": args.seed,
        "n_train": len(train_examples),
        "n_checkpoint_selection": len(dev_examples),
        "max_length": max_length,
        "learning_rate": learning_rate,
        "epochs": 3,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation,
        "selected_source_checkpoint": str(selected_source),
        "selected_model_dir": str(selected_dir),
        "selection_protocol": (
            "posthoc_epoch_checkpoint_macro_f1" if ds == "dialfact"
            else "trainer_best_macro_f1_with_early_stopping_patience_2"
        ),
        "checkpoint_selection": checkpoint_rows,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
