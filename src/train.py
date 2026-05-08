"""Fine-tune ``roberta-base`` on the balanced review dataset.

Mirrors the logic from notebooks/Amazon_Reviews_Fine_Tuning_Roberta_Base.ipynb so
the same training run can happen from a script (e.g. on CI runners with GPU, or
locally) instead of from Colab. Heavy imports are deferred so the module imports
cheaply.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data import INT_TO_LABEL, encode_labels


def train_roberta(
    balanced_df: pd.DataFrame,
    output_dir: str = "./fine_tuned_roberta",
    base_model: str = "roberta-base",
    epochs: int = 3,
    batch_size: int = 16,
    test_size: float = 0.3,
    seed: int = 42,
) -> tuple[object, object, dict]:
    """Train ``roberta-base`` for 3-class sentiment.

    Returns ``(trainer, tokenizer, eval_report)``.
    """
    from datasets import Dataset
    from sklearn.metrics import classification_report
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    df = encode_labels(balanced_df)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    dataset = Dataset.from_pandas(df[["reviews.text", "actual_sentiment"]])
    dataset = dataset.rename_column("actual_sentiment", "labels")

    def tokenize(batch):
        return tokenizer(batch["reviews.text"], truncation=True)

    tokenized = dataset.map(tokenize, batched=True)
    split = tokenized.train_test_split(test_size=test_size, seed=seed)

    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=3)
    args = TrainingArguments(
        output_dir=str(Path(output_dir) / "training_runs"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        logging_strategy="epoch",
        seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    trainer.train()

    predictions = trainer.predict(split["test"])
    import numpy as np

    y_pred = np.argmax(predictions.predictions, axis=1)
    y_test = predictions.label_ids
    target_names = [INT_TO_LABEL[i] for i in range(3)]
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer, tokenizer, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune roberta-base on balanced reviews.")
    parser.add_argument("--data", required=True, help="Path to balanced reviews CSV.")
    parser.add_argument("--out", default="./fine_tuned_roberta", help="Output directory.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    _, _, report = train_roberta(
        df,
        output_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print("Eval report (per-class F1):")
    for cls in ("negative", "neutral", "positive"):
        print(f"  {cls}: {report[cls]['f1-score']:.3f}")
    print(f"  accuracy: {report['accuracy']:.3f}")


if __name__ == "__main__":
    main()
