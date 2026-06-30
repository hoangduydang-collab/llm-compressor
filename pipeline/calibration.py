"""Build the calibration dataset used by ``oneshot``.

Factored out of the example scripts (load -> chat-template -> tokenize). Returns
a tokenized ``datasets.Dataset`` ready to hand to ``oneshot(dataset=...)``.
"""

from pipeline.config import CalibrationConfig


def build_calibration_dataset(cal: CalibrationConfig, tokenizer):
    """Load, format and tokenize the calibration set described by ``cal``."""
    from datasets import load_dataset

    split = f"{cal.dataset_split}[:{cal.num_samples}]"
    ds = load_dataset(cal.dataset_id, split=split)
    ds = ds.shuffle(seed=cal.seed)

    column_names = ds.column_names
    has_messages = "messages" in column_names
    has_text = "text" in column_names

    def preprocess(example):
        if has_messages:
            return {
                "text": tokenizer.apply_chat_template(
                    example["messages"], tokenize=False
                )
            }
        if has_text:
            return {"text": example["text"]}
        # Fall back to the first string column.
        first = column_names[0]
        return {"text": str(example[first])}

    ds = ds.map(preprocess)

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=cal.max_seq_length,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)
    return ds
