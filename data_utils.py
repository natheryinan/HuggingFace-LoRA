
# data_utils.py
from typing import Dict, Any
from dataclasses import dataclass

from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
import torch


def load_imdb(split: str = "train"):
    """
    Load the IMDb dataset from HuggingFace Datasets.
    split: 'train', 'test' or 'unsupervised'
    """
    dataset = load_dataset("imdb")[split]
    return dataset


def prepare_imdb_for_classification(tokenizer: PreTrainedTokenizerBase,
                                    max_length: int = 256,
                                    split: str = "train"):
    """
    Return tokenized IMDb dataset for sequence classification.
    """
    raw = load_imdb(split)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
        )

    tokenized = raw.map(tokenize_fn, batched=True)
    tokenized = tokenized.remove_columns(["text"])
    tokenized.set_format(type="torch")
    return tokenized


def prepare_imdb_for_causal_lm(tokenizer: PreTrainedTokenizerBase,
                               max_length: int = 256,
                               split: str = "train"):
    """
    Prepare IMDb as plain text for causal language modeling (next-token prediction).
    """
    raw = load_imdb(split)

    def tokenize_fn(examples):
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        # For LM, labels are just shifted inputs; Trainer can handle this if labels=input_ids
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = raw.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch")
    return tokenized


@dataclass
class CausalLMCollator:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features):
        batch = {}
        keys = features[0].keys()
        for key in keys:
            batch[key] = torch.stack([f[key] for f in features])

        return batch
