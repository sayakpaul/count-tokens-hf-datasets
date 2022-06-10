"""
Usage:
    python train_unigram.py --export_to_hub

Note that you'd need to execute `huggingface-cli login` before if you passed export_to_hub.

Reference:
    https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/tokenizer_training.ipynb
"""

import argparse
import logging

import datasets
import torch
from datasets import Dataset
from tokenizers import (
    Tokenizer,
    decoders,
    normalizers,
    pre_tokenizers,
    processors,
)
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
from transformers import AlbertTokenizerFast


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a unigram tokenizer on the wikitext dataset."
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size during training.",
    )
    parser.add_argument(
        "-vs",
        "--vocab-size",
        type=int,
        default=10000,
        help="Size of the desired vocabulary.",
    )
    parser.add_argument(
        "-l",
        "--limit",
        default=None,
        type=int,
        help="Limit the number of shards (used for debugging).",
    )
    parser.add_argument(
        "--export_to_hub",
        action="store_true",
    )

    args = parser.parse_args()
    return vars(args)


def get_unigram_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(Unigram())
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.Replace("``", '"'), normalizers.Replace("''", '"')]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    return tokenizer


def get_unigram_trainer(vocab_size: int) -> UnigramTrainer:
    trainer = UnigramTrainer(
        unk_token="<unk>",
        special_tokens=["[CLS]", "[SEP]", "<unk>", "<pad>", "[MASK]"],
        vocab_size=vocab_size,
    )
    return trainer


def main(args):
    wikitext = datasets.load_dataset(
        "wikitext", "wikitext-103-raw-v1", split="train"
    )

    if args.limit is not None:
        wikitext = wikitext[: args.limit]
        wikitext = Dataset.from_dict(wikitext)
        logging.info(f"Limiting the dataset to {args.limit} entries.")

    dataloader = torch.utils.data.DataLoader(
        wikitext, num_workers=0, batch_size=args.batch_size
    )
    logging.info("Training the tokenizer.")
    tokenizer = get_unigram_tokenizer()
    trainer = get_unigram_trainer(args.vocab_size)
    tokenizer.train_from_iterator(dataloader, trainer=trainer)
    logging.info("Tokenizer training complete!")

    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS]:0 $A:0 [SEP]:0",
        pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", cls_token_id),
            ("[SEP]", sep_token_id),
        ],
    )
    tokenizer.decoder = decoders.Metaspace()

    if args.export_to_hub:
        logging.info("Exporting the trained tokenzier to Hub.")
        new_tokenizer = AlbertTokenizerFast(tokenizer_object=tokenizer)
        new_tokenizer.push_to_hub("sayakpaul/unigram-tokenizer-wikitext")


if __name__ == "__main__":
    args = parse_args()
    main(args)
