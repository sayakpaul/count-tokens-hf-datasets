"""
Test run locally:
    python prepare_jsonl_shards.py --limit 1000 --log_interval 20

To run with the full dataset:
    python prepare_jsonl_shards.py --gcs_bucket hf-datasets-wikitext --log_interval 20

Thanks to Quentin Lhoest of 🤗 for providing the meat of this script.
"""

import argparse
import logging
import os

import datasets
from datasets import Dataset
from google.cloud import storage
from tqdm.auto import tqdm

GCP_PROJECT = "fast-ai-exploration"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preparing jsonl shards of the wikitext dataset."
    )
    parser.add_argument(
        "--gcs_bucket",
        default=None,  #  "hf-datasets-wikitext"
        type=str,
        help="If not provided stores the jsonl shards locally. Useful for local debugging."
        " For running on Dataflow GCS bucket is mandatory.",
    )
    parser.add_argument(
        "--local_dir",
        default="tmp",
        type=str,
        help="Local directory to store the jsonl shards. Will be deleted after upload to GCS is complete.",
    )
    parser.add_argument(
        "--batch_size",
        default=None,
        type=int,
        help=f"Batch size to use. If not provided defaults to {datasets.config.DEFAULT_MAX_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help=f"Limits the number of entries within the dataset. Mostly used for debugging purposes.",
    )
    parser.add_argument(
        "--use_multiprocessing",
        action="store_true",
    )
    parser.add_argument("--num_shards", default=100, type=int)
    parser.add_argument("--log_interval", default=None, type=int)
    args = parser.parse_args()
    return args


def upload_json_to_gcs(filepath: str, gcs_bucket: str) -> None:
    """Uploads a JSON file to a GCS bucket."""
    if not hasattr(upload_json_to_gcs, "bucket"):
        client = storage.Client(project=GCP_PROJECT)
        bucket = client.get_bucket(gcs_bucket)
        upload_json_to_gcs.bucket = bucket
    blob = upload_json_to_gcs.bucket.blob(filepath)
    blob.upload_from_filename(filepath)


def main(args):
    total_shards_prepped = 0
    if not os.path.exists(args.local_dir):
        os.makedirs(args.local_dir)

    wikitext = datasets.load_dataset(
        "wikitext", "wikitext-103-raw-v1", split="train"
    )

    if args.limit is not None:
        wikitext = wikitext[: args.limit]
        wikitext = Dataset.from_dict(wikitext)
        logging.info(f"Limiting the dataset to {args.limit} entries.")

    for index in tqdm(range(args.num_shards)):
        size = len(wikitext) // args.num_shards
        shard = Dataset(
            wikitext.data.slice(size * index, size),
            fingerprint=f"{wikitext._fingerprint}_{index}",
        )

        filepath = os.path.join(
            args.local_dir,
            f"wikitext-data-{index}-of-{args.num_shards - 1}.jsonl",
        )
        shard.to_json(
            filepath,
            batch_size=args.batch_size
            if args.batch_size is not None
            else datasets.config.DEFAULT_MAX_BATCH_SIZE,
            num_proc=os.cpu_count() - 1 if args.use_multiprocessing else 1,
        )
        total_shards_prepped += 1

        if args.gcs_bucket is not None:
            upload_json_to_gcs(filepath, args.gcs_bucket)

        if args.log_interval is not None and index % args.log_interval == 0:
            logging.info(
                f"Prepared {total_shards_prepped} out of {args.num_shards}."
            )


if __name__ == "__main__":
    args = parse_args()
    main(args)
