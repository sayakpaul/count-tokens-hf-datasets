import argparse
import json
import logging
import pprint
from datetime import datetime

import apache_beam as beam
from google.cloud import storage
from transformers import AutoTokenizer

GCP_PROJECT = "fast-ai-exploration"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Beam pipeline for tokenizing text shards and obtaining a count of the number of total tokens."
    )
    parser.add_argument(
        "--gcs_bucket",
        default="hf-datasets-wikitext",  # For running on Dataflow, GCS bucket is mandatory.
        type=str,
        help="GCS bucket from which the jsonl shards are to be read.",
    )
    parser.add_argument(
        "--tokenizer_path",
        default="sayakpaul/unigram-tokenizer-wikitext",
        type=str,
        help="If not provided stores the jsonl shards locally. Useful for local debugging.",
    )
    parser.add_argument(
        "-r",
        "--runner",
        type=str,
        choices=["DirectRunner", "DataflowRunner"],
        help="The runner for the pipeline.",
    )
    parser.add_argument(
        "-w",
        "--max-num-workers",
        default="500",
        type=str,
        help="Number of maximum workers for Dataflow",
    )
    parser.add_argument(
        "-m",
        "--machine-type",
        type=str,
        default="n1-standard-1",
    )
    args = parser.parse_args()
    return args


def upload_string(val: str, bucket_name: str, file_path: str) -> None:
    """Uploads a string to a GCP bucket."""
    client = storage.Client(GCP_PROJECT)
    bucket = client.bucket(bucket_name)
    dest_blob = bucket.blob(file_path)
    dest_blob.upload_from_string(val)


def main(args):
    # Defining the beam pipeline options.
    beam_timestamp = datetime.utcnow().strftime("%y%m%d-%H%M%S")
    pipeline_args_dict = {
        "job_name": f"wikitext-count-tokens-{beam_timestamp}",
        "runner": args.runner,
        "machine_type": args.machine_type,
        "num_workers": "1",
        "max_num_workers": args.max_num_workers,
        "setup_file": "setup.py",
        "project": GCP_PROJECT,
        "region": "us-central1",
        "gcs_location": f"gs://{args.gcs_bucket}",
        "temp_location": f"gs://{args.gcs_bucket}/temp",
        "staging_location": f"gs://{args.gcs_bucket}/staging",
        "save_main_session": "True",
    }

    pipeline_args = [(f"--{k}", v) for k, v in pipeline_args_dict.items()]
    pipeline_args = [x for y in pipeline_args for x in y]

    logging.info(
        f"Executing beam pipeline with args:\n{pprint.pformat(pipeline_args_dict)}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    with beam.Pipeline(argv=pipeline_args) as p:
        _ = (
            p
            | "Read text shards"
            >> beam.io.ReadFromText(
                file_pattern=f"gs://{args.gcs_bucket}/tmp/wikitext-data-*-of-*.jsonl"
            )
            | "Load as JSON" >> beam.Map(json.loads)
            | "Tokenize text shards"
            >> beam.FlatMap(lambda x: tokenizer(x["text"]))
            | "Count the number of tokens" >> beam.combiners.Count.Globally()
            | "Training sample count"
            >> beam.Map(lambda x: json.dumps({"training_tokens_count": x}))
            | "Upload to GCS"
            >> beam.Map(
                lambda x: upload_string(
                    x, args.gcs_bucket, "training-token-counts.json"
                )
            )
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
