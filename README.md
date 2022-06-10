# count-tokens-hf-datasets

This project shows how to derive the total number of training tokens from a large text dataset from 🤗 datasets with [Apache Beam](https://beam.apache.org/) and [Cloud Dataflow](https://cloud.google.com/dataflow).

In NLP, the number of training tokens dictates model scaling behaviour (refer to [1, 2]). However, counting the number
of tokens can be non-trivial for large-scale datasets. Hence this project.

## Steps

This project leverages to [`datasets`](https://huggingface.co/datasets) library from Hugging Face 🤗 to load a text dataset. It then prepares shards of the dataset. Once the shards have been prepared, it then executes an Apache Beam pipeline on Cloud Dataflow to generate the number of training tokens. We use Apache Beam to use distributed processing which significantly speeds up the process. We use Cloud Dataflow because it provides seamless autoscaling capabilities. Below are the steps:

* Load the [`wikitext`](https://huggingface.co/datasets/wikitext) dataset using `datasets`. It has over a million number of training samples. So, it's a good candidate for demonstration purposes. 
* Generate `.jsonl` shards of the dataset and have them uploaded to a [Google Cloud Storage (GCS) bucket](https://cloud.google.com/storage). The shard generation step is needed because
Apache Beam reads data on a shard-by-shard basis and is therefore able to induce parallel processing across many workers.
* Train a tokenizer using the 🤗 [`tokenizers`](https://github.com/huggingface/tokenizers) library with the `wikitext` dataset from 🤗 `datasets`. The tokenizer I trained is available here: https://huggingface.co/sayakpaul/unigram-tokenizer-wikitext.
* Execute the Apache Beam pipeline on Dataflow for generating the number of training tokens. The steps of the Beam pipeline
  are as follows:

 <br>
 <div align="center">
 <img src=https://i.ibb.co/qnVS95V/image.png width=200>
 </div>

## Running the code

You can play around with the code locally. But to run it on Cloud Dataflow you'd
need to have a [billing-enabled](https://cloud.google.com/billing/docs/how-to/modify-project) account on Google Cloud Platform along with the necessary quotas.

Get started by installing the dependencies: `pip install -r requirements.txt`. 

Here's the sequence in which the scripts are expected to be executed:

```sh
$ python prepare_jsonl_shards.py --gcs_bucket hf-datasets-wikitext --log_interval 20
$ python train_unigram.py --export_to_hub 
$ python count_training_tokens.py --runner DataflowRunner
```

_Make sure you've run `hugging-cli login` before running `python train_unigram.py --export_to_hub `._

In case you're using Cloud Dataflow, you'll also need to create a bucket on GCS. Refer to the [official
documentation](https://cloud.google.com/storage/docs/creating-buckets) to know how.

## Expected output

After the execution of `count_training_tokens.py`, one should expect to get a JSON file in the 
location provided during the execution. The content of that JSON file should look like so:

```json
{"training_tokens_count": 5403900}
```

## Costs

Here's component-by-component breakdown of the costs:

* Compute ([n1-highmem-16](https://cloud.google.com/compute/docs/general-purpose-machines)): USD 0.95 (Total time up: An hour)
* Storage (GCS): USD 0.10 (Assumed 5 GBs of storage)
* Dataflow: USD 0.30 (Total CPU workers used: 4)

**The total costs are under USD 5.**

## Acknowledgements

* Thanks to the ML GDE Program (ML Ecosystem Team) at Google that provided GCP credits to support the project.
* Thanks to [Quentin Lhoest](https://github.com/lhoestq) from Hugging face (maintainer of 🤗 `datasets`) for insightful discussions and for provding the shard generation snippet.

## References

[1] Scaling Laws for Neural Language Models (OpenAI): https://arxiv.org/abs/2001.08361

[2] Training Compute-Optimal Large Language Models (DeepMind): https://arxiv.org/abs/2203.15556

