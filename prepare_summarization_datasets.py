# https://huggingface.co/datasets?filter=task_ids:summarization,languages:en,multilinguality:monolingual
#
# https://github.com/google-research-datasets/sentence-compression
# https://github.com/abisee/cnn-dailymail
import os

from datasets import load_dataset

from split_train_val_test import write_splits


def load_text_compression_dataset():
    dataset = load_dataset("msr_text_compression", data_dir="~/.manual_dir/msr_text_compression")
    return dataset


def load_cnn_daily_mail():
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    return dataset


def load_arxiv():
    dataset = load_dataset("arxiv_dataset", data_dir="~/.manual_dir/arxiv")
    return dataset


def generate_text_compression_dataset(args):
    dataset = load_text_compression_dataset()

    def generate(split):
        for sample in split:
            source = sample["source_text"]
            for target in sample["targets"]["compressed_text"]:
                yield args.prefix + source, target
                break

    dataset_path = os.path.join(args.output, "text_compression")
    if not os.path.isdir(dataset_path):
        os.mkdir(dataset_path)

    write_splits(
        dataset_path,
        train=generate(dataset["train"]), val=generate(dataset["validation"]), test=generate(dataset["test"]),
        src="original", tgt="summary", tokenizer=args.tokenizer
    )


def generate_arxiv_dataset(args):
    dataset = load_arxiv()

    def generate(split):
        for source, target in zip(split["abstract"], split["title"]):
            # source = sample["abstract"]
            # target = sample["title"]
            yield args.prefix + source, target

    dataset_path = os.path.join(args.output, "arxiv")
    if not os.path.isdir(dataset_path):
        os.mkdir(dataset_path)

    train_test = dataset["train"].train_test_split()
    val_test = train_test["test"].train_test_split()

    write_splits(
        dataset_path,
        train=generate(train_test["train"]), val=generate(val_test["train"]), test=generate(val_test["test"]),
        src="original", tgt="summary", tokenizer=args.tokenizer
    )


def generate_cnn_dailymail_dataset(args):
    dataset = load_cnn_daily_mail()

    def generate(split):
        for source, target in zip(split["article"], split["highlights"]):
            yield args.prefix + source, target

    dataset_path = os.path.join(args.output, "cnn_dailymail")
    if not os.path.isdir(dataset_path):
        os.mkdir(dataset_path)

    write_splits(
        dataset_path,
        train=generate(dataset["train"]), val=generate(dataset["validation"]), test=generate(dataset["test"]),
        src="original", tgt="summary", tokenizer=args.tokenizer
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="regular", help="regular|bpe|t5-small")
    parser.add_argument("--output", default="data-bin")
    parser.add_argument("--prefix", default="")

    args = parser.parse_args()

    output_dir = os.path.join(args.output, args.tokenizer)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    args.output = output_dir

    generate_text_compression_dataset(args)
    generate_arxiv_dataset(args)
    generate_cnn_dailymail_dataset(args)
