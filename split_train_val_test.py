import logging
import os
import random
import sys
from typing import Tuple, Iterable

from transformers import T5Tokenizer

import dictionary
from tokenizer import Tokenizer

from data import load_dictionaries
from indexed_dataset import IndexedDatasetBuilder
from tokenizer import create_subword_tokenizer, tokenize_line


def write_split(split, path, split_name, src, tgt):
    with open(os.path.join(path, f"{split_name}.{src}"), "w", encoding='utf-8') as srcfile:
        with open(os.path.join(path, f"{split_name}.{tgt}"), "w", encoding='utf-8') as tgtfile:
            for s, d in split:
                s = s.replace("\n", " ")
                d = d.replace("\n", " ")
                srcfile.write(f"{s}\n")
                tgtfile.write(f"{d}\n")


def write_splits(
        path: str, train: Iterable[Tuple[str, str]] = None, val: Iterable[Tuple[str, str]] = None,
        test: Iterable[Tuple[str, str]] = None, src: str = None, tgt: str = None, tokenizer: str = None
):
    """
    Write data splits and their binarization to disk
    :param path: output_path
    :param train: List of tuples for training. First element of tuple is the source text, and the second - target text.
    :param val: List of tuples for validation. First element of tuple is the source text, and the second - target text.
    :param test: List of tuples for testing. First element of tuple is the source text, and the second - target text.
    :param src: Code for source
    :param tgt: Code for target
    :param tokenizer: String that represents tokenizer. Possible values: bpe|regular|t5-XXX. Need to specify model size
        for t5 tokenizer.
    :return: None
    """
    assert src is not None and tgt is not None
    assert train is not None and val is not None and test is not None

    write_split(train, path, "train", src, tgt)
    write_split(val, path, "valid", src, tgt)
    write_split(test, path, "test", src, tgt)

    if tokenizer == "bpe":
        tok = create_subword_tokenizer("multi", 1000000)

        def tokenize(text):
            return [t for t in tok(text.replace("\n", " "))]

        logging.warning("Using bpe tokenizer")
    if tokenizer.startswith("t5"):
        tok = T5Tokenizer.from_pretrained(tokenizer)

        def tokenize(text):
            """
            This tokenization should be compatible with both <pad> and </s>
            """
            return [t for t in tok.tokenize(text.replace("\n", " "))]

    elif tokenizer == "regular":
        from nltk import RegexpTokenizer
        tokenizer = RegexpTokenizer("[\w]+|[^\w\s]")

        # tokenize = tokenize_line

        def tokenize(text):
            return tokenizer.tokenize(text.replace("\n", " "))

        logging.warning("Using regular tokenizer")
    else:
        raise ValueError("Supported tokenizers are: bpe|regular|t5-XXX")

    if not tokenizer.startswith("t5"):
        def create_dictionary(direction):
            logging.warning("Only train set is used for generating the dictionary")
            dict_ = Tokenizer.build_dictionary(os.path.join(path, f"train.{direction}"), tokenize=tokenize)
            dict_.save(os.path.join(path, f"dict.{direction}.txt"))
    else:
        def create_dictionary(direction):
            dict_ = dictionary.Dictionary()
            t5_vocab = [[tok.sp_model.id_to_piece(id), id] for id in range(tok.sp_model.get_piece_size())]
            assert t5_vocab.pop(0)[0] == "<pad>"
            assert t5_vocab.pop(0)[0] == "</s>"
            assert t5_vocab.pop(0)[0] == "<unk>"
            for word, id in t5_vocab:
                dict_.add_symbol(word)
            for word, id in sorted(
                    zip(tok.additional_special_tokens, tok.additional_special_tokens_ids), key=lambda x: x[1]
            ):
                dict_.add_symbol(word)

            t5_vocab_dict = dict(((w, id) for w, id in t5_vocab))
            t5_vocab_dict.update(
                zip(tok.additional_special_tokens, tok.additional_special_tokens_ids)
            )


            for word, id in dict_.indices.items():
                if word in {"<Lua heritage>", "<pad>", "</s>", "<unk>"}:
                    continue
                assert id == t5_vocab_dict[word] + 1

            dict_.finalize()
            dict_.save(os.path.join(path, f"dict.{direction}.txt"))

    create_dictionary(src)
    create_dictionary(tgt)

    # dataset = load_raw_text_dataset(path, ["train", "valid", "test"], src, tgt, maxlen=None, tokenize_fn=tokenize)

    len_src = 0
    len_tgt = 0

    src_dict, tgt_dict = load_dictionaries(path, src, tgt)

    for split in ["train", "valid", "test"]:
        src_bin = IndexedDatasetBuilder(os.path.join(path, f"{split}.{src}-{tgt}.{src}.bin"))
        tgt_bin = IndexedDatasetBuilder(os.path.join(path, f"{split}.{src}-{tgt}.{tgt}.bin"))

        src_lines = open(os.path.join(path, f"{split}.{src}")).readlines()
        tgt_lines = open(os.path.join(path, f"{split}.{tgt}")).readlines()
        assert len(src_lines) == len(tgt_lines)
        for src_line, tgt_line in zip(src_lines, tgt_lines):
            src_tokens = Tokenizer.tokenize(src_line.strip("\n"), src_dict, tokenize=tokenize, add_if_not_exist=False)
            tgt_tokens = Tokenizer.tokenize(tgt_line.strip("\n"), tgt_dict, tokenize=tokenize, add_if_not_exist=False)
            src_bin.add_item(src_tokens)
            tgt_bin.add_item(tgt_tokens)
            len_src += len(src_tokens)
            len_tgt += len(tgt_tokens)
        src_bin.finalize(os.path.join(path, f"{split}.{src}-{tgt}.{src}.idx"))
        tgt_bin.finalize(os.path.join(path, f"{split}.{src}-{tgt}.{tgt}.idx"))

    print(f"Source average length {len_src / len(src_dict)}")
    print(f"Target average length {len_tgt / len(tgt_dict)}")


def generate_nmt_splits(
        src_lang, tgt_lang, src_path, tgt_path, output_path,
        max_sent_len=1000, test_val_size=0.1, train_size=0.8, maximum_sents=90000, tokenizer=None,
        shuffle=True, random_seed=None
):
    """

    :param src_lang: Source code
    :param tgt_lang: Target code
    :param src_path: File with source sentences
    :param tgt_path: File with target sentences
    :param output_path: path where data will be stored. For output filenames, see `write_splits`
    :param max_sent_len: Maximum sentence length
    :param test_val_size: Fraction of dataset used for testing and validation
    :param train_size: Fraction of dataset used for training
    :param maximum_sents: Maximum number of sentences to use
    :param tokenizer: Tokenizer code, see `write_splits`
    :param shuffle: Shuffle data before splitting
    :param random_seed: Seed
    :return:
    """

    assert train_size + test_val_size * 2 == 1

    def read_lines(path):
        with open(path, "r") as file:
            return file.readlines()

    src_lines = read_lines(src_path)
    tgt_lines = read_lines(tgt_path)

    assert len(src_lines) == len(tgt_lines)

    filtered = [
        (src, tgt) for ind, (src, tgt) in enumerate(zip(src_lines, tgt_lines)) if
        src != tgt and (max_sent_len > len(src.strip()) != 0) and (max_sent_len > len(tgt.strip()) != 0)
        and ind < maximum_sents
    ]

    if random_seed is not None:
        random.seed(random_seed)

    if shuffle:
        random.shuffle(filtered)

    total_sents = len(filtered)
    bad_counter = len(tgt_lines) - total_sents

    def generate_slice(sents, min_ind, max_ind):
        for ind, src_tgt in enumerate(sents):
            if min_ind <= ind < max_ind:
                yield src_tgt

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    write_splits(
        output_path,
        generate_slice(filtered, 0, int(total_sents * train_size)),
        generate_slice(filtered, int(total_sents * train_size) + 1, int(total_sents * (train_size + test_val_size))),
        generate_slice(filtered, int(total_sents * (train_size + test_val_size)), total_sents),
        src_lang, tgt_lang, tokenizer
    )

    print("bad sentences:", bad_counter)


def generate_nmt_splits2(
        src_lang, tgt_lang, dataset_path, output_path, tokenizer=None, prefix=""
):

    def generate_partition(partition):
        src_sents = open(os.path.join(dataset_path, f"{partition}.{src_lang}")).readlines()
        tgt_sents = open(os.path.join(dataset_path, f"{partition}.{tgt_lang}")).readlines()
        assert len(src_sents) == len(tgt_sents)

        for src_sent, tgt_sent in zip(src_sents, tgt_sents):
            yield prefix+src_sent, tgt_sent

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    write_splits(
        output_path,
        generate_partition("train"),
        generate_partition("valid"),
        generate_partition("test"),
        src_lang, tgt_lang, tokenizer
    )


if __name__ == "__main__":
    src_lang = 'en'
    tgt_lang = 'ro'

    file_dir = 'data-bin/'

    generate_nmt_splits2(
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        dataset_path="wmt16_en-ro",
        output_path=file_dir + src_lang + '-' + tgt_lang,
        tokenizer="t5-small",
        prefix="translate English to Romanian: "
    )

    # # TODO check out dataset binarization
    # generate_nmt_splits(
    #     src_lang,
    #     tgt_lang,
    #     src_path=file_dir + src_lang + '-' + tgt_lang + '/tokenized_full.' + src_lang,
    #     tgt_path=file_dir + src_lang + '-' + tgt_lang + '/tokenized_full.' + tgt_lang,
    #     output_path=file_dir + src_lang + '-' + tgt_lang,
    #     max_sent_len=1000,
    #     test_val_size=0.1,
    #     train_size=0.8,
    #     maximum_sents=90000,
    #     tokenizer="t5-small"
    # )

    # src_file_n = open(file_dir + src_lang + '-' + tgt_lang + '/tokenized_full.' + src_lang, 'r')
    # tgt_file_n = open(file_dir + src_lang + '-' + tgt_lang + '/tokenized_full.' + tgt_lang, 'r')
    #
    # src_lines = src_file_n.readlines()
    # tgt_lines = tgt_file_n.readlines()
    #
    # src_file_n.close()
    # tgt_file_n.close()
    #
    # src_file_train = open(file_dir + src_lang + '-' + tgt_lang + '/train.' + src_lang, 'w')
    # tgt_file_train = open(file_dir + src_lang + '-' + tgt_lang + '/train.' + tgt_lang, 'w')
    #
    # src_file_test = open(file_dir + src_lang + '-' + tgt_lang + '/test.' + src_lang, 'w')
    # tgt_file_test = open(file_dir + src_lang + '-' + tgt_lang + '/test.' + tgt_lang, 'w')
    #
    # src_file_val = open(file_dir + src_lang + '-' + tgt_lang + '/valid.' + src_lang, 'w')
    # tgt_file_val = open(file_dir + src_lang + '-' + tgt_lang + '/valid.' + tgt_lang, 'w')
    #
    # # limit = 802000
    # assert len(src_lines) == len(tgt_lines)
    #
    # max_sent_len = 1000
    # test_sents = 0
    # val_sents = 0
    # train_sents = 0
    # test_val_size = 6000
    # train_size = 80000
    # bad_counter = 0
    # for i in range(len(src_lines)):
    #     # if limit == 0:
    #     #     break
    #     src = src_lines[i]
    #     tgt = tgt_lines[i]
    #     # check for duplicates and empty lines and filter by length
    #     if src != tgt and (max_sent_len > len(src.strip()) != 0) and (max_sent_len > len(tgt.strip()) != 0):
    #         # limit -= 1
    #         if test_sents < test_val_size:
    #             src_file_test.write(src)
    #             tgt_file_test.write(tgt)
    #             test_sents += 1
    #         elif val_sents < test_val_size:
    #             src_file_val.write(src)
    #             tgt_file_val.write(tgt)
    #             val_sents += 1
    #         else:
    #             src_file_train.write(src)
    #             tgt_file_train.write(tgt)
    #             train_sents += 1
    #             if train_sents == train_size:
    #                 break
    #     else:
    #         print(src, tgt)
    #         bad_counter += 1
    #
    # print("bad sentences:", bad_counter)
    #
    # src_file_train.close()
    # tgt_file_train.close()
    #
    # src_file_test.close()
    # tgt_file_test.close()
    #
    # src_file_val.close()
    # tgt_file_val.close()

    # procedure for files - clean and split to 3 parts - 6k test, 6k validation, remainder - train
