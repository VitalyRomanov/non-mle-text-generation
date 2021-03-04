import logging
import os

import torch

from indexed_dataset import IndexedDatasetBuilder
from tokenizer import create_subword_tokenizer


def write_split(split, path, split_name, src, tgt):
    with open(os.path.join(path, f"{split_name}.{src}"), "w", encoding='utf-8') as srcfile:
        with open(os.path.join(path, f"{split_name}.{tgt}"), "w", encoding='utf-8') as tgtfile:
            for s, d in split:
                s = s.replace("\n", " ")
                d = d.replace("\n", " ")
                srcfile.write(f"{s}\n")
                tgtfile.write(f"{d}\n")



def write_splits(path, train=None, val=None, test=None, src=None, tgt=None):
    assert src is not None and tgt is not None
    assert train is not None and val is not None and test is not None

    write_split(train, path, "train", src, tgt)
    write_split(val, path, "valid", src, tgt)
    write_split(test, path, "test", src, tgt)

    from data import load_raw_text_dataset
    from tokenizer import Tokenizer

    tok = create_subword_tokenizer("en", 200000)
    def tokenize(text):
        return [t for t in tok(text.replace("\n", " "))]

    def create_dictionary(direction):
        dict_ = Tokenizer.build_dictionary(os.path.join(path, f"train.{direction}"), tokenize=tokenize)
        dict_.save(os.path.join(path, f"dict.{direction}.txt"))

    create_dictionary(src)
    create_dictionary(tgt)

    dataset = load_raw_text_dataset(path, ["train", "valid", "test"], src, tgt, maxlen=None)

    len_src = 0
    len_tgt = 0

    for split in ["train", "valid", "test"]:
        src_bin = IndexedDatasetBuilder(os.path.join(path, f"{split}.{src}-{tgt}.{src}.bin"))
        tgt_bin = IndexedDatasetBuilder(os.path.join(path, f"{split}.{src}-{tgt}.{tgt}.bin"))
        assert len(dataset.splits[split].src) == len(dataset.splits[split].dst)
        for i in range(len(dataset.splits[split].src)):
            src_bin.add_item(dataset.splits[split].src[i]-1)
            tgt_bin.add_item(dataset.splits[split].dst[i]-1)
            len_src += len(dataset.splits[split].src[i])
            len_tgt += len(dataset.splits[split].dst[i])
        src_bin.finalize(os.path.join(path, f"{split}.{src}-{tgt}.{src}.idx"))
        tgt_bin.finalize(os.path.join(path, f"{split}.{src}-{tgt}.{tgt}.idx"))

    print(f"Source average length {len_src / len(dataset.splits[split].src)}")
    print(f"Target average length {len_tgt / len(dataset.splits[split].dst)}")






if __name__ == "__main__":

    src_lang = 'cs'
    tgt_lang = 'en'

    file_dir = 'data-bin/'

    src_file_n = open(file_dir + src_lang + '-' + tgt_lang + '/tokenized_full.' + src_lang, 'r')
    tgt_file_n = open(file_dir + src_lang + '-' + tgt_lang + '/tokenized_full.' + tgt_lang, 'r')

    src_lines = src_file_n.readlines()
    tgt_lines = tgt_file_n.readlines()

    src_file_n.close()
    tgt_file_n.close()

    src_file_train = open(file_dir + src_lang + '-' + tgt_lang + '/train.' + src_lang, 'w')
    tgt_file_train = open(file_dir + src_lang + '-' + tgt_lang + '/train.' + tgt_lang, 'w')

    src_file_test = open(file_dir + src_lang + '-' + tgt_lang + '/test.' + src_lang, 'w')
    tgt_file_test = open(file_dir + src_lang + '-' + tgt_lang + '/test.' + tgt_lang, 'w')

    src_file_val = open(file_dir + src_lang + '-' + tgt_lang + '/valid.' + src_lang, 'w')
    tgt_file_val = open(file_dir + src_lang + '-' + tgt_lang + '/valid.' + tgt_lang, 'w')

    # limit = 802000
    assert len(src_lines) == len(tgt_lines)

    max_sent_len = 1000
    test_sents = 0
    val_sents = 0
    train_sents = 0
    test_val_size = 6000
    train_size = 80000
    bad_counter = 0
    for i in range(len(src_lines)):
        # if limit == 0:
        #     break
        src = src_lines[i]
        tgt = tgt_lines[i]
        # check for duplicates and empty lines and filter by length
        if src != tgt and (max_sent_len > len(src.strip()) != 0) and (max_sent_len > len(tgt.strip()) != 0):
            # limit -= 1
            if test_sents < test_val_size:
                src_file_test.write(src)
                tgt_file_test.write(tgt)
                test_sents += 1
            elif val_sents < test_val_size:
                src_file_val.write(src)
                tgt_file_val.write(tgt)
                val_sents += 1
            else:
                src_file_train.write(src)
                tgt_file_train.write(tgt)
                train_sents += 1
                if train_sents == train_size:
                    break
        else:
            print(src, tgt)
            bad_counter += 1

    print("bad sentences:", bad_counter)

    src_file_train.close()
    tgt_file_train.close()

    src_file_test.close()
    tgt_file_test.close()

    src_file_val.close()
    tgt_file_val.close()

    # procedure for files - clean and split to 3 parts - 6k test, 6k validation, remainder - train
