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
