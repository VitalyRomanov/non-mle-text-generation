from nltk.tokenize import word_tokenize
# import nltk
# nltk.download('punkt')

# lowercases and tokenizes a file

file_dir = 'data-bin/cs-en/'

inp_file_name = "full.en"
f = open(file_dir + inp_file_name, 'r')

out_file_name = "tokenized_" + inp_file_name
tokenized_f = open(file_dir + out_file_name, 'w')

src_lines = f.readlines()

for i in range(len(src_lines)):
    line = str.lower(src_lines[i])
    tokens = word_tokenize(line)
    tokens_string = ' '.join(tokens)
    tokenized_f.write(tokens_string + "\n")

f.close()
tokenized_f.close()