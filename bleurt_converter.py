import bleurt
from BleurtTorch import BleurtModel
from bleurt import score as bleurt_score
import sys
sys.argv = sys.argv[:1] ##thanks https://github.com/google-research/bleurt/issues/4

# import tensorflow.compat.v1 as tf
import tensorflow as tf
import torch
import transformers
import json

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

checkpoint = "bleurt/bleurt-base-128"
# imported = tf.saved_model.load_v2(checkpoint) was like this before
imported = tf.saved_model.load(checkpoint)

state_dict = {}
for variable in imported.variables:
    n = variable.name
    if n.startswith('global'):
        continue
    data = variable.numpy()
    if 'kernel' in n:  # was only for "dense"
        data = data.T
    n = n.split(':')[0]
    n = n.replace('/','.')
    n = n.replace('_','.')
    n = n.replace('kernel','weight')
    if 'LayerNorm' in n:
        n = n.replace('beta','bias')
        n = n.replace('gamma','weight')
    elif 'embeddings' in n:
        n = n.replace('word.embeddings','word_embeddings')
        n = n.replace('position.embeddings','position_embeddings')
        n = n.replace('token.type.embeddings','token_type_embeddings')
        n = n + '.weight'
    state_dict[n] = torch.from_numpy(data)

config = transformers.BertConfig()
bleurt_model = BleurtModel(config)
bleurt_model.load_state_dict(state_dict, strict=False)  # strict=False added otherwise crashes.
# Should be safe, according to this https://github.com/huggingface/transformers/issues/6882#issuecomment-884730078
for param in bleurt_model.parameters():
    param.requires_grad = False
bleurt_model.eval()

scorer = bleurt_score.BleurtScorer(checkpoint)

## this is what the answer should be (using bleurt's python API)
references = ["a bird chirps by the window and decided to quit"]
candidates = ["a bird chirps by the window but then changed its mind"]
scores = scorer.score(references=references, candidates=candidates)
print(scores)

with open(f'{checkpoint}/bleurt_config.json','r') as f:
    bleurt_config = json.load(f)

max_seq_length = bleurt_config["max_seq_length"]
vocab_file = f'{checkpoint}/{bleurt_config["vocab_file"]}'
do_lower_case = bleurt_config["do_lower_case"]

# tokenizer = bleurt.lib.tokenization.FullTokenizer(    # this is what was before
#     vocab_file=vocab_file, do_lower_case=do_lower_case)

tokenizer = bleurt.lib.tokenizers.create_tokenizer(
    vocab_file=vocab_file, do_lower_case=do_lower_case, sp_model=None)

input_ids, input_mask, segment_ids = bleurt.encoding.encode_batch(
    references, candidates, tokenizer, max_seq_length)

## this is the answer that the pytorch model gives
bleurt_torch_score = bleurt_model(input_ids = torch.from_numpy(input_ids),
                                  input_mask = torch.from_numpy(input_mask),
                                  segment_ids = torch.from_numpy(segment_ids))

print(bleurt_torch_score)

# torch.save(bleurt_model.state_dict(), "bleurt/bleurt-base-128-torch.pb")

# usage example
checkpoint = "bleurt/bleurt-base-128-torch.pb"
config = transformers.BertConfig()
restored_bleurt_model = BleurtModel(config)
restored_bleurt_model.load_state_dict(torch.load(checkpoint))
for param in restored_bleurt_model.parameters():
    param.requires_grad = False
restored_bleurt_model.eval()

restored_bleurt_torch_score = restored_bleurt_model(input_ids = torch.from_numpy(input_ids),
                                                    input_mask = torch.from_numpy(input_mask),
                                                    segment_ids = torch.from_numpy(segment_ids))

print(restored_bleurt_torch_score)