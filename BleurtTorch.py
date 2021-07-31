import transformers
from torch import nn
import numpy as np

class BleurtModel(nn.Module):
    """
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = transformers.BertModel(config)
        self.dense = nn.Linear(config.hidden_size, 1)

    def forward(self, input_mask, segment_ids, input_ids=None, inputs_embeds=None):
        if (input_ids is not None) == (inputs_embeds is not None):
            raise ValueError('Either input_ids or imput_embeds should be not None, and one of them should be None')
        cls_state = self.bert(input_ids=input_ids,
                              attention_mask=input_mask,
                              token_type_ids=segment_ids,
                              inputs_embeds=inputs_embeds).pooler_output
        return self.dense(cls_state)


# these three methods are just copied from bleurt lib: lib/python3.8/site-packages/bleurt/encoding.py
# such that no need to run tensorflow which is imported in that file
def _truncate_seq_pair(tokens_ref, tokens_cand, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_ref) + len(tokens_cand)
        if total_length <= max_length:
            break
        if len(tokens_ref) > len(tokens_cand):
            tokens_ref.pop()
        else:
            tokens_cand.pop()


def encode_example(reference, candidate, tokenizer, max_seq_length):
    """Tokenization and encoding of an example rating.

    Args:
      reference: reference sentence.
      candidate: candidate sentence.
      tokenizer: instance of lib.tokenizers.Tokenizer.
      max_seq_length: maximum length of BLEURT's input after tokenization.

    Returns:
      input_ids: contacatenated token ids of the reference and candidate.
      input_mask: binary mask to separate the input from the padding.
      segment_ids: binary mask to separate the sentences.
    """
    # Tokenizes, truncates and concatenates the sentences, as in:
    #  bert/run_classifier.py
    tokens_ref = tokenizer.tokenize(reference)
    tokens_cand = tokenizer.tokenize(candidate)

    _truncate_seq_pair(tokens_ref, tokens_cand, max_seq_length - 3)

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_ref:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    for token in tokens_cand:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids


def encode_batch(references, candidates, tokenizer, max_seq_length):
    """Encodes a batch of sentence pairs to be fed to a BLEURT checkpoint.

    Args:
      references: list of reference sentences.
      candidates: list of candidate sentences.
      tokenizer: BERT-style WordPiece tokenizer.
      max_seq_length: maximum length of BLEURT's input after tokenization.

    Returns:
      A triplet (input_ids, input_mask, segment_ids), all numpy arrays with type
        np.int64<n_sentences, max_seq_length>.
    """
    encoded_examples = []
    for ref, cand in zip(references, candidates):
        triplet = encode_example(ref, cand, tokenizer, max_seq_length)
        example = np.stack(triplet)
        encoded_examples.append(example)
    stacked_examples = np.stack(encoded_examples)
    assert stacked_examples.shape == (len(encoded_examples), 3, max_seq_length)
    return (stacked_examples[:, 0, :], stacked_examples[:, 1, :],
            stacked_examples[:, 2, :])
