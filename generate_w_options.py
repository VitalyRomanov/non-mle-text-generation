import argparse
import json

import tqdm

# Parameters:
#
#             input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
#                 The sequence used as a prompt for the generation. If :obj:`None` the method initializes it as an empty
#                 :obj:`torch.LongTensor` of shape :obj:`(1,)`.
#             max_length (:obj:`int`, `optional`, defaults to 20):
#                 The maximum length of the sequence to be generated.
#             min_length (:obj:`int`, `optional`, defaults to 10):
#                 The minimum length of the sequence to be generated.
#             do_sample (:obj:`bool`, `optional`, defaults to :obj:`False`):
#                 Whether or not to use sampling ; use greedy decoding otherwise.
#             early_stopping (:obj:`bool`, `optional`, defaults to :obj:`False`):
#                 Whether to stop the beam search when at least ``num_beams`` sentences are finished per batch or not.
#             num_beams (:obj:`int`, `optional`, defaults to 1):
#                 Number of beams for beam search. 1 means no beam search.
#             temperature (:obj:`float`, `optional`, defaults tp 1.0):
#                 The value used to module the next token probabilities.
#             top_k (:obj:`int`, `optional`, defaults to 50):
#                 The number of highest probability vocabulary tokens to keep for top-k-filtering.
#             top_p (:obj:`float`, `optional`, defaults to 1.0):
#                 If set to float < 1, only the most probable tokens with probabilities that add up to :obj:`top_p` or
#                 higher are kept for generation.
#             repetition_penalty (:obj:`float`, `optional`, defaults to 1.0):
#                 The parameter for repetition penalty. 1.0 means no penalty. See `this paper
#                 <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
#             pad_token_id (:obj:`int`, `optional`):
#                 The id of the `padding` token.
#             bos_token_id (:obj:`int`, `optional`):
#                 The id of the `beginning-of-sequence` token.
#             eos_token_id (:obj:`int`, `optional`):
#                 The id of the `end-of-sequence` token.
#             length_penalty (:obj:`float`, `optional`, defaults to 1.0):
#                 Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the
#                 model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer
#                 sequences.
#             no_repeat_ngram_size (:obj:`int`, `optional`, defaults to 0):
#                 If set to int > 0, all ngrams of that size can only occur once.
#             encoder_no_repeat_ngram_size (:obj:`int`, `optional`, defaults to 0):
#                 If set to int > 0, all ngrams of that size that occur in the ``encoder_input_ids`` cannot occur in the
#                 ``decoder_input_ids``.
#             bad_words_ids(:obj:`List[List[int]]`, `optional`):
#                 List of token ids that are not allowed to be generated. In order to get the tokens of the words that
#                 should not appear in the generated text, use :obj:`tokenizer(bad_word,
#                 add_prefix_space=True).input_ids`.
#             num_return_sequences(:obj:`int`, `optional`, defaults to 1):
#                 The number of independently computed returned sequences for each element in the batch.
#             max_time(:obj:`float`, `optional`, defaults to None):
#                 The maximum amount of time you allow the computation to run for in seconds. generation will still
#                 finish the current pass after allocated time has been passed.
#             attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
#                 Mask to avoid performing attention on padding token indices. Mask values are in ``[0, 1]``, 1 for
#                 tokens that are not masked, and 0 for masked tokens. If not provided, will default to a tensor the same
#                 shape as :obj:`input_ids` that masks the pad token. `What are attention masks?
#                 <../glossary.html#attention-mask>`__
#             decoder_start_token_id (:obj:`int`, `optional`):
#                 If an encoder-decoder model starts decoding with a different token than `bos`, the id of that token.
#             use_cache: (:obj:`bool`, `optional`, defaults to :obj:`True`):
#                 Whether or not the model should use the past last key/values attentions (if applicable to the model) to
#                 speed up decoding.
#             num_beam_groups (:obj:`int`, `optional`, defaults to 1):
#                 Number of groups to divide :obj:`num_beams` into in order to ensure diversity among different groups of
#                 beams. `this paper <https://arxiv.org/pdf/1610.02424.pdf>`__ for more details.
#             diversity_penalty (:obj:`float`, `optional`, defaults to 0.0):
#                 This value is subtracted from a beam's score if it generates a token same as any beam from other group
#                 at a particular time. Note that :obj:`diversity_penalty` is only effective if ``group beam search`` is
#                 enabled.
#             prefix_allowed_tokens_fn: (:obj:`Callable[[int, torch.Tensor], List[int]]`, `optional`):
#                 If provided, this function constraints the beam search to allowed tokens only at each step. If not
#                 provided no constraint is applied. This function takes 2 arguments :obj:`inputs_ids` and the batch ID
#                 :obj:`batch_id`. It has to return a list with the allowed tokens for the next generation step
#                 conditioned on the previously generated tokens :obj:`inputs_ids` and the batch ID :obj:`batch_id`. This
#                 argument is useful for constrained generation conditioned on the prefix, as described in
#                 `Autoregressive Entity Retrieval <https://arxiv.org/abs/2010.00904>`__.
#             output_attentions (:obj:`bool`, `optional`, defaults to `False`):
#                 Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
#                 returned tensors for more details.
#             output_hidden_states (:obj:`bool`, `optional`, defaults to `False`):
#                 Whether or not to return trhe hidden states of all layers. See ``hidden_states`` under returned tensors
#                 for more details.
#             output_scores (:obj:`bool`, `optional`, defaults to `False`):
#                 Whether or not to return the prediction scores. See ``scores`` under returned tensors for more details.
#             return_dict_in_generate (:obj:`bool`, `optional`, defaults to `False`):
#                 Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
#             forced_bos_token_id (:obj:`int`, `optional`):
#                 The id of the token to force as the first generated token after the :obj:`decoder_start_token_id`.
#                 Useful for multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token
#                 needs to be the target language token.
#             forced_eos_token_id (:obj:`int`, `optional`):
#                 The id of the token to force as the last generated token when :obj:`max_length` is reached.
import os
from pprint import pprint

import datasets


def test_T5(args):
    from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
    from SeqT5 import SeqT5
    import data

    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = SeqT5.from_pretrained(os.path.join(args.ckpt_path, "best_gmodel.pt")).cpu()

    splits = ['test', 'valid']
    dataset = data.load_dataset(args.data_path, splits)
    partition = dataset.splits['test' if args.use_test else 'valid']

    if args.use_parameter_grid:
        generator_params = get_grid()
    else:
        generator_params = [{
            "temperature": args.t,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "do_sample": args.do_sample,
            "max_length": args.max_length,
            "repetition_penalty": args.repetition_penalty,
            "num_beams": args.num_beams,
            "length_penalty": args.length_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "num_beam_groups": args.num_beam_groups,
            "diversity_penalty": args.diversity_penalty
        }]

    bleu_metric = datasets.load_metric('sacrebleu')
    rouge_metric = datasets.load_metric('rouge')

    metrics = {
        "bleu": [],
        "rouge1f1": [],
        "rouge2f1": [],
        "rougeLf1": []
    }

    # experiments = []
    # outs = []



    for par_ind, g_params in enumerate(generator_params):
        output_name = f"experiment_{args.note.replace(' ', '_')}_{par_ind}.jsonl"
        # if os.path.isfile(output_name):
        #     raise Exception("Output file exists, change experiment note")
        with open(output_name, "w") as sink:
            print(g_params)
            print("\n\n\n")
            sink.write(f"{json.dumps(g_params)}\n")

            for ind, entry in tqdm.tqdm(enumerate(partition)):
                input_ids = entry["source"] - 1
                labels = entry["target"] - 1

                outputs = model.generate(input_ids.reshape(1,-1), **g_params)[0][1:]

                inp = tokenizer.decode(input_ids, skip_special_tokens=True)
                trg = tokenizer.decode(labels, skip_special_tokens=True)
                pred = tokenizer.decode(outputs, skip_special_tokens=True)

                bleu = bleu_metric.compute(predictions=[pred], references=[[trg]]) # bleu["score"]
                rouge = rouge_metric.compute(predictions=[pred], references=[inp]) # rouge["rougeL"].high.fmeasure

                # metrics["bleu"].append(bleu["score"])
                # metrics["rouge1f1"].append(rouge["rouge1"].high.fmeasure)
                # metrics["rouge2f1"].append(rouge["rouge2"].high.fmeasure)
                # metrics["rougeLf1"].append(rouge["rougeL"].high.fmeasure)

                metrics = {
                    "bleu": bleu["score"],
                    "rouge1f1": rouge["rouge1"].high.fmeasure,
                    "rouge2f1": rouge["rouge2"].high.fmeasure,
                    "rougeLf1": rouge["rougeL"].high.fmeasure
                }

                out_entry = {
                    "input": inp,
                    "target": trg,
                    "output": pred,
                    "metrics": metrics
                }

                # print("s: ", tokenizer.decode(input_ids, skip_special_tokens=True))
                # print("t: ", tokenizer.decode(labels, skip_special_tokens=True))
                # print("g: ", tokenizer.decode(outputs, skip_special_tokens=True))
                # print("")

                sink.write(f"{json.dumps(out_entry)}\n")

                # if ind > 10 :
                #     break

    # references for repetition penalty https://huggingface.co/blog/how-to-generate
    # print()


def get_grid():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Load args
    parser.add_argument("note", default='Describe experiment')
    parser.add_argument("--ckpt_path", default='checkpoint/SeqT5Mle_t5_mle')
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--use_test", action='store_true')
    parser.add_argument("--t", default=1., type=float)
    parser.add_argument("--top_p", default=0.9, type=float)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--do_sample", default=True, type=bool)
    parser.add_argument("--max_length", default=100, type=int)
    parser.add_argument("--repetition_penalty", default=1., type=float)
    parser.add_argument("--num_beams", default=1, type=int)
    parser.add_argument("--length_penalty", default=1, type=int)
    parser.add_argument("--no_repeat_ngram_size", default=0, type=int)
    parser.add_argument("--num_beam_groups", default=1, type=int)
    parser.add_argument("--diversity_penalty", default=0, type=int)
    parser.add_argument("--use_parameter_grid", action='store_true')

    args = parser.parse_args()

    test_T5(args)