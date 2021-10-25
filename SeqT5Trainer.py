import logging
import math
import random
from copy import copy
from typing import Dict

import numpy as np
from sklearn.preprocessing import normalize
from torch import cuda
from torch.autograd import Variable
from tqdm import tqdm

import utils
from ModelTrainer import ModelTrainer, update_learning_rate
import torch
from discriminator import Discriminator, AttDiscriminator, GumbelDiscriminator, T5Discriminator, \
    T5SemanticDiscriminator, BleurtDiscriminator, BleurtEmbDiscriminator
import os
import json


class SeqT5Trainer(ModelTrainer):
    def __init__(self, args, task_prefix=""):
        """
        Init SeqT5 trainer
        :param args:
        :param task_prefix: For summarization use "summarize: ", for nmt use "translate English to German: "
        """
        super(SeqT5Trainer, self).__init__(args)
        self.task_prefix = torch.LongTensor(self.t5_tokenizer.encode(task_prefix))

    def create_generator(self, args):
        from transformers import T5Tokenizer
        from SeqT5 import SeqT5

        self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
        if self.args.g_ckpt_path is not None:
            print(f"Loading pretrained generator from checkpoint {self.args.g_ckpt_path}")
            self.generator = SeqT5.from_pretrained(self.args.g_ckpt_path)
        else:
            self.generator = SeqT5.from_pretrained('t5-small')
        if self.args.freeze_encoder:
            self.generator.encoder.requires_grad = False

    def create_discriminator(self, args):
        # raise NotImplementedError()
        if self.args.d_ckpt_path is not None:
            logging.debug(f"Loading pretrained discriminator from checkpoint {self.args.d_ckpt_path}")
            self.discriminator = torch.load(self.args.d_ckpt_path)
        else:
            self.discriminator = AttDiscriminator(args, self.dataset.src_dict, self.dataset.dst_dict,
                                              use_cuda=self.use_cuda)
        print("Discriminator loaded successfully!")

    def create_models(self, args):
        self.create_generator(args)
        self.create_discriminator(args)

        if self.use_cuda:
            # if torch.cuda.device_count() > 1:
            #     self.discriminator = torch.nn.DataParallel(self.discriminator).cuda()
            #     self.generator = torch.nn.DataParallel(self.generator).cuda()
            # else:
            self.generator.cuda()
            if hasattr(self, "discriminator"):
                self.discriminator.cuda()
        else:
            if hasattr(self, "discriminator"):
                self.discriminator.cpu()
            self.generator.cpu()

    def handicap_discriminator(self):
        pass
        # # TODO need this?
        # # fix discriminator word embedding (as Wu et al. do)
        # if hasattr(self, "discriminator"):
        #     for p in self.discriminator.embed_src_tokens.parameters():
        #         p.requires_grad = False
        #     for p in self.discriminator.embed_trg_tokens.parameters():
        #         p.requires_grad = False

    # def create_losses(self):
    #     # define loss function
    #     super(SeqT5Trainer, self).create_losses()
    #     self.pg_criterion = lambda pred, true, reward, modified_logits, predicted_tokens: \
    #         self._pg_criterion(
    #             self._logsoftmax(pred),
    #             self.transform_for_t5(true), # transfor for t5 is now in compute_pg_loss
    #             reward,
    #             self._logsoftmax(modified_logits) if modified_logits is not None else None,
    #             self.transform_for_t5(predicted_tokens) if predicted_tokens is not None else None
    #         )

    def transform_for_t5(self, tensor):
        return tensor - 1

    def transform_from_t5(self, tensor):
        return tensor + 1

    def wrap_for_output(self, sample, logits, modified_logits=None, output_tokens=None, input_onehot=None, output_onehot=None, target_onehot=None):
        if input_onehot is not None: # add zeros to use indexing from 1
            zeros = torch.zeros((input_onehot.shape[0], input_onehot.shape[1], 1)).to(input_onehot.device)
            input_onehot = torch.cat([zeros, input_onehot], dim=2)
            zeros = torch.zeros((output_onehot.shape[0], output_onehot.shape[1], 1)).to(output_onehot.device)
            output_onehot = torch.cat([zeros, output_onehot], dim=2)
            zeros = torch.zeros((target_onehot.shape[0], target_onehot.shape[1], 1)).to(target_onehot.device)
            target_onehot = torch.cat([zeros, target_onehot], dim=2)

        if output_tokens is not None:
            pred = output_tokens
        elif output_onehot is not None:
            pred = output_onehot.argmax(-1) - 1
        else:
            pred = logits.argmax(-1)

        output = {
            "logits": logits,
            "target": sample["target"],
            "mask": self.get_length_mask(sample["target"]),
            "prediction": self.transform_from_t5(pred),
            "input_onehot": input_onehot,
            "output_onehot": output_onehot,
            "target_onehot": target_onehot,
            "modified_logits": modified_logits,
        }

        # output["loss"] = self.g_criterion(
        #     output["logits"][output["mask"], :],
        #     self.transform_for_t5(output["target"])[output["mask"]]
        # )
        return output

    def compute_mle_loss(self, generator_input: Dict, generator_output: Dict):
        mask = generator_output["mask"]
        return self.g_criterion(
            generator_output["logits"][mask, :],
            self.transform_for_t5(generator_output["target"])[mask]
        )

    def compute_pg_loss(self, generator_input: Dict, generator_output: Dict, reward):
        prediction = generator_output.get("prediction", None)
        return self.pg_criterion(
            generator_output["logits"], self.transform_for_t5(generator_output['target']), reward,
            generator_output.get("modified_logits", None),
            self.transform_for_t5(prediction) if prediction is not None else None
        )

    def sequential_generation(self, sample, decoding_style="rl", top_k=0, top_p=1.0, temp=.2, ss_prob=0.):
        t5out = self.generator(
            self.transform_for_t5(sample['net_input']['src_tokens']), attention_mask=sample["attention_mask"],
            labels=self.transform_for_t5(sample['target']), decoding_style=decoding_style, top_k=top_k, top_p=top_p,
            temperature=temp, epsilon=self.args.imp_smpl_epsilon, ss_prob=ss_prob
        )

        # if decoding_style == "gumbel":
        #     return self.wrap_for_output(sample, t5out.logits, input_onehot=t5out.input_onehot, output_onehot=t5out.output_onehot, target_onehot=t5out.target_onehot)
        return self.wrap_for_output(
            sample, t5out.logits, modified_logits=t5out.modified_logits, output_tokens=t5out.output_tokens,
            input_onehot=t5out.input_onehot, output_onehot=t5out.output_onehot, target_onehot=t5out.target_onehot
        )

    def teacher_forcing_generation(self, sample):
        t5out = self.generator(
            self.transform_for_t5(sample['net_input']['src_tokens']), attention_mask=sample["attention_mask"],
            labels=self.transform_for_t5(sample['target']), decoding_style="tf"
        )

        return self.wrap_for_output(
            sample, t5out.logits, modified_logits=t5out.modified_logits, output_tokens=t5out.output_tokens,
            input_onehot=t5out.input_onehot, output_onehot=t5out.output_onehot, target_onehot=t5out.target_onehot
        )

    def eval_generation(self, sample):
        return self.sequential_generation(sample, decoding_style=self.sequential_decoding_style, top_k=1, temp=.5)

    def save_generator(self, path):
        self.generator.save_pretrained(path)


class SeqT5Mle(SeqT5Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_strategy = "mle"  # alternate | mle | rl
        self.sequential_decoding_style = "rl"

    def create_discriminator(self, args):
        pass
        # self.discriminator = T5Discriminator(args, self.dataset.src_dict, self.dataset.dst_dict,
        #                                      use_cuda=self.use_cuda)
        self.discriminator = T5SemanticDiscriminator()


class SeqT5RL(SeqT5Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_strategy = "alternate"  # alternate | mle | rl
        self.sequential_decoding_style = "rl"

    def create_discriminator(self, args):
        # self.discriminator = T5Discriminator(args, self.dataset.src_dict, self.dataset.dst_dict,
        #                                      use_cuda=self.use_cuda)
        if self.args.d_ckpt_path is not None:
            print(f"Loading pretrained discriminator from checkpoint {self.args.d_ckpt_path}")
            self.discriminator = torch.load(self.args.d_ckpt_path)
        else:
            self.discriminator = T5SemanticDiscriminator()


class SeqT5Bleurt(SeqT5Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_strategy = "alternate"  # alternate | mle | rl
        self.sequential_decoding_style = "rl"

    def create_discriminator(self, args):
        if self.args.d_ckpt_path is not None:
            print(f"Loading pretrained discriminator from checkpoint {self.args.d_ckpt_path}")
            self.discriminator = torch.load(self.args.d_ckpt_path)
        else:
            self.discriminator = BleurtDiscriminator(decode_fn = lambda pred: self.decode_sentences(pred))

    def create_optimizers(self, args):
        # define optimizer
        self.g_optimizer = eval("torch.optim." + args.g_optimizer)(filter(lambda x: x.requires_grad,
                                                                          self.generator.parameters()),
                                                                   args.g_learning_rate)
        self.d_optimizer = None

    def compute_discriminator_score(self, generator_input: Dict, generator_output: Dict):
        input_token_ids = generator_input["net_input"]["src_tokens"]
        target_token_ids = generator_output["target"]
        predicted_token_ids = generator_output["prediction"]
        reference = self.choose_discriminator_reference(input_token_ids, target_token_ids)
        reward = self.discriminator(reference, predicted_token_ids)  # dim (bsize x 1)
        return reward

    def choose_discriminator_reference(self, source, target):
        """
        Allows to choose whether to use source tokens or target tokens as a reference.
        """
        return target

    # def create_models(self, args):
    #     self.create_generator(args)
    #     self.create_discriminator(args)
    #
    #     if self.use_cuda:
    #         # if torch.cuda.device_count() > 1:
    #         #     self.discriminator = torch.nn.DataParallel(self.discriminator).cuda()
    #         #     self.generator = torch.nn.DataParallel(self.generator).cuda()
    #         # else:
    #         # self.generator.cuda()
    #         self.generator.cuda(f'cuda:{self.args.gpuid[0]}')  # manually placing generator to the specified gpu
    #         if hasattr(self, "discriminator"):
    #             self.discriminator.cuda(f'cuda:{self.args.gpuid[0]}')  #.cuda(f'cuda:{cuda.device_count() - 1 - self.args.gpuid[0]}')  # selects other gpu
    #     else:
    #         if hasattr(self, "discriminator"):
    #             self.discriminator.cpu()
    #         self.generator.cpu()


class SeqEmbT5Bleurt(SeqT5Bleurt):
    def __init__(self, *args, **kwargs):
        super(SeqEmbT5Bleurt, self).__init__(*args, **kwargs)
        self.create_embedding_index()

    def create_embedding_index(self):
        import faiss
        bert_embeddings = self.get_target_embedder().weight.detach().numpy()
        self.faiss_index = faiss.IndexFlatIP(bert_embeddings.shape[1])   # build the index
        self.faiss_index.add(normalize(bert_embeddings))                  # add vectors to the index
        # alternative hnsw implementation. With specified params achieves good search quality - on avg 1.5 mismatch|sent
        # import hnswlib
        # self.hnsw_graph = hnswlib.Index(space='cosine', dim=bert_embeddings.shape[1]) # possible options are l2, cosine or ip
        # ef - the size of the dynamic list for the nearest neighbors (used during the search). Higher ef leads to more
        # accurate but slower search. The value ef of can be anything between k and the size of the dataset.
        # M - the number of bi-directional links created for every new elem during construction. Reasonable range: 2-100
        # self.hnsw_graph.init_index(max_elements=bert_embeddings.shape[0], ef_construction = bert_embeddings.shape[0], M = 500)
        # self.hnsw_graph.add_items(bert_embeddings)

    def create_generator(self, args):
        from transformers import T5Tokenizer
        from SeqT5 import SeqEmbT5

        self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
        if self.args.g_ckpt_path is not None:
            print(f"Loading pretrained generator from checkpoint {self.args.g_ckpt_path}")
            self.generator = SeqEmbT5.from_pretrained(self.args.g_ckpt_path)
        else:
            # self.generator = SeqEmbT5.from_pretrained('t5-small')
            from transformers import T5Config
            config = T5Config(d_model=128, d_ff=512)
            self.generator = SeqEmbT5(config)
        if self.args.freeze_encoder:
            self.generator.encoder.requires_grad = False

    def create_discriminator(self, args):
        if self.args.d_ckpt_path is not None:
            print(f"Loading pretrained discriminator from checkpoint {self.args.d_ckpt_path}")
            self.discriminator = torch.load(self.args.d_ckpt_path)
        else:
            self.discriminator = BleurtEmbDiscriminator(decode_fn = lambda pred: self.decode_sentences(pred))
        self.discriminator.eval()

    def targets_to_bert_ids(self, targets):
        targets_decoded = self.discriminator.decode_fn(targets) # targets in english
        targets_bert_tokens_np = np.zeros(targets.shape, dtype=np.int64)
        for i, t in enumerate(targets_decoded):
            bert_token_ids = self.discriminator.tokenizer.convert_tokens_to_ids(self.discriminator.tokenizer.tokenize(t)+["[SEP]"])
            proper_seq_len = min(len(bert_token_ids), targets.shape[1])
            targets_bert_tokens_np[i, :proper_seq_len] = bert_token_ids[:proper_seq_len]
        target_bert_tokens = torch.from_numpy(targets_bert_tokens_np)
        target_bert_tokens = target_bert_tokens.cuda()
        return target_bert_tokens

    def sequential_generation(self, sample, decoding_style="rl", top_k=0, top_p=1.0, temp=.2, ss_prob=0.):
        t5out = self.generator(
            self.transform_for_t5(sample['net_input']['src_tokens']), attention_mask=sample["attention_mask"],
            labels=self.transform_for_t5(sample['target']), decoding_style=decoding_style, top_k=top_k, top_p=top_p,
            temperature=temp, epsilon=self.args.imp_smpl_epsilon, ss_prob=ss_prob,
            decoder_inputs_embeds=sample["target_embeddings"]
        )

        # if decoding_style == "gumbel":
        #     return self.wrap_for_output(sample, t5out.logits, input_onehot=t5out.input_onehot, output_onehot=t5out.output_onehot, target_onehot=t5out.target_onehot)
        return self.wrap_for_output(
            sample, t5out.logits, modified_logits=t5out.modified_logits, output_tokens=t5out.output_tokens,
            input_onehot=t5out.input_onehot, output_onehot=t5out.output_onehot, target_onehot=t5out.target_onehot
        )

    def compute_discriminator_score(self, generator_input: Dict, generator_output: Dict):
        return 0.0

    def compute_pg_loss(self, generator_input: Dict, generator_output: Dict, reward):
        """
        For this generator, generator_output["logits"] stores embeddings.
        """
        return torch.norm(generator_output["logits"] - generator_input["target_embeddings"], dim=-1, p=1).mean()

    def compute_evaluation_loss(self, generator_input, generator_output):
        with torch.no_grad():
            reward = self.compute_discriminator_score(generator_input, generator_output)

        loss = self.compute_pg_loss(generator_input, generator_output, reward)
        return loss

    def wrap_for_output(self, sample, logits, modified_logits=None, output_tokens=None, input_onehot=None, output_onehot=None, target_onehot=None):
        if input_onehot is not None: # add zeros to use indexing from 1
            zeros = torch.zeros((input_onehot.shape[0], input_onehot.shape[1], 1)).to(input_onehot.device)
            input_onehot = torch.cat([zeros, input_onehot], dim=2)
            zeros = torch.zeros((output_onehot.shape[0], output_onehot.shape[1], 1)).to(output_onehot.device)
            output_onehot = torch.cat([zeros, output_onehot], dim=2)
            zeros = torch.zeros((target_onehot.shape[0], target_onehot.shape[1], 1)).to(target_onehot.device)
            target_onehot = torch.cat([zeros, target_onehot], dim=2)

        pred_bert_token_ids = self.embs2bert_tokens_exact(logits)
        target_bert_token_ids = self.targets_to_bert_ids(sample["target"])
        target_bert_mask = target_bert_token_ids != 0

        output = {
            "logits": logits,
            "target": target_bert_token_ids,
            "mask": target_bert_mask,
            "prediction": pred_bert_token_ids,
            "input_onehot": input_onehot,
            "output_onehot": output_onehot,
            "target_onehot": target_onehot,
            "modified_logits": modified_logits,
        }

        return output

    def get_target_embedder(self):
        return self.discriminator.bleurt_model.bert.embeddings.word_embeddings

    def format_sample(self, sample, extra_tokens=20):
        sample = super().format_sample(sample, extra_tokens=extra_tokens)

        with torch.no_grad():
            sample["target_bert_token_ids"] = self.targets_to_bert_ids(sample["target"])
            sample["target_embeddings"] = self.get_target_embedder()(sample["target_bert_token_ids"])
        return sample

    # def embs2bert_tokens_hnsw(self, predicted_embs):
    #     prediction_embs_np = predicted_embs.cpu().detach().numpy()
    #     # flattening across batch to avoid loops
    #     prediction_embs_np_glued = prediction_embs_np.reshape(-1, predicted_embs.shape[-1])
    #     nnbrs, dist = self.hnsw_graph.knn_query(data=prediction_embs_np_glued, k=1)
    #     nnbrs = np.squeeze(nnbrs, axis=-1)
    #     # reshaping back
    #     nnbrs = nnbrs.reshape((prediction_embs_np.shape[0], predicted_embs.shape[1]))
    #     return torch.from_numpy(nnbrs.astype(np.int64))

    def embs2bert_tokens_exact(self, predicted_embs):
        prediction_embs_np = predicted_embs.cpu().detach().numpy()
        # flattening across batch to avoid loops
        prediction_embs_np_glued = prediction_embs_np.reshape(-1, predicted_embs.shape[-1])
        dist, nnbrs = self.faiss_index.search(normalize(prediction_embs_np_glued), 1)
        nnbrs = np.squeeze(nnbrs, axis=-1)
        # reshaping back
        nnbrs = nnbrs.reshape((prediction_embs_np.shape[0], predicted_embs.shape[1]))
        return torch.from_numpy(nnbrs)

    def crop_after_bert_sep(self, sentences_tokens):
        cropped_sent_tokens = []
        for sentence_tokens in sentences_tokens:
            sep_locations = (sentence_tokens == 102).nonzero(as_tuple=False)
            eos_idx = sep_locations[0] if len(sep_locations) > 0 else None
            if eos_idx is not None:
                sentence_tokens = sentence_tokens[:eos_idx]
            cropped_sent_tokens.append(sentence_tokens)
        return cropped_sent_tokens

    def decode_sentences(self, batch, for_referece=False):
        decoded = self.discriminator.tokenizer.batch_decode(self.crop_after_bert_sep(batch), skip_special_tokens=True)
        sentences = []
        for s in decoded:
            if for_referece:
                s = [s]
            sentences.append(s)
        return sentences


class SeqT5Gumbel(SeqT5RL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_strategy = "alternate"  # alternate | mle | rl
        self.sequential_decoding_style = "gumbel"

    def create_discriminator(self, args):
        # self.discriminator = T5Discriminator(args, self.dataset.src_dict, self.dataset.dst_dict,
        #                                      use_cuda=self.use_cuda)
        if self.args.d_ckpt_path is not None:
            print(f"Loading pretrained discriminator from checkpoint {self.args.d_ckpt_path}")
            self.discriminator = torch.load(self.args.d_ckpt_path)
        else:
            self.discriminator = T5SemanticDiscriminator()

    def mle_step(self, sample, batch_i, epoch, loader_len, seq_decoding=False):

        if seq_decoding:
            print("Seq MLE Training")
            output = self.sequential_generation(sample, decoding_style=self.sequential_decoding_style, top_k=0,
                                                top_p=0.6)
        else:
            print("MLE Training")
            output = self.teacher_forcing_generation(sample)

        loss = output["loss"]
        # sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']

        with torch.no_grad():
            if (batch_i + (epoch - 1) * loader_len) % 20 == 0:
                self.evaluate_generator(
                    sample["net_input"]["src_tokens"], output["prediction"], sample['target'], output["mask"], loss,
                    sample['ntokens'], batch_i=batch_i, epoch_i=epoch, num_batches=loader_len, partition="train",
                    strategy="mle"
                )

        self.g_optimizer.zero_grad()
        loss.backward()
        # all-reduce grads and rescale by grad_denom
        # for p in self.generator.parameters():
        #     if p.requires_grad:
        #         p.grad.data.div_(sample_size)
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.args.clip_norm)
        self.g_optimizer.step()

    # def create_discriminator(self, args):
    #     self.discriminator = GumbelDiscriminator(args, self.dataset.src_dict, self.dataset.dst_dict,
    #                                           use_cuda=self.use_cuda)
    #     print("Discriminator loaded successfully!")

    # def pg_step(self, sample, batch_i, epoch, loader_len):
    #     print("Policy Gradient Training")
    #
    #     output = self.sequential_generation(sample, decoding_style=self.sequential_decoding_style)
    #
    #     reward = self.discriminator(output['input_onehot'], output["output_onehot"])
    #
    #     loss = self.d_criterion(reward, torch.ones_like(reward))
    #
    #     with torch.no_grad():
    #         if (batch_i + (epoch - 1) * loader_len) % 1 == 0:
    #             self.evaluate_generator(
    #                 sample["net_input"]["src_tokens"], output["prediction"], sample['target'], output["mask"], loss,
    #                 sample['ntokens'], batch_i=batch_i, epoch_i=epoch, num_batches=loader_len, partition="train", strategy="rl"
    #             )
    #
    #     self.g_optimizer.zero_grad()
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.args.clip_norm)
    #     self.g_optimizer.step()

    # def discrimnator_loss_acc(self, sample):
    #     bsz = sample['target'].size(0)  # batch_size = 64
    #     src_sentence = sample['net_input']['src_tokens']  # 64 x max-len i.e 64 X 50
    #
    #     # now train with machine translation output i.e generator output
    #     true_labels = Variable(torch.ones(sample['target'].size(0)).float()).unsqueeze(1).repeat(1, sample[
    #         'target'].size(1))
    #     # true_labels = Variable(torch.ones(sample['target'].size(0)).float())
    #
    #     if self.use_cuda:
    #         true_labels = true_labels.cuda()
    #
    #     with torch.no_grad():
    #         gen_output = self.sequential_generation(sample,
    #                                                 decoding_style=self.sequential_decoding_style)  # 64 X 50 X 6632
    #
    #     true_sentence = gen_output["target_onehot"]
    #     fake_sentence = gen_output["output_onehot"]
    #     src_sentence = gen_output["input_onehot"]
    #
    #     fake_labels = Variable(torch.zeros(sample['target'].size(0)).float()).unsqueeze(1).repeat(1, sample[
    #         'target'].size(1))
    #     # fake_labels = Variable(torch.zeros(sample['target'].size(0)).float())
    #
    #     if self.use_cuda:
    #         fake_labels = fake_labels.cuda()
    #
    #     disc_out_neg = self.discriminator(fake_sentence, fake_sentence)
    #     disc_out_pos = self.discriminator(true_sentence, true_sentence)
    #     disc_out = torch.cat([disc_out_neg.squeeze(1), disc_out_pos.squeeze(1)], dim=0)
    #
    #     labels = torch.cat([fake_labels, true_labels], dim=0)
    #
    #     d_loss = self.d_criterion(disc_out, labels)
    #
    #     acc = torch.sum(torch.round(disc_out) == labels).float() / torch.numel(labels) * 100
    #     return d_loss, acc

    # def sequential_generation(self, sample, decoding_style="rl", top_k=0, top_p=0.6, temp=.2):
    #     t5out = self.generator(
    #         self.transform_for_t5(sample['net_input']['src_tokens']), attention_mask=sample["attention_mask"],
    #         labels=self.transform_for_t5(sample['target']), decoding_style=decoding_style, top_k=top_k, top_p=top_p,
    #         temperature=temp, epsilon=self.args.imp_smpl_epsilon
    #     )
    #
    #     return self.wrap_for_output(sample, t5out.logits, input_onehot=t5out.input_onehot, output_onehot=t5out.output_onehot, target_onehot=t5out.target_onehot)

    # def handicap_discriminator(self):
    #     pass
