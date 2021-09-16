import logging
import math
import random
from copy import copy

import numpy as np
from torch import cuda
from torch.autograd import Variable

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

    def create_losses(self):
        # define loss function
        super(SeqT5Trainer, self).create_losses()
        self.pg_criterion = lambda pred, true, reward, modified_logits, predicted_tokens: \
            self._pg_criterion(
                self._logsoftmax(pred),
                self.transform_for_t5(true),
                reward,
                self._logsoftmax(modified_logits) if modified_logits is not None else None,
                self.transform_for_t5(predicted_tokens) if predicted_tokens is not None else None
            )

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

        output["loss"] = self.g_criterion(
            output["logits"][output["mask"], :],
            self.transform_for_t5(output["target"])[output["mask"]]
        )
        return output

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

    def train_loop(self, trainloader, epoch_i, num_update):
        for i, sample in enumerate(trainloader):

            sample = self.format_sample(sample)

            if self.use_cuda:
                # wrap input tensors in cuda tensors
                sample = utils.make_variable(sample, cuda=cuda, gpu_id=f'cuda:{self.args.gpuid[0]}')

            if self.args.reduce_tf_frac:
                mle_frac = max(self.args.epochs - epoch_i, 1) / self.args.epochs
            else:
                mle_frac = 0.5

            if epoch_i > self.args.discriminator_pretraining or not hasattr(self, "discriminator"):
                if hasattr(self, "discriminator"):
                    if self.training_strategy == "alternate":
                        if random.random() <= mle_frac:
                            self.mle_step(sample, i, epoch_i, len(trainloader))
                        else:
                            # if random.random() > 0.5:
                            #     self.mle_step(sample, i, epoch_i, len(trainloader), seq_decoding=True)
                            # else:
                            self.pg_step(sample, i, epoch_i, len(trainloader))
                    elif self.training_strategy == "mle":
                        self.mle_step(sample, i, epoch_i, len(trainloader))
                    elif self.training_strategy == "rl":
                        self.pg_step(sample, i, epoch_i, len(trainloader))
                    else:
                        raise ValueError(
                            f"Invalid training strategy: {self.training_strategy}. Valid options are: alternate|mle|rl.")
                else:
                    self.mle_step(sample, i, epoch_i, len(trainloader))
                num_update += 1
            else:
                if i == 0 and epoch_i == 1:
                    print(f"Pretraining discriminator for {self.args.discriminator_pretraining} epochs")

        return num_update

    def eval_loop(self, valloader, epoch_i, force=False):
        for i, sample in enumerate(valloader):

            sample = self.format_sample(sample, extra_tokens=50)

            with torch.no_grad():
                if self.use_cuda:
                    # wrap input tensors in cuda tensors
                    sample = utils.make_variable(sample, cuda=cuda, gpu_id=f'cuda:{self.args.gpuid[0]}')

                if epoch_i > self.args.discriminator_pretraining or not hasattr(self, "discriminator") or force is True:
                    # generator validation
                    output = self.eval_generation(sample)
                    self.evaluate_generator(
                        sample["net_input"]["src_tokens"], output["prediction"], output["target"], output["mask"], output["loss"], ntokens=sample["ntokens"],
                        batch_i=i, epoch_i=epoch_i, num_batches=len(valloader), partition="valid", strategy="mle", accumulate=i<len(valloader)-1, write_sents=True
                    )

    def pg_step(self, sample, batch_i, epoch, loader_len):
        # print("Policy Gradient Training")

        output = self.sequential_generation(sample, decoding_style=self.sequential_decoding_style, top_k=0, top_p=0.6)

        with torch.no_grad():
            reward = self.discriminator(output["prediction"], sample["target"]) # dim (bsize x 1)
            reward = reward.cuda(f'cuda:{self.args.gpuid[0]}')

        pg_loss = self.pg_criterion(output["logits"], sample['target'], reward, output.get("modified_logits", None), output.get("prediction", None))# + \
        # self.pg_criterion(output["logits"], sample['target'], gen_reward, output.get("modified_logits", None),
        #                   output.get("prediction", None))

        with torch.no_grad():
            if (batch_i + (epoch - 1) * loader_len) % self.args.train_bleu_every == 0:
                self.evaluate_generator(
                    sample["net_input"]["src_tokens"], output["prediction"], sample['target'], output["mask"], pg_loss,
                    sample['ntokens'], batch_i=batch_i, epoch_i=epoch, num_batches=loader_len, partition="train", strategy="rl"
                )

        self.g_optimizer.zero_grad()
        pg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.args.clip_norm)
        self.g_optimizer.step()

    def evaluate_generator(
            self, original, predictions, targets, target_mask, loss, ntokens, batch_i, epoch_i, num_batches, partition=None,
            strategy=None, accumulate=False, write_sents=False
    ):

        assert partition in {"train", "valid", "test"}
        assert strategy in {"mle", "rl", "gumbel"}

        sample_size = targets.size(0) if self.args.sentence_avg else ntokens

        if hasattr(self, "discriminator"):
            with torch.no_grad():
                discr_score_neg = self.discriminator(predictions, targets).mean()
                discr_score_pos = self.discriminator(targets, targets).mean()
        else:
            discr_score_neg = 0.
            discr_score_pos = 0.

        gen_acc = self.token_accuracy(predictions, targets, target_mask)
        bleu = self.compute_bleu(predictions, targets, accumulate=accumulate)
        rouge = self.compute_rouge(predictions, original, accumulate=accumulate)

        self.g_logging_meters[f'{partition}_loss'].update(loss, sample_size)
        self.g_logging_meters[f'{partition}_acc'].update(gen_acc)
        # self.d_logging_meters[f'{partition}_bleu'].update(bleu)
        # self.d_logging_meters[f'{partition}_rouge'].update(rouge)

        logging.debug(f"G loss {self.g_logging_meters[f'{partition}_loss'].avg:.3f}, "
                      f"G acc {self.g_logging_meters[f'{partition}_acc'].avg:.3f} at batch {batch_i}")

        if not hasattr(self, "last_sents"):
            self.last_sents = []
        for sent in self.decode_sentences(predictions):
            if len(self.last_sents) >= self.args.gen_sents_in_tb:
                self.last_sents.pop(0)
            self.last_sents.append(sent)

        if not accumulate:
            self.write_summary({
                f"Loss/{partition}/{strategy}/gen": loss,
                f"Accuracy/{partition}/gen": gen_acc,
                f"bleu/{partition}/score": bleu["score"],
                f"bleu/{partition}/P1": bleu["precisions"][0],
                f"bleu/{partition}/P2": bleu["precisions"][1],
                f"bleu/{partition}/P3": bleu["precisions"][2],
                f"bleu/{partition}/P4": bleu["precisions"][3],
                f"rouge/{partition}/rouge1/high/f1": rouge["rouge1"].high.fmeasure,
                f"rouge/{partition}/rouge2/high/f1": rouge["rouge2"].high.fmeasure,
                f"rouge/{partition}/rougeL/high/f1": rouge["rougeL"].high.fmeasure,
                f"rouge/{partition}/rouge1/high/P": rouge["rouge1"].high.precision,
                f"rouge/{partition}/rouge2/high/P": rouge["rouge2"].high.precision,
                f"rouge/{partition}/rougeL/high/P": rouge["rougeL"].high.precision,
                f"discr_score/{partition}/negative": discr_score_neg,
                f"discr_score/{partition}/positive": discr_score_pos,
            }, batch_i + (epoch_i - 1) * num_batches, write_sents=write_sents)

    def create_models(self, args):
        self.create_generator(args)
        self.create_discriminator(args)

        if self.use_cuda:
            # if torch.cuda.device_count() > 1:
            #     self.discriminator = torch.nn.DataParallel(self.discriminator).cuda()
            #     self.generator = torch.nn.DataParallel(self.generator).cuda()
            # else:
            # self.generator.cuda()
            self.generator.cuda(f'cuda:{self.args.gpuid[0]}')  # manually placing generator to the specified gpu
            if hasattr(self, "discriminator"):
                self.discriminator.cuda(f'cuda:{self.args.gpuid[0]}')  #.cuda(f'cuda:{cuda.device_count() - 1 - self.args.gpuid[0]}')  # selects other gpu
        else:
            if hasattr(self, "discriminator"):
                self.discriminator.cpu()
            self.generator.cpu()

    def save_models(self, epoch_i):
        self.save_generator(os.path.join(self.checkpoints_path, f"joint_{self.g_logging_meters['valid_loss'].avg:.3f}.epoch_{epoch_i}_gen.pt"))
        with open(os.path.join(self.checkpoints_path, "params.json"), "w") as paramsink:
            paramsink.write(json.dumps(self.args.__dict__, indent=4))


class SeqEmbT5Bleurt(SeqT5Bleurt):
    def __init__(self, *args, **kwargs):
        super(SeqEmbT5Bleurt, self).__init__(*args, **kwargs)
        bert_embeddings = self.discriminator.bleurt_model.bert.embeddings.word_embeddings.weight
        bert_embeddings = bert_embeddings.cpu().detach().numpy()
        from sklearn.preprocessing import normalize
        self.normalize_l2 = normalize
        import faiss
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
            self.generator = SeqEmbT5.from_pretrained('t5-small')
        if self.args.freeze_encoder:
            self.generator.encoder.requires_grad = False

    def create_discriminator(self, args):
        if self.args.d_ckpt_path is not None:
            print(f"Loading pretrained discriminator from checkpoint {self.args.d_ckpt_path}")
            self.discriminator = torch.load(self.args.d_ckpt_path)
        else:
            self.discriminator = BleurtEmbDiscriminator(decode_fn = lambda pred: self.decode_sentences(pred))

    def train_loop(self, trainloader, epoch_i, num_update):
        for i, sample in enumerate(trainloader):

            sample = self.format_sample(sample)

            if self.use_cuda:
                # wrap input tensors in cuda tensors
                sample = utils.make_variable(sample, cuda=cuda)

            if self.args.reduce_tf_frac:
                mle_frac = max(self.args.epochs - epoch_i, 1) / self.args.epochs
            else:
                mle_frac = 0.5

            if epoch_i > self.args.discriminator_pretraining or not hasattr(self, "discriminator"):
                if hasattr(self, "discriminator"):
                    self.pg_step(sample, i, epoch_i, len(trainloader))
                else:
                    self.mle_step(sample, i, epoch_i, len(trainloader))
                num_update += 1
            else:
                if i == 0 and epoch_i == 1:
                    print(f"Pretraining discriminator for {self.args.discriminator_pretraining} epochs")

        return num_update

    def targets_to_bert_ids(self, targets):
        targets_decoded = self.discriminator.decode_fn(targets) # targets in english
        targets_bert_tokens_np = np.zeros(targets.shape, dtype=np.int64)
        for i, t in enumerate(targets_decoded):
            bert_token_ids = self.discriminator.tokenizer.convert_tokens_to_ids(self.discriminator.tokenizer.tokenize(t))
            proper_seq_len = min(len(bert_token_ids), targets.shape[1])
            targets_bert_tokens_np[i, :proper_seq_len] = bert_token_ids[:proper_seq_len]
        target_bert_tokens = torch.from_numpy(targets_bert_tokens_np)
        target_bert_tokens = target_bert_tokens.cuda()
        return target_bert_tokens

    def pg_step(self, sample, batch_i, epoch, loader_len):
        # print("Policy Gradient Training")

        output = self.sequential_generation(sample, decoding_style=self.sequential_decoding_style, top_k=0, top_p=0.6)

        reward = self.discriminator(output["logits"], sample["target"]) # dim (bsize x 1)
        reward = reward.cuda(f'cuda:{self.args.gpuid[0]}')

        pg_loss = -reward.mean()  # self.pg_criterion(output["logits"], sample['target'], reward, output.get("modified_logits", None), output.get("prediction", None))# + \
        # self.pg_criterion(output["logits"], sample['target'], gen_reward, output.get("modified_logits", None),
        #                   output.get("prediction", None))
        with torch.no_grad():
            target_bert_tokens = self.targets_to_bert_ids(sample["target"])
            target_bert_embeddings = self.discriminator.bleurt_model.bert.embeddings.word_embeddings(target_bert_tokens)

        l2_emb_loss = torch.norm(output["logits"] - target_bert_embeddings, dim=-1).mean()

        total_loss = pg_loss + l2_emb_loss

        with torch.no_grad():
            if (batch_i + (epoch - 1) * loader_len) % self.args.train_bleu_every == 0:
                self.evaluate_generator(
                    sample["net_input"]["src_tokens"], output["logits"], sample['target'], output["mask"], pg_loss,
                    sample['ntokens'], batch_i=batch_i, epoch_i=epoch, num_batches=loader_len, partition="train", strategy="rl"
                )
                self.evaluate_generator(
                    sample["net_input"]["src_tokens"], output["logits"], sample['target'], output["mask"], l2_emb_loss,
                    sample['ntokens'], batch_i=batch_i, epoch_i=epoch, num_batches=loader_len, partition="train", strategy="l2_emb"
                )

        self.g_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.args.clip_norm)
        self.g_optimizer.step()

    def eval_loop(self, valloader, epoch_i, force=False):
        for i, sample in enumerate(valloader):

            sample = self.format_sample(sample, extra_tokens=50)

            with torch.no_grad():
                if self.use_cuda:
                    # wrap input tensors in cuda tensors
                    sample = utils.make_variable(sample, cuda=cuda, gpu_id=f'cuda:{self.args.gpuid[0]}')

                if epoch_i > self.args.discriminator_pretraining or not hasattr(self, "discriminator") or force is True:
                    # generator validation
                    output = self.eval_generation(sample)

                    with torch.no_grad():
                        reward = self.discriminator(output["logits"], sample["target"]) # dim (bsize x 1)
                        reward = reward.cuda(f'cuda:{self.args.gpuid[0]}')
                        output["loss"] = -reward.mean()

                    self.evaluate_generator(
                        sample["net_input"]["src_tokens"], output["logits"], output["target"], output["mask"], output["loss"], ntokens=sample["ntokens"],
                        batch_i=i, epoch_i=epoch_i, num_batches=len(valloader), partition="valid", strategy="mle", accumulate=i<len(valloader)-1, write_sents=True
                    )

    def evaluate_generator(
            self, original, predictions, targets, target_mask, loss, ntokens, batch_i, epoch_i, num_batches, partition=None,
            strategy=None, accumulate=False, write_sents=False
    ):

        assert partition in {"train", "valid", "test"}
        assert strategy in {"mle", "rl", "gumbel", "l2_emb"}

        sample_size = targets.size(0) if self.args.sentence_avg else ntokens

        # if hasattr(self, "discriminator"):
        #     with torch.no_grad():
        #         discr_score_neg = self.discriminator(predictions, targets).mean()
        #         discr_score_pos = self.discriminator(targets, targets).mean()
        # else:
        discr_score_neg = 0.
        discr_score_pos = 0.

        import time
        pred_bert_tokens = self.embs2bert_tokens_exact(predictions).cuda()
        target_bert_tokens = self.targets_to_bert_ids(targets)
        # validation code, checks if nn search works fine
        # target_embs = self.discriminator.bleurt_model.bert.embeddings.word_embeddings(target_bert_tokens)
        # target_bert_from_embs = self.embs2bert_tokens_exact(target_embs).cuda()
        # equivalent_tokens = target_bert_from_embs == target_bert_tokens
        # print(equivalent_tokens)
        # print("number of mismatches for every sent:", (equivalent_tokens == False).sum(dim=1), "on avg:", (equivalent_tokens == False).sum()/10)
        # for k in range(10):
        #     print(self.discriminator.tokenizer.convert_ids_to_tokens(target_bert_tokens[k]))
        #     print(self.discriminator.tokenizer.convert_ids_to_tokens(target_bert_from_embs[k]))
        target_bert_mask = target_bert_tokens != 0
        gen_acc = self.token_accuracy(pred_bert_tokens, target_bert_tokens, target_bert_mask)
        bleu = self.compute_bleu(pred_bert_tokens, target_bert_tokens, accumulate=accumulate)
        # rouge = self.compute_rouge(pred_bert_tokens, target_bert_tokens, accumulate=accumulate)

        self.g_logging_meters[f'{partition}_loss'].update(loss, sample_size)
        self.g_logging_meters[f'{partition}_acc'].update(gen_acc)
        # self.d_logging_meters[f'{partition}_bleu'].update(bleu)
        # self.d_logging_meters[f'{partition}_rouge'].update(rouge)

        logging.debug(f"G loss {self.g_logging_meters[f'{partition}_loss'].avg:.3f}, "
                      f"G acc {self.g_logging_meters[f'{partition}_acc'].avg:.3f} at batch {batch_i}")

        if not hasattr(self, "last_sents"):
            self.last_sents = []
        for sent in self.discriminator.tokenizer.batch_decode(pred_bert_tokens, skip_special_tokens=True):
            if len(self.last_sents) >= self.args.gen_sents_in_tb:
                self.last_sents.pop(0)
            self.last_sents.append(sent)

        if not accumulate:
            self.write_summary({
                f"Loss/{partition}/{strategy}/gen": loss,
                f"Accuracy/{partition}/gen": gen_acc,
                f"bleu/{partition}/score": bleu["score"],
                f"bleu/{partition}/P1": bleu["precisions"][0],
                f"bleu/{partition}/P2": bleu["precisions"][1],
                f"bleu/{partition}/P3": bleu["precisions"][2],
                f"bleu/{partition}/P4": bleu["precisions"][3],
                f"rouge/{partition}/rouge1/high/f1": 0.,  # rouge["rouge1"].high.fmeasure,
                f"rouge/{partition}/rouge2/high/f1": 0.,  # rouge["rouge2"].high.fmeasure,
                f"rouge/{partition}/rougeL/high/f1": 0.,  # rouge["rougeL"].high.fmeasure,
                f"rouge/{partition}/rouge1/high/P": 0.,  # rouge["rouge1"].high.precision,
                f"rouge/{partition}/rouge2/high/P": 0.,  # rouge["rouge2"].high.precision,
                f"rouge/{partition}/rougeL/high/P": 0.,  # rouge["rougeL"].high.precision,
                f"discr_score/{partition}/negative": discr_score_neg,
                f"discr_score/{partition}/positive": discr_score_pos,
            }, batch_i + (epoch_i - 1) * num_batches, write_sents=write_sents)

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

        output["loss"] = 1. #self.g_criterion(
        # output["logits"][output["mask"], :],
        # self.transform_for_t5(output["target"])[output["mask"]]
        # )
        return output

    def train(self):
        args = self.args

        # start joint training
        best_dev_loss = math.inf
        num_update = 0

        if self.args.start_epoch == 1:
            self.validate(args, epoch_i=0, force=True)

        # main training loop
        for epoch_i in range(self.args.start_epoch, args.epochs + 1):
            logging.info(f"At {epoch_i}-th epoch.")

            seed = args.seed + epoch_i
            torch.manual_seed(seed)

            max_positions_train = (args.fixed_max_len, args.fixed_max_len)

            # Initialize dataloader, starting at batch_offset
            trainloader = self.dataset.train_dataloader(
                'train',
                max_tokens=args.max_tokens,
                max_sentences=args.joint_batch_size,
                max_positions=max_positions_train,
                seed=seed,
                epoch=epoch_i,
                sample_without_replacement=args.sample_without_replacement,
                sort_by_source_size=(epoch_i <= args.curriculum),
                shard_id=args.distributed_rank,
                num_shards=args.distributed_world_size,
            )

            # reset meters
            for key, val in self.g_logging_meters.items():
                if val is not None:
                    val.reset()
            for key, val in self.d_logging_meters.items():
                if val is not None:
                    val.reset()

            # set training mode
            self.generator.train()
            # if hasattr(self, "discriminator"):
                # self.discriminator.train()
            # update_learning_rate(num_update, 8e4, args.g_learning_rate, args.lr_shrink, self.g_optimizer)

            print(f"Training batches: {len(trainloader)}")

            num_update = self.train_loop(trainloader, epoch_i, num_update)

            self.validate(args, epoch_i)

            self.save_models(epoch_i)

            if self.g_logging_meters['valid_loss'].avg < best_dev_loss:
                best_dev_loss = self.g_logging_meters['valid_loss'].avg
                self.save_generator(os.path.join(self.checkpoints_path, "best_gmodel.pt"))

    def format_sample(self, sample, extra_tokens=20):
        sample = copy(sample)

        max_src_len = min(sample["net_input"]['src_tokens'].size(1), max(sample["net_input"]['src_lengths'].tolist()))
        max_trg_len = min(sample["target"].size(1), max(sample["target_lengths"].tolist()) + extra_tokens)

        sample["net_input"]['src_tokens'] = sample["net_input"]['src_tokens'][:, :max_src_len].contiguous()
        sample["target"] = sample["target"][:, :max_trg_len].contiguous()
        sample["attention_mask"] = self.get_length_mask(sample["net_input"]["src_tokens"], sample["net_input"]['src_lengths'])
        return sample

    def embs2bert_tokens_hnsw(self, prediction_embs):
        prediction_embs_np = prediction_embs.cpu().detach().numpy()
        # flattening across batch to avoid loops
        prediction_embs_np_glued = prediction_embs_np.reshape(-1, prediction_embs.shape[-1])
        nnbrs, dist = self.hnsw_graph.knn_query(data=prediction_embs_np_glued, k=1)
        nnbrs = np.squeeze(nnbrs, axis=-1)
        # reshaping back
        nnbrs = nnbrs.reshape((prediction_embs_np.shape[0], prediction_embs.shape[1]))
        return torch.from_numpy(nnbrs.astype(np.int64))

    def embs2bert_tokens_exact(self, prediction_embs):
        prediction_embs_np = prediction_embs.cpu().detach().numpy()
        # flattening across batch to avoid loops
        prediction_embs_np_glued = prediction_embs_np.reshape(-1, prediction_embs.shape[-1])
        # nnbrs = self.nbrs.kneighbors(prediction_embs_np_glued, return_distance=False)
        dist, nnbrs = self.faiss_index.search(self.normalize_l2(prediction_embs_np_glued), 1)
        nnbrs = np.squeeze(nnbrs, axis=-1)
        # reshaping back
        nnbrs = nnbrs.reshape((prediction_embs_np.shape[0], prediction_embs.shape[1]))
        return torch.from_numpy(nnbrs)

    def compute_bleu(self, predictions, references, accumulate=False):
        predictions_decoded = self.discriminator.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        references_decoded = self.discriminator.tokenizer.batch_decode(references, skip_special_tokens=True)
        references_decoded = [[ref] for ref in references_decoded]
        self.bleu_metric.add_batch(
            predictions=predictions_decoded,
            references=references_decoded
        )
        if not accumulate:
            bleu = self.bleu_metric.compute(
                predictions=predictions_decoded,
                references=references_decoded
            )
        else:
            bleu = None
        return bleu


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
