import json
import math
from abc import abstractmethod
from copy import copy
from datetime import datetime
import logging
import dill
import os

from torch.utils.tensorboard import SummaryWriter

import random
import numpy as np
from collections import OrderedDict

import torch
from torch import cuda
from torch.autograd import Variable

import data
import utils
from meters import AverageMeter
from discriminator import Discriminator, AttDiscriminator
from generator import LSTMModel, VarLSTMModel
# from train_generator import train_g
# from train_discriminator import train_d
from PGLoss import PGLoss


class ModelTrainer:
    def __init__(self, args):
        # # Set model parameters
        # args.encoder_embed_dim = 128
        # args.encoder_layers = 2  # 4
        # args.encoder_dropout_out = 0
        # args.decoder_embed_dim = 128
        # args.decoder_layers = 2  # 4
        # args.decoder_out_embed_dim = 128
        # args.decoder_dropout_out = 0
        # args.bidirectional = False

        self.args = args

        self.set_gpu(args)
        self.load_dataset(args)
        self.create_meters()
        self.create_models(args)
        self.create_output_path(args)
        self.create_losses()
        self.handicap_discriminator()
        self.create_optimizers(args)
        self.summary_writer = SummaryWriter(self.checkpoints_path)

        import datasets
        self.bleu_metric = datasets.load_metric('sacrebleu')
        self.rouge_metric = datasets.load_metric('rouge')
        self.training_strategy = "alternate"  # alternate | mle | rl
        self.sequential_decoding_style = "rl"

    def set_gpu(self, args):
        # args.gpuid = ""  # TODO disable cuda
        if args.gpuid[0] == -1:
            self.use_cuda = False
        else:
            torch.cuda.set_device(args.gpuid[0])
            self.use_cuda = True
        # self.use_cuda = (len(args.gpuid) >= 1)

        print("{0} GPU(s) are available".format(cuda.device_count()))
        print("Using GPU {}".format(args.gpuid[0]))

    def load_dataset(self, args):
        # Load dataset
        splits = ['train', 'valid']
        if data.has_binary_files(args.data, splits):
            dataset = data.load_dataset(
                args.data, splits, args.src_lang, args.trg_lang, args.fixed_max_len)
        else:
            dataset = data.load_raw_text_dataset(
                args.data, splits, args.src_lang, args.trg_lang, args.fixed_max_len)
        if args.src_lang is None or args.trg_lang is None:
            # record inferred languages in args, so that it's saved in checkpoints
            args.src_lang, args.trg_lang = dataset.src, dataset.dst

        print('| [{}] dictionary: {} types'.format(dataset.src, len(dataset.src_dict)))
        print('| [{}] dictionary: {} types'.format(dataset.dst, len(dataset.dst_dict)))

        for split in splits:
            print('| {} {} {} examples'.format(args.data, split, len(dataset.splits[split])))

        self.dataset = dataset

    def create_meters(self):
        g_logging_meters = OrderedDict()
        g_logging_meters['train_loss'] = AverageMeter()
        g_logging_meters['valid_loss'] = AverageMeter()
        g_logging_meters['train_acc'] = AverageMeter()
        g_logging_meters['valid_acc'] = AverageMeter()
        g_logging_meters['bsz'] = AverageMeter()  # sentences per batch

        d_logging_meters = OrderedDict()
        d_logging_meters['train_loss'] = AverageMeter()
        d_logging_meters['valid_loss'] = AverageMeter()
        d_logging_meters['train_acc'] = AverageMeter()
        d_logging_meters['valid_acc'] = AverageMeter()
        d_logging_meters['train_bleu'] = AverageMeter()
        d_logging_meters['valid_bleu'] = AverageMeter()
        d_logging_meters['train_rouge'] = AverageMeter()
        d_logging_meters['valid_rouge'] = AverageMeter()
        d_logging_meters['bsz'] = AverageMeter()  # sentences per batch

        self.g_logging_meters = g_logging_meters
        self.d_logging_meters = d_logging_meters

    @abstractmethod
    def create_generator(self, args):
        raise NotImplementedError()
        # self.generator = LSTMModel(args, self.dataset.src_dict, self.dataset.dst_dict, use_cuda=self.use_cuda)
        # self.generator = VarLSTMModel(args, self.dataset.src_dict, self.dataset.dst_dict, use_cuda=self.use_cuda)
        # print("Generator loaded successfully!")

    @abstractmethod
    def create_discriminator(self, args):
        raise NotImplementedError()
        # discriminator = Discriminator(args, dataset.src_dict, dataset.dst_dict, use_cuda=use_cuda)
        # self.discriminator = AttDiscriminator(args, self.dataset.src_dict, self.dataset.dst_dict, use_cuda=self.use_cuda)
        # print("Discriminator loaded successfully!")

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

    def create_output_path(self, args):
        # adversarial training checkpoints saving path
        if args.note is None:
            name = self.__class__.__name__ + " " + str(datetime.now())
        else:
            name = self.__class__.__name__ + " " + args.note
        path = os.path.join(args.model_file, name.replace(" ","_").replace(":","-"))
        if not os.path.exists(path):
            os.makedirs(path)
        self.checkpoints_path = path

    def create_losses(self):
        # define loss function
        self._g_criterion = torch.nn.NLLLoss(reduction='mean')
        self.d_criterion = torch.nn.BCELoss()  #torch.nn.SoftMarginLoss() #
        self._pg_criterion = PGLoss(ignore_index=self.dataset.dst_dict.pad(), size_average=True, reduce=True)
        self._logsoftmax = torch.nn.LogSoftmax(dim=-1)

        self.g_criterion = lambda pred, true: self._g_criterion(self._logsoftmax(pred), true)
        self.pg_criterion = lambda pred, true, reward, modified_logits, predicted_tokens: \
            self._pg_criterion(
                self._logsoftmax(pred),
                true,
                reward,
                self._logsoftmax(modified_logits) if modified_logits is not None else None,
                predicted_tokens,
            )

    def handicap_discriminator(self):
        pass
        # # fix discriminator word embedding (as Wu et al. do)
        # if hasattr(self, "discriminator"):
        #     for p in self.discriminator.embed_src_tokens.parameters():
        #         p.requires_grad = False
        #     for p in self.discriminator.embed_trg_tokens.parameters():
        #         p.requires_grad = False

    def create_optimizers(self, args):
        # define optimizer
        self.g_optimizer = eval("torch.optim." + args.g_optimizer)(filter(lambda x: x.requires_grad,
                                                                     self.generator.parameters()),
                                                              args.g_learning_rate)

        if hasattr(self, "discriminator"):
            self.d_optimizer = eval("torch.optim." + args.d_optimizer)(filter(lambda x: x.requires_grad,
                                                                         self.discriminator.parameters()),
                                                                  args.d_learning_rate,)
                                                                  # momentum=args.momentum,
                                                                  # nesterov=True)

    def write_summary(self, scores, batch_step, write_sents=False, partition=""):
        # main_name = os.path.basename(self.model_base_path)
        for var, val in scores.items():
            # self.summary_writer.add_scalar(f"{main_name}/{var}", val, batch_step)
            self.summary_writer.add_scalar(var, val, batch_step)
        if write_sents:
            if len(self.last_sents) == self.args.gen_sents_in_tb:
                for ind, sent in enumerate(self.last_sents):
                    self.summary_writer.add_text(f"gen/{partition}/{ind}", sent, global_step=batch_step)
        # self.summary_writer.add_scalars(main_name, scores, batch_step)

    def sequential_generation(self, sample, decoding_style="rl", top_k=0, top_p=1.0, temp=1., ss_prob=0.):
        return self.teacher_forcing_generation(sample)

    def pg_step(self, sample, batch_i, epoch, loader_len):
        print("Policy Gradient Training")

        output = self.sequential_generation(sample, decoding_style=self.sequential_decoding_style, top_k=0, top_p=0.6)

        with torch.no_grad():
            # if self.sequential_decoding_style == "gumbel":
            #     reward = self.discriminator(output['input_onehot'], output["output_onehot"])
            # else:
            # reward = self.discriminator(sample['net_input']['src_tokens'], output["prediction"])
            reward = self.discriminator(sample["net_input"]["src_tokens"], output["prediction"])
            prev_step = torch.roll(reward, 1, dims=1)
            prev_step[:, 0] = 0.5
            reward = reward - prev_step
            # reward = self.discriminator(output["prediction"], output["prediction"])
            # gen_reward = (output["prediction"] == sample['target']).float()

        pg_loss = self.pg_criterion(output["logits"], sample['target'], reward, output.get("modified_logits", None), output.get("prediction", None))# + \
                  # self.pg_criterion(output["logits"], sample['target'], gen_reward, output.get("modified_logits", None),
                  #                   output.get("prediction", None))

        with torch.no_grad():
            if (batch_i + (epoch - 1) * loader_len) % min(self.args.train_bleu_every, loader_len) == 0:
                self.evaluate_generator(
                    sample["net_input"]["src_tokens"], output["prediction"], sample['target'], output["mask"], pg_loss,
                    sample['ntokens'], batch_i=batch_i, epoch_i=epoch, num_batches=loader_len, partition="train", strategy="rl"
                )

        self.g_optimizer.zero_grad()
        pg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.args.clip_norm)
        self.g_optimizer.step()

    def get_target_lens(self, target):
        target_lens = (torch.ones(target.size(0), dtype=torch.long) * target.size(1)).to(target.device)
        eos_idx = (target == self.dataset.src_dict.eos()).nonzero(as_tuple=False)
        target_lens[eos_idx[:, 0]] = eos_idx[:, 1] + 1
        return target_lens

    def get_length_mask(self, target, lens=None):
        if lens is None:
            lens = self.get_target_lens(target)
        mask = torch.arange(target.size(1)).to(target.device)[None, :] < lens[:, None]
        return mask

    def wrap_for_output(self, sample, logits):
        output = {
            "logits": logits,
            "target": sample["target"],
            "mask": self.get_length_mask(sample["target"]),
            "prediction": logits.argmax(-1)
        }

        output["loss"] = self.g_criterion(output["logits"][output["mask"], :], output["target"][output["mask"]])
        return output

    def teacher_forcing_generation(self, sample):
        logits = self.generator(sample)

        return self.wrap_for_output(sample, logits)

    def mle_step(self, sample, batch_i, epoch, loader_len, seq_decoding=False):

        if seq_decoding:
            print("Scheduled Sampling Training")
            ss_prob = epoch / self.args.epochs * 0.5
            print("ss_prob (probability of scheduled sampling)", ss_prob)
            output = self.sequential_generation(sample, decoding_style="ss", top_k=0, top_p=0.6, ss_prob=ss_prob)
        else:
            print("MLE Training")
            output = self.teacher_forcing_generation(sample)

        loss = output["loss"]
        # sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']

        with torch.no_grad():
            if (batch_i + (epoch - 1) * loader_len) % self.args.train_bleu_every == 0:
                self.evaluate_generator(
                    sample["net_input"]["src_tokens"], output["prediction"], sample['target'], output["mask"], loss,
                    sample['ntokens'], batch_i=batch_i, epoch_i=epoch, num_batches=loader_len, partition="train", strategy="mle"
                )

        self.g_optimizer.zero_grad()
        loss.backward()
        # all-reduce grads and rescale by grad_denom
        # for p in self.generator.parameters():
        #     if p.requires_grad:
        #         p.grad.data.div_(sample_size)
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.args.clip_norm)
        self.g_optimizer.step()

    def discrimnator_loss_acc(self, sample):
        bsz = sample['target'].size(0)  # batch_size = 64
        src_sentence = sample['net_input']['src_tokens']  # 64 x max-len i.e 64 X 50

        # now train with machine translation output i.e generator output
        true_sentence = sample['target']  # 64*50 = 3200
        true_labels = Variable(torch.ones(sample['target'].size(0)).float()).unsqueeze(1).repeat(1, sample['target'].size(1))
        # true_labels = Variable(torch.ones(sample['target'].size(0)).float())

        if self.use_cuda:
            true_sentence = true_sentence.cuda()
            true_labels = true_labels.cuda()

        with torch.no_grad():
            gen_output = self.sequential_generation(sample, decoding_style=self.sequential_decoding_style, top_k=0, top_p=0.6)  # 64 X 50 X 6632

        # if self.sequential_decoding_style == "gumbel":
        #     true_sentence = gen_output["target_onehot"]
        #     fake_sentence = gen_output["output_onehot"]
        #     src_sentence = gen_output["input_onehot"]
        # else:
        fake_sentence = gen_output["prediction"]

        fake_labels = Variable(torch.zeros(sample['target'].size(0)).float()).unsqueeze(1).repeat(1, sample['target'].size(1))
        # fake_labels = Variable(torch.zeros(sample['target'].size(0)).float())

        if self.use_cuda:
            fake_labels = fake_labels.cuda()

        disc_out_neg = self.discriminator(src_sentence, fake_sentence)
        disc_out_pos = self.discriminator(src_sentence, true_sentence)
        # disc_out_neg = self.discriminator(fake_sentence, fake_sentence)
        # disc_out_pos = self.discriminator(true_sentence, true_sentence)
        disc_out = torch.cat([disc_out_neg.squeeze(1), disc_out_pos.squeeze(1)], dim=0)

        labels = torch.cat([fake_labels, true_labels], dim=0)

        d_loss = self.d_criterion(disc_out, labels)
        acc = torch.sum(torch.round(disc_out) == labels).float() / torch.numel(labels) * 100
        return d_loss, acc

    def discriminator_step(self, sample, batch_i, epoch, loader_len):
        d_loss, acc = self.discrimnator_loss_acc(sample)

        with torch.no_grad():
            if (batch_i + (epoch - 1) * loader_len) % self.args.train_bleu_every == 0:
                self.evaluate_discriminator(
                    d_loss, acc, batch_i=batch_i, epoch_i=epoch, num_batches=loader_len, partition="train"
                )

        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()

    def format_sample(self, sample, extra_tokens=10):
        sample = copy(sample)

        max_src_len = min(sample["net_input"]['src_tokens'].size(1), max(sample["net_input"]['src_lengths'].tolist()))
        max_trg_len = min(sample["target"].size(1), max(sample["target_lengths"].tolist()))

        sample["net_input"]['src_tokens'] = sample["net_input"]['src_tokens'][:, :max_src_len].contiguous()
        sample["target"] = sample["target"][:, :max_trg_len].contiguous()
        sample["attention_mask"] = self.get_length_mask(sample["net_input"]["src_tokens"], sample["net_input"]['src_lengths'])
        return sample

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

            if hasattr(self, "discriminator"):
                self.discriminator_step(sample, i, epoch_i, len(trainloader))

        return num_update

    def decode_sentences(self, batch, for_referece=False):
        sentences = []
        for sentence in batch:
            decoded = self.dataset.dst_dict.string(sentence)
            if "▁" in decoded:
                decoded = decoded.replace(" ", "").replace("▁", " ")
            if for_referece:
                decoded = [decoded]
            sentences.append(decoded)
        return sentences

    def compute_bleu(self, predictions, references, accumulate=False):
        self.bleu_metric.add_batch(
            predictions=self.decode_sentences(predictions),
            references=self.decode_sentences(references, for_referece=True)
        )
        if not accumulate:
            bleu = self.bleu_metric.compute(
                predictions=self.decode_sentences(predictions),
                references=self.decode_sentences(references, for_referece=True)
            )
        else:
            bleu = None
        return bleu

    def compute_rouge(self, predictions, references, accumulate=False):
        self.rouge_metric.add_batch(
            predictions=self.decode_sentences(predictions),
            references=self.decode_sentences(references)
        )
        if not accumulate:
            rouge = self.rouge_metric.compute(
                predictions=self.decode_sentences(predictions),
                references=self.decode_sentences(references)
            )
        else:
            rouge = None
        return rouge

    def token_accuracy(self, predictions, targets, target_mask):
        predictions = predictions[target_mask]
        targets = targets[target_mask]
        gen_acc = torch.sum(predictions == targets).float() / torch.numel(targets) * 100
        return gen_acc

    def evaluate_generator(
            self, original, predictions, targets, target_mask, loss, ntokens, batch_i, epoch_i, num_batches, partition=None,
            strategy=None, accumulate=False, write_sents=False
    ):

        assert partition in {"train", "valid", "test"}
        assert strategy in {"mle", "rl", "gumbel"}

        sample_size = targets.size(0) if self.args.sentence_avg else ntokens

        if hasattr(self, "discriminator"):
            with torch.no_grad():
                discr_score_neg = self.discriminator(original, predictions).mean()
                discr_score_pos = self.discriminator(original, targets).mean()
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

    def evaluate_discriminator(self, d_loss, d_acc, batch_i, epoch_i, num_batches, partition=None):

        assert partition in {"train", "valid", "test"}

        self.d_logging_meters[f'{partition}_acc'].update(d_acc)
        self.d_logging_meters[f'{partition}_loss'].update(d_loss)

        logging.debug(f"D loss {self.d_logging_meters[f'{partition}_loss'].avg:.3f}, "
                      f"acc {self.d_logging_meters[f'{partition}_acc'].avg:.3f} at batch {batch_i}")
        self.write_summary({
            f"Loss/{partition}/desc": d_loss,
            f"Accuracy/{partition}/desc": d_acc,
        }, batch_i + (epoch_i - 1) * num_batches)

    def eval_generation(self, sample):
        return self.teacher_forcing_generation(sample)

    def eval_loop(self, valloader, epoch_i, force=False):
        for i, sample in enumerate(valloader):

            sample = self.format_sample(sample, extra_tokens=50)

            with torch.no_grad():
                if self.use_cuda:
                    # wrap input tensors in cuda tensors
                    sample = utils.make_variable(sample, cuda=cuda)

                if epoch_i > self.args.discriminator_pretraining or not hasattr(self, "discriminator") or force is True:
                    # generator validation
                    output = self.eval_generation(sample)
                    self.evaluate_generator(
                        sample["net_input"]["src_tokens"], output["prediction"], output["target"], output["mask"], output["loss"], ntokens=sample["ntokens"],
                        batch_i=i, epoch_i=epoch_i, num_batches=len(valloader), partition="valid", strategy="mle", accumulate=i<len(valloader)-1, write_sents=True
                    )

                # discriminator validation
                if hasattr(self, "discriminator"):
                    d_loss, acc = self.discrimnator_loss_acc(sample)
                    self.evaluate_discriminator(
                        d_loss, acc, batch_i=i, epoch_i=epoch_i, num_batches=len(valloader), partition="valid"
                    )

    def validate(self, args, epoch_i, force=False):
        # validation
        # set validation mode
        self.generator.eval()
        if hasattr(self, "discriminator"):
            self.discriminator.eval()
        # Initialize dataloader
        max_positions_valid = (args.fixed_max_len, args.fixed_max_len)
        valloader = self.dataset.eval_dataloader(
            'valid',
            max_tokens=args.max_tokens,
            max_sentences=args.joint_batch_size,
            max_positions=max_positions_valid,
            skip_invalid_size_inputs_valid_test=True,
            descending=True,  # largest batch first to warm the caching allocator
            shard_id=args.distributed_rank,
            num_shards=args.distributed_world_size,
            seed=args.seed,  # keep this constant
            sample_without_replacement=args.sample_val_without_replacement
        )

        # reset meters
        for key, val in self.g_logging_meters.items():
            if val is not None:
                val.reset()
        for key, val in self.d_logging_meters.items():
            if val is not None:
                val.reset()

        print(f"Validation batches: {len(valloader)}")

        self.eval_loop(valloader, epoch_i, force=force)

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
            if hasattr(self, "discriminator"):
                self.discriminator.train()
            # update_learning_rate(num_update, 8e4, args.g_learning_rate, args.lr_shrink, self.g_optimizer)

            print(f"Training batches: {len(trainloader)}")

            num_update = self.train_loop(trainloader, epoch_i, num_update)

            self.validate(args, epoch_i)

            self.save_models(epoch_i)

            if self.g_logging_meters['valid_loss'].avg < best_dev_loss:
                best_dev_loss = self.g_logging_meters['valid_loss'].avg
                self.save_generator(os.path.join(self.checkpoints_path, "best_gmodel.pt"))

    def save_generator(self, path):
        torch.save(self.generator, open(path, 'wb'), pickle_module=dill)

    def save_discriminator(self, path):
        if hasattr(self, "discriminator"):
            torch.save(self.discriminator, open(path, 'wb'), pickle_module=dill)

    def save_models(self, epoch_i):
        self.save_generator(os.path.join(self.checkpoints_path, f"joint_{self.g_logging_meters['valid_loss'].avg:.3f}.epoch_{epoch_i}_gen.pt"))
        self.save_discriminator(os.path.join(self.checkpoints_path, f"joint_{self.g_logging_meters['valid_loss'].avg:.3f}.epoch_{epoch_i}_discr.pt"))
        with open(os.path.join(self.checkpoints_path, "params.json"), "w") as paramsink:
            paramsink.write(json.dumps(self.args.__dict__, indent=4))


def update_learning_rate(update_times, target_times, init_lr, lr_shrink, optimizer):

    lr = init_lr * (lr_shrink ** (update_times // target_times))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr