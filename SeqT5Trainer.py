import logging
import random

from torch import cuda
from torch.autograd import Variable

import utils
from ModelTrainer import ModelTrainer, update_learning_rate
import torch
from discriminator import Discriminator, AttDiscriminator, GumbelDiscriminator, T5Discriminator, T5SemanticDiscriminator, BleurtDiscriminator


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

        return num_update

    def pg_step(self, sample, batch_i, epoch, loader_len):
        print("Policy Gradient Training")

        output = self.sequential_generation(sample, decoding_style=self.sequential_decoding_style, top_k=0, top_p=0.6)

        with torch.no_grad():
            reward = self.discriminator(output["prediction"], sample["target"]) # dim (bsize x 1)

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
