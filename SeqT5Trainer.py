import random

from torch import cuda
from torch.autograd import Variable

import utils
from ModelTrainer import ModelTrainer, update_learning_rate
import torch
from discriminator import Discriminator, AttDiscriminator, GumbelDiscriminator


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
        self.generator = SeqT5.from_pretrained('t5-small')

    def create_discriminator(self, args):
        # raise NotImplementedError()
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
        # TODO need this?
        # fix discriminator word embedding (as Wu et al. do)
        if hasattr(self, "discriminator"):
            for p in self.discriminator.embed_src_tokens.parameters():
                p.requires_grad = False
            for p in self.discriminator.embed_trg_tokens.parameters():
                p.requires_grad = False

    def transform_for_t5(self, tensor):
        return tensor - 1

    def transform_from_t5(self, tensor):
        return tensor + 1

    def wrap_for_output(self, sample, logits, input_onehot=None, output_onehot=None, target_onehot=None):
        if input_onehot is not None: # add zeros to use indexing from 1
            zeros = torch.zeros((input_onehot.shape[0], input_onehot.shape[1], 1)).to(input_onehot.device)
            input_onehot = torch.cat([zeros, input_onehot], dim=2)
            zeros = torch.zeros((output_onehot.shape[0], output_onehot.shape[1], 1)).to(output_onehot.device)
            output_onehot = torch.cat([zeros, output_onehot], dim=2)
            zeros = torch.zeros((target_onehot.shape[0], target_onehot.shape[1], 1)).to(target_onehot.device)
            target_onehot = torch.cat([zeros, target_onehot], dim=2)

        output = {
            "logits": logits,
            "target": sample["target"],
            "mask": self.get_length_mask(sample["target"]),
            "prediction": self.transform_from_t5(logits.argmax(-1)),
            "input_onehot": input_onehot,
            "output_onehot": output_onehot,
            "target_onehot": target_onehot,
        }

        output["loss"] = self.g_criterion(
            output["logits"][output["mask"], :],
            self.transform_for_t5(output["target"])[output["mask"]]
        )
        return output

    def sequential_generation(self, sample, decoding_style="rl", top_k=0, top_p=0.6, temp=.2):
        t5out = self.generator(
            self.transform_for_t5(sample['net_input']['src_tokens']),
            labels=self.transform_for_t5(sample['target']), decoding_style=decoding_style, top_k=top_k, top_p=top_p,
            temperature=temp, epsilon=self.args.imp_smpl_epsilon
        )

        # if decoding_style == "gumbel":
        #     return self.wrap_for_output(sample, t5out.logits, input_onehot=t5out.input_onehot, output_onehot=t5out.output_onehot, target_onehot=t5out.target_onehot)
        return self.wrap_for_output(sample, t5out.logits)

    def teacher_forcing_generation(self, sample):
        logits = self.generator(
            self.transform_for_t5(sample['net_input']['src_tokens']),
            labels=self.transform_for_t5(sample['target']), decoding_style="tf"
        ).logits

        return self.wrap_for_output(sample, logits)

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


class SeqT5RL(SeqT5Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_strategy = "alternate"  # alternate | mle | rl
        self.sequential_decoding_style = "rl"

    def seq_mle_step(self, sample, batch_i, epoch, loader_len):
        # MLE training
        print("Seq MLE Training")

        output = self.sequential_generation(sample, decoding_style="gumbel", top_k=1)
        loss = output["loss"]
        # sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']

        with torch.no_grad():
            if (batch_i + (epoch - 1) * loader_len) % 1 == 0:
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

    def train_loop(self, trainloader, epoch_i, num_update):
        for i, sample in enumerate(trainloader):

            sample = self.format_sample(sample)

            if self.use_cuda:
                # wrap input tensors in cuda tensors
                sample = utils.make_variable(sample, cuda=cuda)

            if epoch_i > self.args.discriminator_pretraining or not hasattr(self, "discriminator"):
                if hasattr(self, "discriminator"):
                    if self.training_strategy == "alternate":
                        if random.random() >= 0.5:  # TODO why use both?
                            self.mle_step(sample, i, epoch_i, len(trainloader))
                        else:
                            if random.random() > 0.5:
                                self.seq_mle_step(sample, i, epoch_i, len(trainloader))
                            else:
                                self.pg_step(sample, i, epoch_i, len(trainloader))
                    elif self.training_strategy == "mle":
                        self.mle_step(sample, i, epoch_i, len(trainloader))
                    elif self.training_strategy == "rl":
                        self.pg_step(sample, i, epoch_i, len(trainloader))
                    else:
                        raise ValueError(f"Invalid training strategy: {self.training_strategy}. Valid options are: alternate|mle|rl.")
                else:
                    self.mle_step(sample, i, epoch_i, len(trainloader))
                num_update += 1
            else:
                if i == 0:
                    print("Pretraining discriminator for one epoch")

            if hasattr(self, "discriminator"):
                self.discriminator_step(sample, i, epoch_i, len(trainloader))

        return num_update


class SeqT5Gumbel(SeqT5RL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_strategy = "alternate"  # alternate | mle | rl
        self.sequential_decoding_style = "gumbel"

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

    def sequential_generation(self, sample, decoding_style="rl", top_k=0, top_p=0.6, temp=.2):
        t5out = self.generator(
            self.transform_for_t5(sample['net_input']['src_tokens']),
            labels=self.transform_for_t5(sample['target']), decoding_style=decoding_style, top_k=top_k, top_p=top_p,
            temperature=temp, epsilon=self.args.imp_smpl_epsilon
        )

        return self.wrap_for_output(sample, t5out.logits, input_onehot=t5out.input_onehot, output_onehot=t5out.output_onehot, target_onehot=t5out.target_onehot)

    # def handicap_discriminator(self):
    #     pass

    def seq_mle_step(self, sample, batch_i, epoch, loader_len):
        # MLE training
        print("Seq MLE Training")

        output = self.sequential_generation(sample, decoding_style="gumbel", top_k=1)
        loss = output["loss"]
        # sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']

        with torch.no_grad():
            if (batch_i + (epoch - 1) * loader_len) % 1 == 0:
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

    def train_loop(self, trainloader, epoch_i, num_update):
        for i, sample in enumerate(trainloader):

            sample = self.format_sample(sample)

            if self.use_cuda:
                # wrap input tensors in cuda tensors
                sample = utils.make_variable(sample, cuda=cuda)

            if epoch_i > self.args.discriminator_pretraining or not hasattr(self, "discriminator"):
                if hasattr(self, "discriminator"):
                    if self.training_strategy == "alternate":
                        if random.random() >= 0.5:  # TODO why use both?
                            self.mle_step(sample, i, epoch_i, len(trainloader))
                        else:
                            if random.random() > 0.5:
                                self.seq_mle_step(sample, i, epoch_i, len(trainloader))
                            else:
                                self.pg_step(sample, i, epoch_i, len(trainloader))
                    elif self.training_strategy == "mle":
                        self.mle_step(sample, i, epoch_i, len(trainloader))
                    elif self.training_strategy == "rl":
                        self.pg_step(sample, i, epoch_i, len(trainloader))
                    else:
                        raise ValueError(f"Invalid training strategy: {self.training_strategy}. Valid options are: alternate|mle|rl.")
                else:
                    self.mle_step(sample, i, epoch_i, len(trainloader))
                num_update += 1
            else:
                if i == 0:
                    print("Pretraining discriminator for one epoch")

            if hasattr(self, "discriminator"):
                self.discriminator_step(sample, i, epoch_i, len(trainloader))

        return num_update