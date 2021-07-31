import argparse
import logging
import math

import torch
from torch import cuda

import options
import utils

from ModelTrainer import ModelTrainer
from SeqT5Trainer import SeqT5Trainer, SeqT5Mle, SeqT5RL, SeqT5Gumbel, SeqT5Bleurt, SeqEmbT5Bleurt
from discriminator import AttDiscriminator
from generator import VarLSTMModel, LSTMModel

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Adversarial-NMT.")

# Load args
parser.add_argument("--model_name", default=None)
parser.add_argument("--note", default=None)
options.add_general_args(parser)
options.add_dataset_args(parser)
options.add_distributed_training_args(parser)
options.add_optimization_args(parser)
options.add_checkpoint_args(parser)
options.add_generator_model_args(parser)
options.add_discriminator_model_args(parser)
options.add_generation_args(parser)


class GanLSTMTrainer(ModelTrainer):
    def __init__(self, args):
        # Set model parameters
        args.encoder_embed_dim = 128
        args.encoder_layers = 2  # 4
        args.encoder_dropout_out = 0
        args.decoder_embed_dim = 128
        args.decoder_layers = 2  # 4
        args.decoder_out_embed_dim = 128
        args.decoder_dropout_out = 0
        args.bidirectional = False

        super(GanLSTMTrainer, self).__init__(args)

    def create_generator(self, args):
        self.generator = LSTMModel(args, self.dataset.src_dict, self.dataset.dst_dict, use_cuda=self.use_cuda)
        print("Generator loaded successfully!")

    def create_discriminator(self, args):
        # discriminator = Discriminator(args, dataset.src_dict, dataset.dst_dict, use_cuda=use_cuda)
        self.discriminator = AttDiscriminator(args, self.dataset.src_dict, self.dataset.dst_dict,
                                              use_cuda=self.use_cuda)
        print("Discriminator loaded successfully!")


class LstmMleTrainer(ModelTrainer):
    def __init__(self, args):
        # Set model parameters
        args.encoder_embed_dim = 128
        args.encoder_layers = 2  # 4
        args.encoder_dropout_out = 0
        args.decoder_embed_dim = 128
        args.decoder_layers = 2  # 4
        args.decoder_out_embed_dim = 128
        args.decoder_dropout_out = 0
        args.bidirectional = False

        super(LstmMleTrainer, self).__init__(args)
        self.training_strategy = "mle"

    def create_generator(self, args):
        self.generator = LSTMModel(args, self.dataset.src_dict, self.dataset.dst_dict, use_cuda=self.use_cuda)
        print("Generator loaded successfully!")

    def create_discriminator(self, args):
        pass


class VarLSTMTrainer(LstmMleTrainer):
    def __init__(self, args):
        # Set model parameters
        args.encoder_embed_dim = 128
        args.encoder_layers = 2  # 4
        args.encoder_dropout_out = 0
        args.decoder_embed_dim = 128
        args.decoder_layers = 2  # 4
        args.decoder_out_embed_dim = 128
        args.decoder_dropout_out = 0
        args.bidirectional = False

        super(VarLSTMTrainer, self).__init__(args)
        self.training_strategy = "mle"

    def create_generator(self, args):
        self.generator = VarLSTMModel(args, self.dataset.src_dict, self.dataset.dst_dict, use_cuda=self.use_cuda)
        self.kld_weight = 1.
        print("Generator loaded successfully!")

    def wrap_for_output(self, sample, logits, kld):
        output = {
            "logits": logits,
            "target": sample["target"],
            "mask": self.get_length_mask(sample["target"]),
            "prediction": logits.argmax(-1)
        }

        output["loss"] = self.g_criterion(output["logits"][output["mask"], :], output["target"][output["mask"]]) + self.kld_weight * kld
        return output

    def teacher_forcing_generation(self, sample):
        logits, kld = self.generator(sample)

        return self.wrap_for_output(sample, logits, kld)


if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        logging.warning(f"unknown arguments: {parser.parse_known_args()[1]}")
    model_name = options.model_name
    assert model_name is not None
    # options.note = None
    if model_name == "gan":
        trainer = GanLSTMTrainer(options)
    elif model_name == "vae":
        trainer = VarLSTMTrainer(options)
    elif model_name == "mle":
        trainer = LstmMleTrainer(options)
    elif model_name == "t5mle":
        trainer = SeqT5Mle(options)
    elif model_name == "t5rl":
        trainer = SeqT5RL(options)
    elif model_name == "t5gumbel":
        trainer = SeqT5Gumbel(options)
    elif model_name == "t5bleurt":
        trainer = SeqT5Bleurt(options)
    elif model_name == "embt5bleurt":
        trainer = SeqEmbT5Bleurt(options)
    else:
        raise ValueError("Choose appropriate model")
    trainer.train()