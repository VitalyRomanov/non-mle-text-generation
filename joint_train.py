import argparse
import logging

from torch import cuda

import options
import utils

from ModelTrainer import ModelTrainer
from discriminator import AttDiscriminator
from generator import VarLSTMModel

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Adversarial-NMT.")

# Load args
parser.add_argument("model_name")
options.add_general_args(parser)
options.add_dataset_args(parser)
options.add_distributed_training_args(parser)
options.add_optimization_args(parser)
options.add_checkpoint_args(parser)
options.add_generator_model_args(parser)
options.add_discriminator_model_args(parser)
options.add_generation_args(parser)


class GanLSTMTrainer(ModelTrainer):
    def create_generator(self, args):
        # self.generator = LSTMModel(args, self.dataset.src_dict, self.dataset.dst_dict, use_cuda=self.use_cuda)
        self.generator = VarLSTMModel(args, self.dataset.src_dict, self.dataset.dst_dict, use_cuda=self.use_cuda)
        print("Generator loaded successfully!")

    def create_discriminator(self, args):
        # discriminator = Discriminator(args, dataset.src_dict, dataset.dst_dict, use_cuda=use_cuda)
        self.discriminator = AttDiscriminator(args, self.dataset.src_dict, self.dataset.dst_dict,
                                              use_cuda=self.use_cuda)
        print("Discriminator loaded successfully!")


class VarLSTMTrainer(ModelTrainer):
    def create_generator(self, args):
        self.generator = VarLSTMModel(args, self.dataset.src_dict, self.dataset.dst_dict, use_cuda=self.use_cuda)
        self.kld_weight = 1.
        print("Generator loaded successfully!")

    def create_discriminator(self, args):
        # discriminator = Discriminator(args, dataset.src_dict, dataset.dst_dict, use_cuda=use_cuda)
        self.discriminator = AttDiscriminator(args, self.dataset.src_dict, self.dataset.dst_dict,
                                              use_cuda=self.use_cuda)
        print("Discriminator loaded successfully!")

    def mle_generator_loss(self, sample):
        sys_out_batch, kld = self.generator(sample)
        out_batch = sys_out_batch.contiguous().view(-1, sys_out_batch.size(-1))  # (64 X 50) X 6632
        trg_batch = sample['target'].view(-1)  # 64*50 = 3200

        loss = self.g_criterion(out_batch, trg_batch) + self.kld_weight * kld
        return loss

    def train_loop(self, trainloader, epoch_i, num_update):
        for i, sample in enumerate(trainloader):

            if self.use_cuda:
                # wrap input tensors in cuda tensors
                sample = utils.make_variable(sample, cuda=cuda)

            self.mle_step(sample, i, epoch_i, len(trainloader))
            num_update += 1

        return num_update



if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        logging.warning(f"unknown arguments: {parser.parse_known_args()[1]}")
    model_name = options.model_name
    if model_name == "gan":
        trainer = GanLSTMTrainer(options)
    elif model_name == "var":
        trainer = VarLSTMTrainer(options)
    elif model_name == "mle":
        pass
        # trainer =
    trainer.train()