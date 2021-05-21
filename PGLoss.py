import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math


class PGLoss(torch.nn.Module):
    
    def __init__(self, ignore_index=None, size_average=False, reduce=True):
        super(PGLoss, self).__init__()
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.reduce = reduce

    def forward(self, logprobs, label, reward, modified_logprobs=None, predicted_tokens=None):
        bsz, seqlen, _ = logprobs.size()

        logprobs = logprobs.clone()

        def create_mask(l_probs, lbl):
            with torch.no_grad():
                mask = torch.zeros_like(l_probs)
                mask = torch.scatter(mask, 2, lbl.unsqueeze(2), 1.)
            return mask

        logprobs_mask = create_mask(logprobs, label)
        loss = -torch.sum(torch.sum(logprobs * reward.unsqueeze(2) * logprobs_mask, dim=-1), dim=-1)

        if self.size_average:
            loss = loss/bsz

        if modified_logprobs is not None:
            modified_logprobs_mask = create_mask(modified_logprobs, predicted_tokens)
            with torch.no_grad():
                modified_logprobs_sum = torch.sum(torch.log(torch.sum(torch.exp(modified_logprobs) * modified_logprobs_mask, dim=-1)), dim=-1)
                logprobs_sum = torch.sum(torch.log(torch.sum(torch.exp(logprobs) * modified_logprobs_mask, dim=-1)), dim=-1)
                importance_sampling_correct_coef = torch.exp(logprobs_sum - modified_logprobs_sum)
            loss = importance_sampling_correct_coef * loss

        return loss.sum()
