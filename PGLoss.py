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

    def forward(self, logprobs, label, reward, use_cuda):
        bsz, seqlen, _ = logprobs.size()

        logprobs = logprobs.clone()

        with torch.no_grad():
            mask = torch.zeros_like(logprobs)
            mask = torch.scatter(mask, 2, label.unsqueeze(2), 1.)

        loss = -torch.sum(logprobs * reward.unsqueeze(2) * mask)

        if self.size_average:
            loss = loss/bsz

        
        return loss
