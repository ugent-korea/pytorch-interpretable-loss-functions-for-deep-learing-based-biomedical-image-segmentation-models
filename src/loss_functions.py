import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, reduce=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, input, target):
        if self.reduce == False:
            N, C, H, W = input.size()
        input = input.view(input.size(0), input.size(1), -1)  # (N,C,H,W) => (N,C,H*W)
        input = input.transpose(1, 2)    # (N,C,H*W) => (N,H*W,C)
        input = input.contiguous().view(-1, input.size(2))   # (N,H*W,C) => (N*H*W,C)
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)  # (N*H*W,C)
        # only get the softmax score of the target logit.gather(index, target)
        logpt = logpt.gather(1, target)  # (N*H*W,1)
        logpt = logpt.view(-1)  # (N*H*W)
        pt = Variable(logpt.data.exp())
        loss = -1 * (1-pt)**self.gamma * logpt

        if self.reduce == True:
            return loss.mean()
        else:
            return loss.view(N, H, W)


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=0, reduce=True):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        N, C, H, W = inputs.size()
        inputs = inputs.contiguous().view(N, H, W)  # (N,1,H,W) => (N,H,W)
        targets = targets.float()
        # print("input", input, target)
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, self.reduce)
        pt = Variable(torch.exp(-BCE_loss))
        F_loss = (1-pt)**self.gamma * BCE_loss
        return F_loss


class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self, reduce=True, class_weight=None):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.reduce = reduce
        self.class_weight = class_weight

    def forward(self, inputs, targets):
        inputs = inputs.contiguous().view(inputs.size(0), inputs.size(2), inputs.size(3))  # (N,1,H,W) => (N,H,W)
        # print("input", input, target)
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduce=self.reduce, pos_weight=self.class_weight)
        return BCE_loss
