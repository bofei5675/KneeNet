import torch
import torch.nn as nn
import torch.nn.functional as F

class EntropyLoss(nn.Module):
    def __init__(self, beta = 1.0):
        super(EntropyLoss,self).__init__()
        self.beta = beta
        self.CE = nn.CrossEntropyLoss()

    def forward(self, inp, target):
        CE = self.CE(inp, target)
        batch_size = target.shape[0]
        h = self._compute_entropy(inp) / batch_size
        return CE - self.beta * h

    def _compute_entropy(self,inp):
        h = F.softmax(inp, dim = 1) * F.log_softmax(inp,dim =1)
        h = -1 * h.sum()
        return h