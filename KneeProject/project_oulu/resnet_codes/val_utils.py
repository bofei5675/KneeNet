"""
Validation utils
(c) Aleksei Tiulpin, University of Oulu, 2017
"""


import gc
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
import torch
import os

def validate_epoch(net, val_loader, criterion,use_cuda = False,loss_type="CE"):

    net.train(False)

    running_loss = 0.0
    sm = nn.Softmax(dim = 1)

    truth = []
    preds = []
    bar = tqdm(total=len(val_loader),desc='Processing', ncols=90)
    names_all = []
    n_batches = len(val_loader)
    for i, (batch, targets, names) in enumerate(val_loader):

        # forward + backward + optimize
        if use_cuda:
            labels = Variable(targets.long().cuda())
            inputs = Variable(batch.cuda())
        else:
            labels = Variable(targets.long())
            inputs = Variable(batch)

        outputs = net(inputs)

        loss = criterion(outputs, labels)

        if loss_type == 'CE':
            probs = sm(outputs).data.cpu().numpy()
            preds.append(probs)
            truth.append(targets.cpu().numpy())
        names_all.extend(names)


        running_loss += loss.item()
        bar.update(1)
        gc.collect()
    gc.collect()
    bar.close()
    preds = np.vstack(preds)
    truth = np.hstack(truth)

    return running_loss/n_batches, preds, truth, names_all
