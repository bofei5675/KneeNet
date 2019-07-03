import gc
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
import torch
import os


def validate_epoch(net, val_loader, criterion,use_cuda = True,loss_type='CE'):
    net.train(False)

    running_loss = 0.0
    sm = nn.Softmax(dim=1)

    truth = []
    preds = []
    bar = tqdm(total=len(val_loader), desc='Processing', ncols=90)
    names_all = []
    n_batches = len(val_loader)
    for i, (batch, targets, names) in enumerate(val_loader):
        # forward + backward + optimize
        if use_cuda:
            if loss_type == 'CE':
                labels = Variable(targets.long().cuda())
                inputs = Variable(batch.cuda())
            elif loss_type == 'MSE':
                labels = Variable(targets.float().cuda())
                inputs = Variable(batch.cuda())
        else:
            if loss_type == 'CE':
                labels = Variable(targets.float())
                inputs = Variable(batch.cuda())
            elif loss_type == 'MSE':
                labels = Variable(targets.float())
                inputs = Variable(batch)

        outputs = net(inputs)

        loss = criterion(outputs, labels)
        if loss_type =='CE':
            probs = sm(outputs).data.cpu().numpy()
        elif loss_type =='MSE':
            probs = outputs
            probs[probs < 0] = 0
            probs[probs > 4] = 4
            probs = probs.view(1,-1).squeeze(0).round().data.cpu().numpy()
        preds.append(probs)
        truth.append(targets.cpu().numpy())
        names_all.extend(names)

        running_loss += loss.item()
        bar.update(1)
        gc.collect()
    gc.collect()
    bar.close()
    if loss_type =='CE':
        preds = np.vstack(preds)
    else:
        preds = np.hstack(preds)
    truth = np.hstack(truth)

    return running_loss / n_batches, preds, truth, names_all