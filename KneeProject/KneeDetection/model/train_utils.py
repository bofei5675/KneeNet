
from torch.autograd import Variable
import gc
import torch.nn as nn
import torch
import time
import numpy as np
import os
from sklearn.metrics import confusion_matrix, mean_squared_error, cohen_kappa_score
from tqdm import tqdm
def train_iterations(net, optimizer, train_loader,val_loader,
                     criterion, max_ep,use_cuda = True,iterations = 1000,
                     log_dir = None, model_dir = None):
    '''

    :param net:
    :param optimizer:
    :param train_loader:
    :param val_loader:
    :param criterion:
    :param max_ep:
    :param use_cuda:
    :param iterations:
    :param log_dir:
    :param model_dir:
    :return:
    '''
    net.train(True)


    n_batches = len(train_loader)
    train_losses = []
    val_losses = []
    pre_model = None # best model
    train_start = time.time()
    for epoch in range(max_ep):
        running_loss = 0.0
        for i, (batch, targets, names) in enumerate(train_loader):
            optimizer.zero_grad()
            targets = torch.stack(targets).transpose(0,1)
            # forward + backward + optimize
            if use_cuda:
                labels = Variable(targets.float().cuda())
                inputs = Variable(batch.cuda())
            else:
                labels = Variable(targets.float())
                inputs = Variable(batch)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            log_output = '[%d | %d, %5d / %d] | Running loss: %.5f / loss %.5f ]' % (epoch + 1, max_ep, i + 1,
                                                                            n_batches, running_loss / (i + 1),
                                                                            loss.item())
            print(log_output)
            with open(log_dir,'a+') as f:
                f.write(log_output + '\n')

            if (i + 1) % iterations == 0 or (i + 1) == n_batches:
                val_loss = validate_epoch(net,criterion,val_loader,use_cuda)
                val_losses.append(val_losses)
                train_losses.append(running_loss / (i + 1))
                log_output = '[Epoch %d | Val Loss %.5f | Train Loss %.5f ]' % (epoch + 1,val_loss, running_loss / (i + 1))
                print(log_output)
                with open(log_dir, 'a+') as f:
                    f.write(log_output + '\n')
                if pre_model is None or val_loss < pre_model:
                    print('Save the model at epoch {}, iter {}'.format(epoch,i + 1))
                    snapshot = model_dir + '/' + 'epoch_{}.pth'.format(epoch)
                    torch.save(net.state_dict(), snapshot)
                    pre_model = val_loss
            gc.collect()
        gc.collect()
    print('Training takes {} seconds'.format(time.time() - train_start))

def validate_epoch(net,criterion,val_loader,use_cuda):
    net.eval()
    running_loss = 0.0
    n_batches = len(val_loader)
    bar = tqdm(total=len(val_loader), desc='Processing', ncols=90)
    for i, (batch, targets, names) in enumerate(val_loader):
        targets = torch.stack(targets).transpose(0, 1)
        print(names)
        # forward + backward + optimize
        if use_cuda:
            labels = Variable(targets.float().cuda())
            inputs = Variable(batch.cuda())
        else:
            labels = Variable(targets.float())
            inputs = Variable(batch)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        bar.update(1)
        gc.collect()
    net.train(True)
    return running_loss / n_batches

def metrics_iou(boxA,boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou
