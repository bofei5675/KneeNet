from __future__ import print_function
from torch.autograd import Variable
import gc
import torch.nn as nn
import torch
from val_utils import validate_epoch
import time
import numpy as np
import os
from sklearn.metrics import confusion_matrix, mean_squared_error, cohen_kappa_score

def adjust_learning_rate(optimizer, epoch, args):
    """
    Decreases the initial LR by 10 every drop_step epochs.
    Conv layers learn slower if specified in the optimizer.
    """
    lr = args.lr * (0.1 ** (epoch // args.lr_drop))
    if lr < args.lr_min:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer, lr


def train_epoch(epoch, net, optimizer, train_loader, criterion, max_ep,use_cuda = True,output_file_path = None):
    net.train(True)

    running_loss = 0.0
    n_batches = len(train_loader)
    sm = nn.Softmax(dim=1)
    batch_correct = 0
    num_samples = 0

    for i, (batch, targets, names) in enumerate(train_loader):
        optimizer.zero_grad()

        # forward + backward + optimize
        if use_cuda:
            labels = Variable(targets.long().cuda())
            inputs = Variable(batch.cuda())
        else:
            labels = Variable(targets.long())
            inputs = Variable(batch)
        outputs = net(inputs)
        probs = sm(outputs).data.cpu().numpy()
        preds = probs.argmax(1)
        truth = targets.data.cpu().numpy()
        batch_correct += (preds == truth).sum()
        num_samples += probs.shape[0]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        output = '[%d | %d, %5d / %d] | Running loss: %.3f / loss %.3f | Acc: %.4f' % (epoch + 1, max_ep, i + 1,
                                                                        n_batches, running_loss / (i + 1),
                                                                        loss.item(), batch_correct / num_samples)
        print(output)
        with open(output_file_path,'a+') as f:
            f.write(output + '\n')
        gc.collect()
    gc.collect()

    return running_loss / n_batches, batch_correct / num_samples

def train_iterations(epoch, net,
                     optimizer, train_loader,
                     val_loader,criterion,
                     max_ep,
                     use_cuda = True,
                     output_file_path = None,
                     iteration = None,
                     start_val = 0,
                     model_file_path="",
                     loss_type='CE'):
    '''

    :param epoch:
    :param net:
    :param optimizer:
    :param train_loader:
    :param val_loader:
    :param criterion:
    :param max_ep:
    :param use_cuda:
    :param output_file_path:
    :param iteration:
    :return:
    '''
    net.train(True)
    running_loss = 0.0
    n_batches = len(train_loader)
    sm = nn.Softmax(dim=1)
    batch_correct = 0
    num_samples = 0
    train_losses = []
    val_losses = []
    val_mse = []
    val_kappa = []
    val_acc = []
    best_dice = 0
    prev_model = None
    train_started = time.time()
    for i, (batch, targets, names) in enumerate(train_loader):
        optimizer.zero_grad()

        # forward + backward + optimize
        if use_cuda:
            if loss_type =='CE':
                labels = Variable(targets.long().cuda())
                inputs = Variable(batch.cuda())
            elif loss_type =='MSE':
                labels = Variable(targets.float().cuda())
                inputs = Variable(batch.cuda())
        else:
            if loss_type == 'CE':
                labels = Variable(targets.float())
                inputs = Variable(batch.cuda())
            elif loss_type == 'MSE':
                labels = Variable(targets.float())
                inputs = Variable(batch)
        print('label:',labels)
        outputs = net(inputs)
        print('outputs:',outputs)
        truth = targets.data.cpu().numpy()
        if loss_type == 'CE':
            probs = sm(outputs).data.cpu().numpy()
            preds = probs.argmax(1)

        elif loss_type =='MSE':
            outputs[outputs < 0] = 0
            outputs[outputs > 4] = 4
            preds = outputs.round().data.cpu().numpy()
        batch_correct += (preds == truth).sum()
        num_samples += truth.shape[0]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        output = '[%d | %d, %5d / %d] | Running loss: %.3f / loss %.3f | Acc: %.4f' % (epoch + 1, max_ep, i + 1,
                                                                    n_batches, running_loss / (i + 1),
                                                                        loss.item(), batch_correct / num_samples)
        print(output)
        with open(output_file_path, 'a+') as f:
            f.write(output + '\n')
        if (i+1) % iteration == 0 or (i+1) == n_batches:
            start = time.time()
            net.eval()
            val_loss, probs, truth, _ = validate_epoch(net, val_loader, criterion, use_cuda,loss_type)
            preds = probs.argmax(1)
            # Validation metrics
            cm = confusion_matrix(truth, preds)
            kappa = np.round(cohen_kappa_score(truth, preds, weights="quadratic"), 4)
            # avoid divide by 0 error when testing...
            acc = np.round(np.mean(cm.diagonal().astype(float) / (cm.sum(axis=1) + 1e-12) ), 4)
            mse = np.round(mean_squared_error(truth, preds), 4)
            val_time = np.round(time.time() - start, 4)
            train_losses.append(running_loss / (i  + 1))
            val_losses.append(val_loss)
            val_mse.append(mse)
            val_acc.append(acc)
            val_kappa.append(kappa)
            with open(output_file_path, 'a+') as f:
                f.write(str(cm) + '\n')
                f.write(str(cm.diagonal() / (cm.sum(axis=1) + 1e-12)) + '\n')
                f.write('Epoch {}; Iteration {}: Val Loss {}; Val Acc {}; Val MSE {}; Val Kappa {};\n' \
                        .format(epoch + 1,i, val_loss, acc, mse, kappa))

            # Making logs backup
            # epoch, current iteration, number of batches, loss, train acc, validation metrics.
            logs = [epoch + 1, i, n_batches, train_losses, batch_correct / num_samples, val_losses, val_mse, val_acc, val_kappa]

            if epoch >= start_val:
                # We will be saving only the snapshot which has lowest loss value on the validation set
                cur_snapshot_name = os.path.join(model_file_path, 'epoch_{}.pth'.format(epoch + 1))
                if prev_model is None:
                    torch.save(net.state_dict(), cur_snapshot_name)
                    prev_model = cur_snapshot_name
                    best_kappa = kappa
                    best_acc = acc
                    with open(output_file_path, 'a+') as f:
                        f.write('Save {}'.format(cur_snapshot_name))
                else:
                    if acc > best_acc:
                        os.remove(prev_model)
                        best_kappa = kappa
                        best_acc = acc
                        print('Saved snapshot:', cur_snapshot_name)
                        torch.save(net.state_dict(), cur_snapshot_name)
                        with open(output_file_path, 'a+') as f:
                            f.write('Save {}\n'.format(cur_snapshot_name))
                        prev_model = cur_snapshot_name
            net.train()
        gc.collect()
    gc.collect()
    return running_loss / n_batches, batch_correct / num_samples, logs

def load_my_state_dict(model, state_dict):
    own_state = model.state_dict()
    print('Total Module to load {}'.format(len(own_state.keys())))
    print('Total Module from weights file {}'.format(len(state_dict.keys())))
    count = 0
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        count +=1
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)
    print('Load Successful {} / {}'.format(count, len(own_state.keys())))