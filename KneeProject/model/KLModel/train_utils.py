from __future__ import print_function
from torch.autograd import Variable
import gc
import torch.nn as nn
import torch


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
        with open(output_file_path,'a+') as f:
            f.write(str(outputs.data.cpu().numpy()))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        output = '[%d | %d, %5d / %d] | Running loss: %.3f / loss %.3f' % (epoch + 1, max_ep, i + 1,
                                                                        n_batches, running_loss / (i + 1),
                                                                        loss.item())
        print(output)
        with open(output_file_path,'a+') as f:
            f.write(output + '\n')
        gc.collect()
    gc.collect()

    return running_loss / n_batches
