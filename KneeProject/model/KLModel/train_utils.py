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
        #print(preds,truth)
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