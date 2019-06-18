from __future__ import print_function
import sys
sys.path.append('../../KLModel')
from dataloader import *
from augmentation import *
import matplotlib
import matplotlib.pyplot as plt
from dataloader import KneeGradingDataset
from tqdm import tqdm
import numpy as np
import argparse
import os
import itertools
from train_utils import *
from val_utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import gc

import torchvision.transforms as transforms
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, mean_squared_error, cohen_kappa_score, roc_auc_score, roc_curve, log_loss
from sklearn.preprocessing import OneHotEncoder
from torchvision.models import resnet34
import time
import pickle
import pandas as pd
import time
import sys
def load_img(fname):
    f = h5py.File(fname)
    img = f['data']
    print(img.shape)
    img = np.expand_dims(img,axis=2)
    img = np.repeat(img[:, :], 3, axis=2)
    f.close()
    return img

def weigh_maps(weights, maps,use_cuda = False):
    maps = maps.squeeze()
    weights = weights.squeeze()
    if use_cuda:
        res = Variable(torch.zeros(maps.size()[-2:]).cuda(), requires_grad=False)
    else:
        res = Variable(torch.zeros(maps.size()[-2:]), requires_grad=False)
    for i, w in enumerate(weights):
        res += w * maps[i]

    return res

# Producing the GradCAM output using the equations provided in the article
def gradcam_resnet(fname,net,scale_tensor_transform,use_cuda = False):
    img = load_img(fname)
    img = scale_tensor_transform(img)
    print('After processing:',img.shape)
    inp = img.view(1, 3, 896, 896)
    net.train(False);
    net.zero_grad()

    features = nn.Sequential(net.module.conv1,
                             net.module.bn1,
                             net.module.relu,
                             net.module.maxpool,
                             net.module.layer1,
                             net.module.layer2,
                             net.module.layer3,
                             net.module.layer4)
    if use_cuda:
        maps = features(Variable(inp.cuda()))
    else:
        maps = features(Variable(inp))
    maps_avg = F.avg_pool2d(maps, 28).view(1, 512)

    grads = []
    maps_avg.register_hook(lambda x: grads.append(x));

    out = net.module.fc(maps_avg)

    ohe = OneHotEncoder(sparse=False, n_values=5)
    index = np.argmax(out.cpu().data.numpy(), axis=1).reshape(-1, 1)
    if use_cuda:
        out.backward(torch.from_numpy(ohe.fit_transform(index)).float().cuda())
    else:
        out.backward(torch.from_numpy(ohe.fit_transform(index)).float())
    heatmap =F.relu(weigh_maps(grads[0], maps,use_cuda)).data.cpu().numpy()
    print(heatmap.shape)
    heatmap = cv2.resize(heatmap, (896, 896), cv2.INTER_CUBIC)

    probs = F.softmax(out,dim=1).cpu().data[0].numpy()

    return img, heatmap, probs

if __name__ == '__main__':
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    job_number = int(1)
    HOME_PATH = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed/mix/'
    summary_path = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed/'
    log_file_path = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/model/model_torch/model_flatten_linear_layer/train_log{}'.format(job_number)
    model_file_path = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/model/model_torch/model_flatten_linear_layer/Experiment/model_resnet34_dropout0.2_weightdecay/'

    test = pd.read_csv(summary_path + 'test.csv')#.sample(n=50).reset_index() # split train - test set.

    start_test = 0
    tensor_transform_test = transforms.Compose([
                    CenterCrop(896),
                    transforms.ToTensor(),
                    lambda x: x.float(),
                ])
    dataset_test = KneeGradingDataset(test,HOME_PATH,tensor_transform_test,stage = 'test')

    test_loader = data.DataLoader(dataset_test,batch_size=8)
    print('Test data:', len(dataset_test))
    net = resnet34(pretrained=True)
    net.avgpool = nn.AvgPool2d(28, 28)
    net.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 5))  # OULU's paper.
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    print(net)
    # Network
    if USE_CUDA:
        net.load_state_dict(torch.load(model_file_path + '/epoch_4.pth'))
    else:
        net.load_state_dict((torch.load(model_file_path + '/epoch_4.pth',map_location='cpu')))
    net = nn.DataParallel(net)
    if USE_CUDA:
        net.cuda()
    net.eval()

    print('############### Model Finished ####################')
    criterion = F.cross_entropy
    print(test.head())
    file_name = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed/mix/96m/9002817_96m_LEFT_KNEE.hdf5'
    path_name = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed/mix'
    save_dir = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/test/attention_map'
    for idx,each in test.iterrows():
        data = each['File Name']
        print(data)
        klg = int(each['KLG'])
        side = each['Description']
        month = data.split('_')[1]
        file_name = os.path.join(path_name,month,data)
        img, heatmap, probs = gradcam_resnet(file_name, net, tensor_transform_test, use_cuda = USE_CUDA)
        pred = probs.argmax()
        plt.figure(figsize=(7, 7))
        img = np.array(img)
        plt.imshow(img[1, :, :], cmap=plt.cm.Greys_r)
        plt.imshow(heatmap, cmap=plt.cm.jet, alpha=0.3)
        plt.xticks([])
        plt.yticks([])
        plt.title('{} KLG: {}; Prediction: {}'.format(side, klg, pred))
        if pred == klg:
            subfolder = 'correct_pred'
            klg =str(klg)
            if not os.path.exists(os.path.join(save_dir, subfolder,klg, klg)):
                os.makedirs(os.path.join(save_dir, subfolder,klg, klg))
            plt.savefig(os.path.join(save_dir, subfolder, klg, klg,data.replace('.hdf5', '.png')), dpi=300)
        else:
            klg = str(klg)
            pred = str(pred)
            subfolder ='wrong_pred'
            if not os.path.exists(os.path.join(save_dir, subfolder,klg, pred)):
                os.makedirs(os.path.join(save_dir, subfolder, klg, pred))
            plt.savefig(os.path.join(save_dir, subfolder, klg, pred, data.replace('.hdf5', '.png')), dpi=300)


