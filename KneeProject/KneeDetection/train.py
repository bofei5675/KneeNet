from model.model import *
from model.dataloader import *
from model.train_utils import *
import pandas as pd
import torch.utils.data as data
import torchvision.transforms as transforms
import time
import torch.optim as optim
import os
print('Start training')
model_dir = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/KneeDetection/saved'
log_dir = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/KneeDetection/log.txt'
train_contents = '/gpfs/data/denizlab/Users/bz1030/data/bounding_box/train.csv'
train_df = pd.read_csv(train_contents)#.sample(n = 4).reset_index()
val_contents = '/gpfs/data/denizlab/Users/bz1030/data/bounding_box/val.csv'
val_df = pd.read_csv(val_contents)#.sample(n = 2).reset_index()
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
try:
    train_df.drop(['index'],axis = 1, inplace = True)
    val_df.drop(['index'],axis = 1, inplace = True)
except KeyError:
    pass
USE_CUDA = torch.cuda.is_available()
tensor_transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    lambda x: x.float(),
                ])

dataset_train = KneeDetectionDataset(train_df,tensor_transform_train,stage = 'train')
dataset_val = KneeDetectionDataset(val_df,tensor_transform_train,stage = 'val')

train_loader = data.DataLoader(dataset_train,batch_size=16)
val_loader = data.DataLoader(dataset_val,batch_size=8)

net = ResNet(pretrained = True,dropout = 0.2,use_cuda = USE_CUDA)

print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))

optimizer = optim.Adam(net.parameters(),lr=1e-4,weight_decay=1e-4)

criterion = MSELoss()

eval_iterations = 1000
train_iterations(net,optimizer,train_loader,val_loader,criterion,50,USE_CUDA,eval_iterations,log_dir,model_dir)






