from model.model import *
from model.dataloader import *
from model.train_utils import *
import pandas as pd
import torch.utils.data as data
import torchvision.transforms as transforms
import time
import torch.optim as optim
print('Start training')

model_dir = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/KneeDetection/saved/epoch_45.pth'
test_contents = '/gpfs/data/denizlab/Users/bz1030/data/bounding_box/test.csv'
test_df = pd.read_csv(test_contents)#.sample(n = 4).reset_index()
try:
    test_df.drop(['index'],axis = 1, inplace = True)
except KeyError:
    pass
USE_CUDA = torch.cuda.is_available()
tensor_transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    lambda x: x.float(),
                ])

dataset_test = KneeDetectionDataset(test_df,tensor_transform_train,stage = 'test')

test_loader = data.DataLoader(dataset_test,batch_size=16)

net = ResNet(pretrained = True,dropout = 0.2,use_cuda = USE_CUDA)

if USE_CUDA:
    net.load_state_dict(torch.load(model_dir))
else:
    net.load_state_dict(torch.load(model_dir,map_location='cpu'))
print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))

optimizer = optim.Adam(net.parameters(),lr=1e-4,weight_decay=1e-4)

criterion = MSELoss()

val_loss, all_names, all_labels, all_preds = validate_epoch(net,criterion,test_loader,USE_CUDA)

all_labels = np.vstack(all_labels)
all_preds = np.vstack(all_preds)
print(all_labels.shape, all_preds.shape)
df = [all_labels, all_preds]
df = np.hstack(df)
df = pd.DataFrame(df)
print(df.shape)
df['file_path'] = all_names
df.to_csv('test_output.csv',index=False)
print('Val Loss {}'.format(val_loss))

