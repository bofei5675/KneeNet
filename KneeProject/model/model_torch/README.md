7/9
ResNet + CBAM + epoch 9
############### Model Finished ####################
Confusion Matrix:
 [0.75553191 0.63491379 0.60947003 0.81700405 0.94405594]
Confusion Matrix:
 [[3551 1070   72    7    0]
 [ 534 1473  294   19    0]
 [ 113  586 1403  200    0]
 [   0   26  128 1009   72]
 [   0    0    1   23  405]]
Oulu Acc 0.7522
[0.75553191 0.63491379 0.60947003 0.81700405 0.94405594]
Test losses 0.7667353523479712; Test mse [0.3545]; Test acc [0.7522]; Test Kappa 0.8736;
Testing took: 1213.7639863491058 seconds

ResNet + epoch 4.
############### Model Finished ####################
Confusion Matrix:
 [[3569  891  211   29    0]
 [ 630  957  642   91    0]
 [ 113  272 1507  407    3]
 [   2   10   62 1104   57]
 [   0    0    2   27  400]]
Vanilla Accuracy:0.6860549790642636; Oulu Acc 0.7306
Test losses 0.7737671552699915; Test mse [0.4539]; Test acc [0.6860549790642636]; Test Kappa 0.8471;
Training took: 859.1213908195496 seconds


ResNet
6/24
Job Number 121 to normal queue
DenseNet121
Job Number 122 to short queue
DenseNet 121

Job Number 161 to normal queue
DenseNet 161

Job Number to 161 to short queue
DenseNet 161

6/21
ResNet 34 + ImageNet weight + Entorpy loss(beta = 0.1) + CE
Training for 7 epochs
############### Model Finished ####################
Confusion Matrix:
 [[1783  817  179   22    1]
 [ 310  579  270   67    1]
 [  86  256  971  336    4]
 [   5   14   57  727   65]
 [   0    0    4   21  207]]
Vanilla Accuracy:0.6291654379239162; Oulu Acc 0.6851
Test losses 1.4238149416369368; Test mse [0.5627]; Test acc [0.6291654379239162]; Test Kappa 0.8086;
Training took: 897.1893000602722 seconds

6/20

ResNet 34 + ImageNet wights + CBAM
2 epochs
Epoch 2: Train Loss (0.7739429858086346, 0.6727409198949168)
[[1147  116  146    2    0]
 [ 302  138  268    8    1]
 [  53   27  609   57    4]
 [   1    2   58  320   13]
 [   0    0    1   17   96]]
Epoch 2: Val Loss 0.7506751835275933; Val Acc 0.6943; Val MSE 0.5168; Val Kappa 0.8212;

Epoch 4
############### Model Finished ####################
Confusion Matrix:
 [[2315  263  219    5    0]
 [ 527  282  395   23    0]
 [ 125  149 1147  232    0]
 [   4    1   82  753   28]
 [   6    0    5   29  192]]
Oulu Acc 0.689
Test losses 0.7396732236198573; Test mse [0.4975]; Test acc [0.689]; Test Kappa 0.8344;

model_torch - store all model wrote by torch
    model_flatten_linear_layer - since the image size changed, this file handle this issue by changing linear layer to larger one.



Experiments:

5/20 ResNet34 with changed linear layer
Test losses 1.7898125279694796; Test mse [0.7785]; Test acc [0.5648]; Test Kappa 0.7545;
Training took: 967.7842710018158 seconds
This file has been discarded because the fully connected layer is wrong.


5/27
############### Model Finished ####################
[[50  0 37  0  0]
 [13  0 16  0  0]
 [21  0 23  0  0]
 [11  0 22  0  0]
 [ 3  0  4  0  0]]
73.0 200
0.2195
Test losses 1.427018610239029; Test mse [2.23]; Test acc [0.365]; Test Kappa 0.1492;
Training took: 33.30069303512573 seconds

Job number 1
'''
net = resnet34(pretrained=True)
        net.avgpool = nn.AvgPool2d(28,28)
        net.fc = nn.Linear(512,5)
        net = net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=0.001,weight_decay=1e-4)
'''

