import torch
import torch.nn as nn


class MyModel(nn.Module):
    # You can use pre-existing models but change layers to recieve full credit.
    def __init__(self):
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################
        self.conv1 = nn.Conv2d(in_channels = 3,out_channels= 256, kernel_size = 5, stride = 1)
        self.bn1 = nn.BatchNorm2d(256,track_running_stats=False)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(in_channels = 256,out_channels= 128, kernel_size = 2, stride = 1)
        self.bn2 = nn.BatchNorm2d(128,track_running_stats=False)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv3 = nn.Conv2d(in_channels = 128,out_channels= 64, kernel_size = 2, stride = 1)
        self.bn3 = nn.BatchNorm2d(64,track_running_stats=False)
        self.relu = nn.ReLU(inplace = True)
        self.fc = nn.Linear(in_features=5*5*64,out_features = 10*10*32)
        self.fc2 = nn.Linear(in_features = 10*10*32, out_features = 10)
        ## dimensional considerations
        # image dim : 32x32x3 -> 32-5//1 + 1 = 28
        # conv out dim : 28x28x256 -> 28-2//2 + 1 = 14
        # pool out dim : 14x14x256 -> 14-2 // 1 + 1 = 13
        # conv2 out dim : 13x13x128 -> 13-2//2 + 1 = 6
        # pool cout dim: 6x6x128 
        # conv2 out dim : 6x6x64 -> 6-2//1 + 1 = 5
        #linear input : 12x12x128 -> 10*10*32
        #linear2 input : 10*10*32
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        outs = self.conv1(x)
        outs = self.bn1(outs)
        outs = self.maxpool1(outs)
        outs = self.relu(outs)
        outs = self.conv2(outs)
        outs = self.bn2(outs)
        outs = self.maxpool2(outs)
        outs = self.relu(outs)
        outs = self.conv3(outs)
        outs = self.bn3(outs)
        outs = self.fc(outs.view(x.shape[0],-1))
        outs = self.fc2(outs.view(x.shape[0],-1))
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs