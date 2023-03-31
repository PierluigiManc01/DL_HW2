import torch
import torch.nn as nn

class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        #############################################################################
        # TODO: Initialize the Vanilla CNN                                          #
        #       Conv: 7x7 kernel, stride 1 and padding 0                            #
        #       Max Pooling: 2x2 kernel, stride 2                                   #
        #############################################################################
        self.conv = nn.Conv2d(in_channels = 3 ,out_channels=32, kernel_size = 7,stride = 1)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2,stride = 2)
        self.fc = nn.Linear(in_features=13*13*32,out_features = 10)
        ## dimensional considerations
        # image dim : 32x32x3 -> 32-7//1 + 1 = 26
        # conv out dim : 26x26x32 -> 26-2//2 + 1 = 13
        # pool out dim : 13x13x32 
        # linar input : 13x13x32
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        N = x.shape[0]
        outs = self.conv(x)
        outs = self.relu(outs)
        outs = self.max_pool(outs)
        outs = self.fc(outs.view(N,-1))
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return outs