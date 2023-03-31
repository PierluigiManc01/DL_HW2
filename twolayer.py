import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        '''
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        '''
        super(TwoLayerNet, self).__init__()
        #############################################################################
        # TODO: Initialize the TwoLayerNet, use sigmoid activation between layers   #
        #############################################################################
        self.input_dim, self.hidden_size, self.num_classes = input_dim, hidden_size, num_classes
        self.lin1 = nn.Linear(self.input_dim,self.hidden_size)
        self.sig = nn.Sigmoid()
        self.lin2 = nn.Linear(self.hidden_size, self.num_classes)
        self.softm = nn.Softmax(dim = 1)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
    def forward(self, x):
        out = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        lin1 = self.lin1(x.view(x.shape[0],-1))
        out1 = self.sig(lin1)
        lin2 = self.lin2(out1)
        out = self.softm(lin2)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out