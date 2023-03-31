import numpy as np

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        '''
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels,  self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None
    
    def forward(self, x):
        '''
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        '''
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################
        w, b = self.weight, self.bias
        stride, pad = self.stride, self.padding
        out_ch, C, HH, WW = w.shape
        x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad))) 
        N, C, H, W = x.shape
        
        new_H = int(1 + (H + 2 * pad- HH) / stride)
        new_W = int(1 + (W + 2 * pad - WW) / stride)
        out = np.zeros((N, out_ch, new_H, new_H))
        
        for image in range(N):
          for kernel in range(out_ch):
            for a in range(0, new_H):
              for b in range(0, new_W):
                aa = a*stride
                bb = b*stride
                out[image, kernel, a , b ] = np.sum(x_padded[image, :, aa : aa + HH, bb : bb + WW]* self.weight[kernel, :, :, :]) + self.bias[kernel]        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x , new_H, new_W
        return out

    def backward(self, dout):
        '''
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        '''
        x, new_H, new_W = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################
        stride, pad = self.stride, self.padding
        N, C, H, W = x.shape
        out_ch, C, HH, WW = self.weight.shape
        dx = np.zeros_like(x)
        dw = np.zeros_like(self.weight)
        db = np.zeros_like(self.bias)
        x_pad = np.pad(x, [(0,0), (0,0), (pad,pad), (pad,pad)])
        padded_dx = np.pad(dx, [(0,0), (0,0), (pad,pad), (pad,pad)])

        for image in range(N):
          for kernel in range(out_ch):
            for h in range(new_H):
              hh = h * stride
              for w in range(new_W):
                ww = w * stride
                filt = x_pad[image, :, hh:hh+HH, ww:ww+WW]
                db[kernel] += dout[image, kernel, h, w]
                dw[kernel] += filt*dout[image, kernel, h, w]
                padded_dx[image, :, hh:hh+HH, ww:ww+WW] += self.weight[kernel] * dout[image, kernel, hh, ww]

        dx = padded_dx[:, :, pad:pad+H, pad:pad+W]
        #dx = padded_dx[:, :, 1:-1, 1:-1]
        self.dx = dx
        self.dw = dw
        self.db = db
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################