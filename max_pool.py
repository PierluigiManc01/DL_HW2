import numpy as np

class MaxPooling:
    '''
    Max Pooling of input
    '''
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        '''
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        '''
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        stride = self.stride
        N, C, H, W = x.shape
        pool_h , pool_w = self.kernel_size , self.kernel_size 
        H_out = int(1 + (H - pool_h) / stride)
        W_out = int(1 + (W - pool_w) / stride)
        out = np.zeros((N, C, H_out, W_out))
        for h in range(0, H_out):      
          for w in range(0, W_out ):
            hh = h* stride 
            ww = w* stride
            pooling_region = x[:, :, hh : hh + pool_h, ww : ww + pool_w]
            max_vals = np.amax(pooling_region,axis = (2,3)) 
            out[:, :, h, w] = max_vals
       # for image in range(N):
        #  for kernel in range(C):
         #   for h in range(0, H_out):
          #      
           #   for w in range(0, W_out ):
            #    hh = h* stride 
             #   ww = w* stride
              #  pooling_region = x[image, kernel, hh : hh + pool_height, ww : ww + pool_width]
               # out[image, kernel, h, w] = np.max(pooling_region)        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        '''
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return:
        '''
        x, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                     #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################
        stride = self.stride
        N, C, H, W = x.shape
        pool_h , pool_w = self.kernel_size , self.kernel_size 
        dx = np.zeros((N, C, H, W))
      
        for image in range(N):
          for kernel in range(C):
            for h in range(0, H_out):
              hh = h*stride
              for w in range(0, W_out ):
                ww = w*stride
                ind1,ind2 = np.where(np.max(x[image,kernel,hh : hh + pool_h, ww : ww + pool_w]) == x[image,kernel,hh : hh + pool_h, ww : ww + pool_w])
                dx[image, kernel, hh : hh + pool_h, ww : ww + pool_w][ind1[0],ind2[0]] = dout[image, kernel,h,w]
             
      #  for h in range(0, H_out):
       #       hh = h*stride
        #      for w in range(0, W_out ):
         #       ww = w*stride
          #      inds = np.unravel_index(max_inds,x.shape[1:3])
                #ind1,ind2 = np.where(np.max(x[:,:,hh : hh + pool_h, ww : ww + pool_w]) == x[:,:,hh : hh + pool_h, ww : ww + pool_w])
           #     dx[:, :, hh : hh + pool_h, ww : ww + pool_w][inds[0],inds[1]] = dout[:, :,h,w]
        self.dx = dx
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
