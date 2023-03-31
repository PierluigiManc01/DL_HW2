from ._base_optimizer import _BaseOptimizer
class SGD(_BaseOptimizer):
    def __init__(self, model, learning_rate=1e-4, reg=1e-3, momentum=0.9):
        super().__init__(model, learning_rate, reg)
        self.momentum = momentum
        self.lr = learning_rate       
        self.velocity_w = {}
        self.velocity_b = {}
        for idx, m in enumerate(model.modules):
            if hasattr(m, 'weight'):
              self.velocity_w[m] = [0.0 for i in range(m.weight.shape[0]) for j in range(m.weight.shape[1])]
            if hasattr(m, 'bias'):
              self.velocity_b[m] = [0.0 for i in range(m.bias.shape[0])]
        # initialize the velocity terms for each weight

    def update(self, model):
        '''
        Update model weights based on gradients
        :param model: The model to be updated
        :return: None, but the model weights should be updated
        '''
        #self.apply_regularization(model)

        for idx, m in enumerate(model.modules):
            if hasattr(m, 'weight'):
                
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for weights                                        #
                #############################################################################
                grad_w = m.dw
                self.velocity_w[m] = [self.momentum * velocity - self.lr *grad_w_i for velocity,grad_w_i in zip(self.velocity_w[m],grad_w)]
                m.weight += self.velocity_w[m]
                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################
            if hasattr(m, 'bias'):
                #############################################################################
                # TODO:                                                                    #
                #    1) Momentum updates for bias                                           #
                #############################################################################
                grad_b = m.db
                self.velocity_b[m] = [self.momentum * velocity - self.lr *grad_b_i for velocity,grad_b_i in zip(self.velocity_b[m],grad_b)]
                m.bias += self.velocity_b[m]
                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################
