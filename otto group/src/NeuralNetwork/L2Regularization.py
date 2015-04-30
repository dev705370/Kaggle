'''
Created on Apr 20, 2015

@author: Devendra
'''
from nolearn.lasagne import Objective
import lasagne

class l2_regularization(Objective):
    def get_loss(self, input=None, target=None, **kwargs):
        loss = super(l2_regularization, self).get_loss(input=input, target=target, **kwargs)
        reg = lasagne.regularization.l2(self.input_layer)
        return loss + reg * 0.00000000000001