'''
Created on Apr 20, 2015

@author: Devendra
'''
import numpy as np

def float32(k):
    return np.cast['float32'](k)

class AdjustVariable(object):
    def __init__(self, name, step = 20):
        self.name = name
        self.step = step

    def __call__(self, nn, train_history):
        epoch = train_history[-1]['epoch']
        p = (epoch // self.step) + 1
        if self.name == 'update_learning_rate' :
            new_value = float32(10**(-p))
        else :
            new_value = float32(((10**p) - 1) * (10**(-p)))
        getattr(nn, self.name).set_value(new_value)
    

class AdjustVariableCopy(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)