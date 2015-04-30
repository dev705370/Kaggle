'''
Created on Apr 15, 2015

@author: Devendra
'''

from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from lasagne.nonlinearities import softmax
import matplotlib.pyplot as plt
import numpy as np
from NeuralNetwork import L2Regularization, EarlyStoping
import theano
from AdjustLRandMou import AdjustVariable, float32, AdjustVariableCopy



def Network1(max_epochs = 400, input_size = 93, output_size = 9):
    __layers = [('input', layers.InputLayer), 
                ('hidden1', layers.DenseLayer), 
#                 ('hidden2', layers.DenseLayer), 
                ('output', layers.DenseLayer)]
    return NeuralNet(layers = __layers, 
                    input_shape=(None, input_size),
                    hidden1_num_units=25,
#                     hidden2_num_units=50,
#                     hidden1_nonlinearity=softmax,
                    output_nonlinearity=softmax,
                    output_num_units=output_size,
                        
                     update=nesterov_momentum,
                     update_learning_rate=0.01,
                     update_momentum=0.9,
                     
                     regression=False,  
                     max_epochs=max_epochs,
                     eval_size=0.15, 
                     verbose=1)
    
def Network2(max_epochs = 400, input_size = 93, output_size = 9):
    __layers = [('input', layers.InputLayer), 
                ('conv1', layers.Conv2DLayer),
                ('pool1', layers.MaxPool2DLayer),
                ('conv2', layers.Conv2DLayer),
                ('pool2', layers.MaxPool2DLayer),
                ('conv3', layers.Conv2DLayer),
                ('pool3', layers.MaxPool2DLayer),
                ('hidden1', layers.DenseLayer), 
                ('hidden2', layers.DenseLayer), 
                ('output', layers.DenseLayer)]
    return NeuralNet(layers = __layers, 
                     input_shape=(None, 1, 1, input_size),
                     conv1_num_filters=32, conv1_filter_size=(1, 3), pool1_ds=(1, 2),
                     conv2_num_filters=64, conv2_filter_size=(1, 2), pool2_ds=(1, 2),
                     conv3_num_filters=128, conv3_filter_size=(1, 2), pool3_ds=(1, 2),
                     hidden1_num_units=500,
                     hidden2_num_units=50,
                     output_nonlinearity=softmax,
                     output_num_units=output_size,
                     
                     update=nesterov_momentum,
                     update_learning_rate=0.01,
                     update_momentum=0.9,
                     
                     regression=False,  
                     max_epochs=max_epochs, 
                     verbose=1)        

def Network3(max_epochs = 400, input_size = 93, output_size = 9, prob=0.1):
    __layers = [('input', layers.InputLayer), 
                ('dense1', layers.DenseLayer), 
                ('dropout0', layers.DropoutLayer),
                ('dense2', layers.DenseLayer), 
                ('output', layers.DenseLayer)]
    return NeuralNet(layers = __layers, 
                     input_shape=(None, input_size),
                     dense1_num_units=200,
                     dropout0_p = prob,
                     dense2_num_units=200,
                     output_num_units=output_size,
                     
                     output_nonlinearity=softmax,
                     
                     update=nesterov_momentum,
                     update_learning_rate=theano.shared(float32(0.03)),
                     update_momentum=theano.shared(float32(0.9)),
                     
                     regression=False,  
                     max_epochs=max_epochs, 
                     objective=L2Regularization.l2_regularization,
                     on_epoch_finished=[AdjustVariableCopy('update_learning_rate', start=0.003, stop=0.00001),
                                        AdjustVariableCopy('update_momentum', start=0.9, stop=0.9999),
                                        EarlyStoping.EarlyStopping(patience = 10),
                                        ],
                     verbose=1)
    
def Network4(max_epochs = 400, input_size = 93, output_size = 9):
    __layers = [('input', layers.InputLayer)] 
    args = {'input_shape':(None, input_size),
            'output_nonlinearity':softmax,
            'output_num_units':output_size}
    
#     for i in xrange(5) :
#         layer_name = 'dense'+str(i)
#         __layers.append((layer_name, layers.DenseLayer))
#         args[layer_name+'_num_units'] = 100*(5-i)
    
#     for i in xrange(5) :
#         layer_name = 'dense'+str(i)
#         __layers.append((layer_name, layers.DenseLayer))
#         args[layer_name+'_num_units'] = 20*(5-i)

    layer_name = 'dense0'
    __layers.append((layer_name, layers.DenseLayer))
    args[layer_name+'_num_units'] = 1500
    layer_name = 'dense1'
    __layers.append((layer_name, layers.DenseLayer))
    args[layer_name+'_num_units'] = 1000
    layer_name = 'dense2'
    __layers.append((layer_name, layers.DenseLayer))
    args[layer_name+'_num_units'] = 500
    layer_name = 'dense3'
    __layers.append((layer_name, layers.DenseLayer))
    args[layer_name+'_num_units'] = 100
        
    __layers.append(('output', layers.DenseLayer))

    return NeuralNet(layers = __layers, 
                     
                     update=nesterov_momentum,
                     update_learning_rate=theano.shared(float32(0.03)),
                     update_momentum=theano.shared(float32(0.9)),
                     
                     regression=False,  
                     max_epochs=max_epochs, 
                     objective=L2Regularization.l2_regularization,
                     on_epoch_finished=[AdjustVariableCopy('update_learning_rate', start=0.003, stop=0.000001),
                                        AdjustVariableCopy('update_momentum', start=0.9, stop=0.99999),
                                        EarlyStoping.EarlyStopping(patience = 20),
                                        ],
                     verbose=1,
                     **args)

def Network5(max_epochs = 400, input_size = 93, output_size = 9):
    __layers = [('input', layers.InputLayer), 
                ('dense1', layers.DenseLayer), 
                ('dropout0', layers.DropoutLayer),
                ('dense2', layers.DenseLayer), 
                ('dropout1', layers.DropoutLayer),
                ('dense3', layers.DenseLayer), 
                ('output', layers.DenseLayer)]
    return NeuralNet(layers = __layers, 
                     input_shape=(None, input_size),
                     dense1_num_units=2000,
                     dropout0_p = 0.5,
                     dense2_num_units=1500,
                     dropout1_p = 0.5,
                     dense3_num_units=1000,
                     output_num_units=output_size,
                     
                     output_nonlinearity=softmax,
                     
                     update=nesterov_momentum,
                     update_learning_rate=theano.shared(float32(0.03)),
                     update_momentum=theano.shared(float32(0.9)),
                     
                     regression=False,  
                     max_epochs=max_epochs, 
                     objective=L2Regularization.l2_regularization,
                     on_epoch_finished=[AdjustVariableCopy('update_learning_rate', start=0.003, stop=0.000001),
                                        AdjustVariableCopy('update_momentum', start=0.9, stop=0.99999),
                                        EarlyStoping.EarlyStopping(patience = 20),
                                        ],
                     verbose=1)
    
def Network6(max_epochs = 400, input_size = 93, output_size = 9):
    __layers = [('input', layers.InputLayer), 
                ('dense1', layers.DenseLayer), 
                ('dropout0', layers.DropoutLayer),
                ('dense2', layers.DenseLayer), 
                ('dropout1', layers.DropoutLayer),
                ('dense3', layers.DenseLayer), 
                ('dropout2', layers.DropoutLayer),
                ('dense4', layers.DenseLayer), 
                ('output', layers.DenseLayer)]
    return NeuralNet(layers = __layers, 
                     input_shape=(None, input_size),
                     dense1_num_units=2000,
                     dropout0_p = 0.5,
                     dense2_num_units=1500,
                     dropout1_p = 0.5,
                     dense3_num_units=1000,
                     dropout2_p = 0.5,
                     dense4_num_units=200,
                     output_num_units=output_size,
                     
                     output_nonlinearity=softmax,
                     
                     update=nesterov_momentum,
                     update_learning_rate=theano.shared(float32(0.03)),
                     update_momentum=theano.shared(float32(0.9)),
                     
                     regression=False,  
                     max_epochs=max_epochs, 
                     objective=L2Regularization.l2_regularization,
                     on_epoch_finished=[AdjustVariableCopy('update_learning_rate', start=0.003, stop=0.000001),
                                        AdjustVariableCopy('update_momentum', start=0.9, stop=0.99999),
                                        EarlyStoping.EarlyStopping(patience = 10),
                                        ],
                     verbose=1)   
    
def Network7(max_epochs = 400, input_size = 93, output_size = 9, units = 200):
    __layers = [('input', layers.InputLayer), 
                ('dense1', layers.DenseLayer), 
                ('dropout0', layers.DropoutLayer),
                ('dense2', layers.DenseLayer), 
                ('output', layers.DenseLayer)]
    return NeuralNet(layers = __layers, 
                     input_shape=(None, input_size),
                     dense1_num_units=units,
                     dropout0_p = 0.5,
                     dense2_num_units=200,
                     output_num_units=output_size,
                     
                    output_nonlinearity=softmax,
                     
                     update=nesterov_momentum,
                     update_learning_rate=theano.shared(float32(0.03)),
                     update_momentum=theano.shared(float32(0.9)),
                     
                    eval_size=0.02, 
                     regression=False,  
                     max_epochs=max_epochs, 
                     objective=L2Regularization.l2_regularization,
                     on_epoch_finished=[AdjustVariableCopy('update_learning_rate', start=0.003, stop=0.0000000000000001),
                                        AdjustVariableCopy('update_momentum', start=0.9, stop=0.9999999),
                                        EarlyStoping.EarlyStopping(patience = 50),
                                        ],
                     verbose=1) 

def Network8(max_epochs = 400, input_size = 93, output_size = 9):
    __layers = [('input', layers.InputLayer), 
                ('dense1', layers.DenseLayer), 
                ('gaussian0', layers.GaussianNoiseLayer),
                ('dense2', layers.DenseLayer), 
                ('output', layers.DenseLayer)]
    return NeuralNet(layers = __layers, 
                     input_shape=(None, input_size),
                     dense1_num_units=200,
                     gaussian0_sigma = 0.1,
                     dense2_num_units=200,
                     output_num_units=output_size,
                     
                     output_nonlinearity=softmax,
                     
                     update=nesterov_momentum,
                     update_learning_rate=theano.shared(float32(0.03)),
                     update_momentum=theano.shared(float32(0.9)),
                     
                     regression=False,  
                     max_epochs=max_epochs, 
                     objective=L2Regularization.l2_regularization,
                     on_epoch_finished=[AdjustVariableCopy('update_learning_rate', start=0.003, stop=0.00001),
                                        AdjustVariableCopy('update_momentum', start=0.9, stop=0.9999),
                                        EarlyStoping.EarlyStopping(patience = 20),
                                        ],
                     verbose=1)


class PlotGraph(object):
    def __init__(self, fileName):
        self.fileName = fileName
        self.fig = plt.figure()
        ax = self.fig.add_subplot(111)
        self.li1 = ax.plot([], linewidth=3, label="train")
        self.li2 = ax.plot([], linewidth=3, label="valid")
        plt.grid()
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.yscale("log")
#         plt.savefig(fileName + '.png')
        self.fig.canvas.draw()
        plt.ion()
        plt.show()

    def __call__(self, nn, train_history):
        train_loss = np.array([i["train_loss"] for i in nn.train_history_])
        valid_loss = np.array([i["valid_loss"] for i in nn.train_history_])
        self.li1.set_data(train_loss)
        self.li2.set_data(valid_loss)
        self.fig.canvas.draw()
        plt.savefig(self.fileName + '.png')