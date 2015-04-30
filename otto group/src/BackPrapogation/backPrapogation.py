# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 22:50:33 2014

@author: Devendra
"""
import numpy as np
import random
import math
import sys
import time

class BadTrainingSetException(Exception) :
    def _init_(self, inputSize, outputSize) :
        self.inputSize = inputSize
        self.outputSize = outputSize
    
    def _str_(self) :
        return "Not a good training Set because " + repr(self.inputSize) + " != " + repr(self.outputSize)
    

def randomNumber(a, b) :
    return a+(b-a)*random.random();

def randomMatrix(row, column) :
    mat = np.empty([row, column])
    for i in xrange(row):
        for j in xrange(column) :
            mat[i][j] = randomNumber(-0.5, 0.5)
    return mat;

def intialWeightList(inputSize, layers, layerSize, outputSize) :
    wList = []
    wList.append(randomMatrix(layerSize, inputSize))
    for i in xrange(layers - 1) :
        wList.append(randomMatrix(layerSize, layerSize+1))
    wList.append(randomMatrix(outputSize, layerSize+1))
    return wList
    
def intialWeightDifferenceList(wList) :
    wDifferenceList = [np.zeros_like(a) for a in wList]
    return wDifferenceList
    
def segmoid(x) :
    try :
        return 1 / (1 + math.exp(-x))
    except OverflowError :
        print "Overflow error ", x
        raise
        

class BackPrapogation:
    Layers = 1
    LayerSize = 5
    iteration = 1000
    minError = 10
    
    def __init__(self, Layers = 1, LayerSize = 5, iteration = -1, minError = 10, learningRate = 0.01, activationFunction = segmoid):
        self.Layers = Layers
        self.LayerSize = LayerSize
        self.iteration = iteration
        self.minError = minError
        self.activationFunction = activationFunction
        self.learningRate = learningRate
        
    def appendBias(self, x) :
        ai = np.ones(x.shape[0] + 1)
        for i in xrange(x.shape[0]) :
            ai[i+1] = x[i]
        return ai
        
    def forwardProp(self, x) :
        aList = []
        ai = self.appendBias(x)
        aList.append(ai)
        for i in xrange(self.Layers) :
            ao = np.dot(self.wList[i], ai)
            try :
                ao[:] = [self.activationFunction(l) for l in ao]
            except OverflowError :
                print "Overflow error ", ao
                raise
            ai = self.appendBias(ao)
            aList.append(ai)
        ao = np.dot(self.wList[self.Layers], ai)
        ao[:] = [self.activationFunction(l) for l in ao]
        aList.append(ao)
        return aList
        
    def calculateError(self, expected, actual) :
        errorVector = expected - actual
        return np.sum(0.5 * errorVector**2)
        
    def derivativeOfSigmoidFuntion(self, z) :
        return (1-z)*z
        
    def dotProductOneDArray(self, A, B) :
        return np.dot(A.reshape(A.shape[0], -1), B.reshape(-1, B.shape[0]))
        
    def removeBias(self, x) :
        a = np.ones(x.shape[0] - 1)
        for i in xrange(a.shape[0]) :
            a[i] = x[i+1]
        return a
        
    def backProp(self, aList, y) :
        delta = aList[self.Layers+1] - y
        for i in xrange(self.Layers + 1) :
            ai = aList[self.Layers - i]
            weight = self.wList[self.Layers - i]
            weightDifference = self.wDifferenceList[self.Layers - i]
            dp = self.dotProductOneDArray(delta, ai)
            weightDifference = weightDifference + dp
            self.wDifferenceList[self.Layers - i] = weightDifference
            if i != self.Layers :
                preDelta = np.dot(weight.transpose(), delta) * self.derivativeOfSigmoidFuntion(ai)
                delta = self.removeBias(preDelta) #remove bais factor from it
        
    def runAlgo(self, m) :
        error = 0.0
        self.wDifferenceList = intialWeightDifferenceList(self.wList)
        for i in xrange(m) :
            x = self.inputData[i]
            y = self.outputData[i]
            aList = self.forwardProp(x)
            self.backProp(aList, y)
            error = error + self.calculateError(y, aList[self.Layers+1]) 
        for i in xrange(len(self.wList)) :
            self.wList[i] = self.wList[i] - (self.learningRate / m) * self.wDifferenceList[i]
        return error
    
    def fit(self, X, Y):
        self.inputData = X
        self.outputData = Y
        self.features = X.shape[1]
        m = X.shape[0]
        try :
            self.outputSize = Y.shape[1]
        except IndexError :
            self.outputSize = 1
        if m != Y.shape[0] :
            raise BadTrainingSetException(m, Y.shape[0])
        self.wList = intialWeightList(self.features + 1, self.Layers, self.LayerSize, self.outputSize)
        prevError = sys.maxint
        i = 0
        while True :
            start = time.clock()
            error = self.runAlgo(m)
            if prevError - error > self.minError:
                prevError = error
                print "iteration=", i+1, " with error=", error, " in ", time.clock() - start
            else:
                print "Terminating at iteration ", i+1
                break
            i += 1
            if i == self.iteration :
                print "Terminating because of max iteration"
#            if error < self.minError :
#                break;
        
    def score(self, X, Y) :
        error = 0
        for i in xrange(X.shape[0]) :
            x = X[i]
            y = Y[i]
            aList = self.forwardProp(x)
            error = error + self.calculateError(y, aList[self.Layers+1])
        return 1 - error / X.shape[0]
    
    def predict(self, X) :
        Y = []
        for i in xrange(X.shape[0]) :
            ai = self.appendBias(X[i])
            for j in xrange(self.Layers) :
                try:
                    ao = np.dot(self.wList[j], ai.reshape(ai.shape[0], 1))
                    ao = ao.reshape(ao.shape[0])
                except ValueError:
                    print self.wList[j].shape, " ", ao.shape
                    raise
                ao[:] = [self.activationFunction(l) for l in ao]
                ai = self.appendBias(ao)
            ao = np.dot(self.wList[self.Layers], ai)
            ao[:] = [self.activationFunction(l) for l in ao]
            Y.append(ao)
        return np.array(Y)