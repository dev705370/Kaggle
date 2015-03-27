'''
Created on Mar 22, 2015

@author: Devendra
'''

from sklearn import svm, cross_validation, gaussian_process
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.decomposition import FastICA

class Compare():
    def __init__(self, _path, trainData):
        self.__path = _path
        self.__trainData = trainData
        
    def run(self):
        X = self.__trainData.values
        Y = X[:, -1]
        X = X[:, :-1]
#         fica = FastICA()
#         X = fica.fit_transform(X)
#         self.compareAll(X, Y)
#         self.plotSVM(X, Y)
#         self.plotKneighbours(X, Y)
#         self.plotExtraTree(X, Y)
        self.plotAdaBoost(X, Y)
#         self.plotGradientBoosting(X, Y)
#         self.plotRandomForest(X, Y)
#         self.logisticReg(X, Y)
#         self.bernoulliRBM(X, Y)

    def plotSVM(self, X, Y):
        meanScore = []
        x = []
        for i in range(1, 11) :
            start = time.clock()
            clf = svm.SVC(kernel='poly', degree=i)
            score = cross_validation.cross_val_score(clf, X, Y, n_jobs = -1)
            meanScore.append(np.mean(score))
            x.append(i)
            print 'completed for ', i, ' and score=', np.mean(score), ' in ', time.clock() - start, ' secs'
        plt.Figure()
        plt.plot(x, meanScore, 'green')
        plt.scatter(x, meanScore, s = 5, c = 'red')
        plt.savefig('SVM_poly_degree_1_10.png')
        plt.close()
        
    def logisticReg(self, X, Y):    
        startTime = time.clock()
        print 'running logistic Regr'
        lr = LogisticRegression(tol=0.00001)
        score = cross_validation.cross_val_score(lr, X, Y, n_jobs = -1)
        print 'Logistic Regression score = ', np.mean(score), ' and time=', time.clock() - startTime, ' secs'
        
    
    def bernoulliRBM(self, X, Y):
        startTime = time.clock()
        print 'running bernoulli RBM'
        lr = LogisticRegression()
        br = BernoulliRBM()
        classifier = Pipeline(steps=[('br', br), ('lr', lr)])
        br.n_components = 100
        score = cross_validation.cross_val_score(classifier, X, Y, n_jobs = -1)
        print 'Bernoulli RBM score = ', np.mean(score), ' and time=', time.clock() - startTime, ' secs'
        
        
    def compareAll(self, X, Y):
        startTime = time.clock()
        print 'running svm'
        clf = svm.SVC(cache_size = 2000)
        score = cross_validation.cross_val_score(clf, X, Y, n_jobs = -1)
        t = time.clock() - startTime
        print 'svm score = ', np.mean(score), ' and time=', t, ' secs'
         
        startTime = time.clock()
        print 'running Decision Tree'
        dtc = DecisionTreeClassifier()
        score = cross_validation.cross_val_score(dtc, X, Y, n_jobs = -1)
        print 'Decision Tree score = ', np.mean(score), ' and time=', time.clock() - startTime, ' secs'
        
        startTime = time.clock()
        print 'running K neighbor classifier'
        knc = KNeighborsClassifier(n_neighbors = 9)
        score = cross_validation.cross_val_score(knc, X, Y, n_jobs = -1)
        print 'K neighbor classifier score = ', np.mean(score), ' and time=', time.clock() - startTime, ' secs'
        
        startTime = time.clock()
        print 'running Random forest classifier'
        rfc = RandomForestClassifier(n_jobs = -1)
        score = cross_validation.cross_val_score(rfc, X, Y, n_jobs = -1)
        print 'Random forest classifier score = ', np.mean(score), ' and time=', time.clock() - startTime, ' secs'   
        
        startTime = time.clock()
        print 'running Extra tree classifier'
        etc = ExtraTreesClassifier(n_jobs = -1)
        score = cross_validation.cross_val_score(etc, X, Y, n_jobs = -1)
        print 'Extra tree classifier score = ', np.mean(score), ' and time=', time.clock() - startTime, ' secs'   
        
        startTime = time.clock()
        print 'running Ada boost classifier'
        abc = AdaBoostClassifier()
        score = cross_validation.cross_val_score(abc, X, Y, n_jobs = -1)
        print 'Ada boost classifier score = ', np.mean(score), ' and time=', time.clock() - startTime, ' secs'   
        
#         startTime = time.clock()
#         print 'running Gradient boosting classifier'
#         gbc = GradientBoostingClassifier(max_depth = 100)
#         score = cross_validation.cross_val_score(gbc, X, Y, n_jobs = -1)
#         print 'Gradient boosting classifier score = ', np.mean(score), ' and time=', time.clock() - startTime, ' secs'  
         
#         startTime = time.clock()
#         print 'running Gaussian'
#         gp = gaussian_process.GaussianProcess(regr = 'quadratic')
#         score = cross_validation.cross_val_score(gp, X, Y, n_jobs = -1)
#         print 'Gaussian score = ', np.mean(score), ' and time=', time.clock() - startTime, ' secs'

    def plotKneighbours(self, X, Y):
        meanScore = []
        x = []
        for i in range(1, 31) :
            start = time.clock()
            knc = KNeighborsClassifier(n_neighbors = i)
            score = cross_validation.cross_val_score(knc, X, Y, n_jobs = -1)
            meanScore.append(np.mean(score))
            x.append(i)
            print 'completed for ', i, ' and score=', np.mean(score), ' in ', time.clock() - start, ' secs'
        plt.Figure()
        plt.plot(x, meanScore, 'green')
        plt.scatter(x, meanScore, s = 5, c = 'red')
        plt.savefig('Kneighbours.png')
        plt.close()
        
    def plotRandomForest(self, X, Y):
        print 'running randomForest'
        meanScore = []
        x = []
        for i in range(1, 11) :
            start = time.clock()
            rfc = RandomForestClassifier(n_estimators=i*500, )
            score = cross_validation.cross_val_score(rfc, X, Y, n_jobs = -1)
            meanScore.append(np.mean(score))
            x.append(i*500)
            print 'completed for ', i, ' and score=', np.mean(score), ' in ', time.clock() - start, ' secs'
        plt.Figure()
        plt.plot(x, meanScore, 'green')
        plt.scatter(x, meanScore, s = 5, c = 'red')
        plt.savefig('RandomForest_500.png')
        plt.close()
        
    def plotExtraTree(self, X, Y):
        print 'running extratree'
        meanScore = []
        x = []
        for i in range(1, 11) :
            start = time.clock()
            etc = ExtraTreesClassifier(n_estimators= i * 500, n_jobs = -1)
            score = cross_validation.cross_val_score(etc, X, Y)
            meanScore.append(np.mean(score))
            x.append(i*500)
            print 'completed for ', i, ' and score=', np.mean(score), ' in ', time.clock() - start, ' secs'
        plt.Figure()
        plt.plot(x, meanScore, 'green')
        plt.scatter(x, meanScore, s = 5, c = 'red')
        plt.savefig('ExtraTree_500.png')
        plt.close()
        
    def plotAdaBoost(self, X, Y):
        print 'running adaboost'
        meanScore = []
        x = []
        for i in range(1, 11) :
            start = time.clock()
            abc = AdaBoostClassifier(learning_rate=(1.0 / (i*10.0)))
            score = cross_validation.cross_val_score(abc, X, Y, n_jobs = -1)
            meanScore.append(np.mean(score))
            x.append(1.0 / (i*10.0))
            print 'completed for ', i, ' and score=', np.mean(score), ' in ', time.clock() - start, ' secs'
        plt.Figure()
        plt.plot(x, meanScore, 'green')
        plt.scatter(x, meanScore, s = 10, c = 'red')
        plt.savefig('AdaBoost_learningrate.png')
        plt.close()
        
    def plotGradientBoosting(self, X, Y):
        print 'running GradientBoosting'
        meanScore = []
        x = []
        for i in range(1, 2) :
            start = time.clock()
            gbc = GradientBoostingClassifier(n_estimators=i*100)
            score = cross_validation.cross_val_score(gbc, X, Y, n_jobs = -1)
            meanScore.append(np.mean(score))
            x.append(1.0 / (i * 10.0))
            print 'completed for ', i, ' and score=', np.mean(score), ' in ', time.clock() - start, ' secs'
        plt.Figure()
        plt.plot(x, meanScore, 'green')
        plt.scatter(x, meanScore, s = 5, c = 'red')
        plt.savefig('GradientBoosting.png')
        plt.close()