'''
Created on Mar 22, 2015

@author: Devendra
'''

from sklearn import svm, cross_validation, gaussian_process
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.decomposition import FastICA
from sklearn.preprocessing import Binarizer
from sklearn.metrics import make_scorer
from scoring import Gini
import datetime
import time
from sklearn.ensemble.forest import ExtraTreesRegressor

class Compare():
    def __init__(self, _path = None, trainData = None, n_jobs = -1):
        self.__path = _path
        self.__trainData = trainData
        self.n_jobs = n_jobs
        
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
#         self.plotAdaBoost(X, Y)
#         self.plotGradientBoosting(X, Y)
        self.plotRandomForest(X, Y)
#         self.logisticReg(X, Y)
#         self.bernoulliRBM(X, Y)


    def printLine(self, i, score, start):
        print 'completed for ', i, ' and score=', np.mean(score), ' in ', time.clock() - start, ' secs at ', datetime.datetime.now()
        print score

    def savePic(self, plot, name):
        try :
            plot.savefig(name)
        except :
            print "unable to save, close ", name
            time.sleep(2)
            self.savePic(plot, name)

    def plotSVM(self, X, Y):
        meanScore = []
        x = []
        print "running svm"
        for i in range(1, 11) :
            start = time.clock()
            clf = svm.SVC(kernel='poly', degree=i, verbose=False)
            score = cross_validation.cross_val_score(clf, X, Y, scoring=make_scorer(Gini), cv = 5, n_jobs = self.n_jobs)
            meanScore.append(np.mean(score))
            x.append(i)
            self.printLine(i, score, start)
            plt.Figure()
            plt.plot(x, meanScore, 'green')
            plt.scatter(x, meanScore, s = 5, c = 'red')
            self.savePic(plt, 'resrc/SVM_poly_degree_1_10.png')
            plt.close()
        
    def plotSVR(self, X, Y):
        meanScore = []
        x = []
        print "running svr"
        for i in range(1, 11) :
            start = time.clock()
            clf = svm.SVR(kernel='poly', degree=i, verbose=True)
            score = cross_validation.cross_val_score(clf, X, Y, scoring=make_scorer(Gini), cv = 5, n_jobs = self.n_jobs)
            meanScore.append(np.mean(score))
            x.append(i)
            self.printLine(i, score, start)
            plt.Figure()
            plt.plot(x, meanScore, 'green')
            plt.scatter(x, meanScore, s = 5, c = 'red')
            self.savePic(plt, 'resrc/SVR_poly_degree_1_10.png')
            plt.close()
        
    def logisticReg(self, X, Y):    
        startTime = time.clock()
        print 'running logistic Regr'
        lr = LogisticRegression(tol=0.00001)
        score = cross_validation.cross_val_score(lr, X, Y, n_jobs = self.n_jobs)
        print 'Logistic Regression score = ', np.mean(score), ' and time=', time.clock() - startTime, ' secs'
           
    def bernoulliRBM(self, X, Y):
        startTime = time.clock()
        print 'running bernoulli RBM'
        br = Binarizer().fit(X)
        X = br.transform(X)
        print 'x transformed'
        lr = LogisticRegression()
        br = BernoulliRBM(learning_rate = 0.01, n_iter = 5000, batch_size=1000, verbose=True)
        classifier = Pipeline(steps=[('br', br), ('lr', lr)])
        score = cross_validation.cross_val_score(classifier, X, Y, n_jobs = self.n_jobs)
        print 'Bernoulli RBM score = ', np.mean(score), ' and time=', time.clock() - startTime, ' secs'
              
    def compareAll(self, X, Y):
        startTime = time.clock()
        print 'running svm'
        clf = svm.SVC(cache_size = 2000)
        score = cross_validation.cross_val_score(clf, X, Y, n_jobs = self.n_jobs)
        t = time.clock() - startTime
        print 'svm score = ', np.mean(score), ' and time=', t, ' secs'
         
        startTime = time.clock()
        print 'running Decision Tree'
        dtc = DecisionTreeClassifier()
        score = cross_validation.cross_val_score(dtc, X, Y, n_jobs = self.n_jobs)
        print 'Decision Tree score = ', np.mean(score), ' and time=', time.clock() - startTime, ' secs'
        
        startTime = time.clock()
        print 'running K neighbor classifier'
        knc = KNeighborsClassifier(n_neighbors = 9)
        score = cross_validation.cross_val_score(knc, X, Y, n_jobs = self.n_jobs)
        print 'K neighbor classifier score = ', np.mean(score), ' and time=', time.clock() - startTime, ' secs'
        
        startTime = time.clock()
        print 'running Random forest classifier'
        rfc = RandomForestClassifier(n_jobs = self.n_jobs)
        score = cross_validation.cross_val_score(rfc, X, Y, n_jobs = self.n_jobs)
        print 'Random forest classifier score = ', np.mean(score), ' and time=', time.clock() - startTime, ' secs'   
        
        startTime = time.clock()
        print 'running Extra tree classifier'
        etc = ExtraTreesClassifier(n_jobs = self.n_jobs)
        score = cross_validation.cross_val_score(etc, X, Y, n_jobs = self.n_jobs)
        print 'Extra tree classifier score = ', np.mean(score), ' and time=', time.clock() - startTime, ' secs'   
        
        startTime = time.clock()
        print 'running Ada boost classifier'
        abc = AdaBoostClassifier()
        score = cross_validation.cross_val_score(abc, X, Y, n_jobs = self.n_jobs)
        print 'Ada boost classifier score = ', np.mean(score), ' and time=', time.clock() - startTime, ' secs'   
        
#         startTime = time.clock()
#         print 'running Gradient boosting classifier'
#         gbc = GradientBoostingClassifier(max_depth = 100)
#         score = cross_validation.cross_val_score(gbc, X, Y, n_jobs = self.n_jobs)
#         print 'Gradient boosting classifier score = ', np.mean(score), ' and time=', time.clock() - startTime, ' secs'  
         
#         startTime = time.clock()
#         print 'running Gaussian'
#         gp = gaussian_process.GaussianProcess(regr = 'quadratic')
#         score = cross_validation.cross_val_score(gp, X, Y, n_jobs = self.n_jobs)
#         print 'Gaussian score = ', np.mean(score), ' and time=', time.clock() - startTime, ' secs'

    def plotKneighbours(self, X, Y):
        meanScore = []
        x = []
        for i in range(1, 31) :
            start = time.clock()
            knc = KNeighborsClassifier(n_neighbors = i)
            score = cross_validation.cross_val_score(knc, X, Y, scoring=make_scorer(Gini), cv = 5, n_jobs = self.n_jobs)
            meanScore.append(np.mean(score))
            x.append(i)
            self.printLine(i, score, start)
            plt.Figure()
            plt.plot(x, meanScore, 'green')
            plt.scatter(x, meanScore, s = 5, c = 'red')
            self.savePic(plt, 'resrc/Kneighbours.png')
            plt.close()
        
    def plotRandomForest(self, X, Y):
        print 'running randomForest'
        meanScore = []
        x = []
        for i in range(1, 11) :
            start = time.clock()
            rfc = RandomForestClassifier(n_estimators=i*50, verbose=1)
            score = cross_validation.cross_val_score(rfc, X, Y, scoring=make_scorer(Gini), cv = 5, n_jobs = self.n_jobs)
            meanScore.append(np.mean(score))
            x.append(i*50)
            self.printLine(i, score, start)
            plt.Figure()
            plt.plot(x, meanScore, 'green')
            plt.scatter(x, meanScore, s = 5, c = 'red')
            self.savePic(plt, 'resrc/RandomForest_50.png')
            plt.close()
   
    def plotRandomForestReg(self, X, Y):
        print 'running randomForest'
        meanScore = []
        x = []
        for i in range(1, 11) :
            start = time.clock()
            rfc = RandomForestRegressor(n_estimators=i*100, verbose=1)
            score = cross_validation.cross_val_score(rfc, X, Y, scoring=make_scorer(Gini), cv = 5, n_jobs = self.n_jobs)
            meanScore.append(np.mean(score))
            x.append(i*100)
            self.printLine(i, score, start)
            plt.Figure()
            plt.plot(x, meanScore, 'green')
            plt.scatter(x, meanScore, s = 5, c = 'red')
            self.savePic(plt, 'resrc/RandomForestReg_100.png')
            plt.close()
        
    def plotExtraTree(self, X, Y, estimator_mul = 100):
        print 'running extratree'
        meanScore = []
        x = []
        for i in range(1, 11) :
            start = time.clock()
            etc = ExtraTreesClassifier(n_estimators= i * estimator_mul)
            score = cross_validation.cross_val_score(etc, X, Y, scoring=make_scorer(Gini), cv = 5, n_jobs=self.n_jobs)
            meanScore.append(np.mean(score))
            x.append(i*estimator_mul)
            self.printLine(i, score, start)
            plt.Figure()
            plt.plot(x, meanScore, 'green')
            plt.scatter(x, meanScore, s = 5, c = 'red')
            self.savePic(plt, 'resrc/ExtraTree_' + str(estimator_mul) + '.png')
            plt.close()
        
    def plotExtraTreeReg(self, X, Y, estimator_mul = 100):
        print 'running extratree Reg'
        meanScore = []
        x = []
        for i in range(1, 11) :
            start = time.clock()
            etc = ExtraTreesRegressor(n_estimators= i * estimator_mul, verbose=True)
            score = cross_validation.cross_val_score(etc, X, Y, scoring=make_scorer(Gini), cv = 5, n_jobs=self.n_jobs)
            meanScore.append(np.mean(score))
            x.append(i*estimator_mul)
            self.printLine(i, score, start)
            plt.Figure()
            plt.plot(x, meanScore, 'green')
            plt.scatter(x, meanScore, s = 5, c = 'red')
            self.savePic(plt, 'resrc/ExtraTreeReg_' + str(estimator_mul) + '.png')
            plt.close()
        
    def plotAdaBoost(self, X, Y, lr_fact = 10.0):
        print 'running adaboost'
        meanScore = []
        x = []
        for i in range(1, 11) :
            start = time.clock()
            abc = AdaBoostClassifier(learning_rate=(1.0 / (i*lr_fact)))
            score = cross_validation.cross_val_score(abc, X, Y, scoring=make_scorer(Gini), cv = 5, n_jobs = self.n_jobs)
            meanScore.append(np.mean(score))
            x.append(1.0 / (i*lr_fact))
            self.printLine(i, score, start)
            plt.Figure()
            plt.plot(x, meanScore, 'green')
            plt.scatter(x, meanScore, s = 10, c = 'red')
            self.savePic(plt, 'resrc/AdaBoost_learningrate_'+str(lr_fact) + '.png')
            plt.close()
        
    def plotAdaBoostRegWithDecisionReg(self, X, Y, estimator_mul = 100):
        print 'running adaboostReg with Decision Reg'
        meanScore = []
        x = []
        for i in range(1, 11) :
            start = time.clock()
            clf = AdaBoostRegressor(base_estimator = DecisionTreeRegressor(), n_estimators = i*estimator_mul, learning_rate = 0.1)
            score = cross_validation.cross_val_score(clf, X, Y, scoring=make_scorer(Gini), cv = 5, n_jobs = self.n_jobs)
            meanScore.append(np.mean(score))
            x.append(i*estimator_mul)
            self.printLine(i, score, start)
            plt.Figure()
            plt.plot(x, meanScore, 'green')
            plt.scatter(x, meanScore, s = 10, c = 'red')
            self.savePic(plt, 'resrc/AdaBoostRegWithDeciReg_' + str(estimator_mul) + '.png')
            plt.close()
        
    def plotGradientBoosting(self, X, Y):
        print 'running GradientBoosting'
        meanScore = []
        x = []
        for i in range(1, 2) :
            start = time.clock()
            gbc = GradientBoostingClassifier(n_estimators=i*100)
            score = cross_validation.cross_val_score(gbc, X, Y, scoring=make_scorer(Gini), cv = 5, n_jobs = self.n_jobs)
            meanScore.append(np.mean(score))
            x.append(1.0 / (i * 10.0))
            print 'completed for ', i, ' and score=', np.mean(score), ' in ', time.clock() - start, ' secs'
        plt.Figure()
        plt.plot(x, meanScore, 'green')
        plt.scatter(x, meanScore, s = 5, c = 'red')
        plt.savefig('GradientBoosting.png')
        plt.close()
        
        
