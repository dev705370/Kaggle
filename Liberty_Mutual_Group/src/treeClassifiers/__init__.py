import pandas as pd
from LoadData import LoadData
from sklearn import svm, cross_validation
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier, AdaBoostRegressor, AdaBoostClassifier
from sklearn.metrics import make_scorer
import time
import numpy as np
from Compare import Compare
import warnings
import datetime
import pandas as pd

TRAIN = 'E:\Python\Workspace\Kaggle\Liberty Mutual Group\resources\train.csv.zip'
TEST = 'E:\Python\Workspace\Kaggle\Liberty Mutual Group\resources\test.csv.zip'

def compareAll(X, Y, n_jobs = -1):
#     startTime = time.clock()
#     print 'running svm'
#     clf = svm.SVC(cache_size = 2000)
#     score = cross_validation.cross_val_score(clf, X, Y, n_jobs = -1)
#     t = time.clock() - startTime
#     print 'svm score = ', np.mean(score), ' and time=', t, ' secs'
     
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


def runSVRwithCache(X, Y, n_jobs = 5):
    print 'starting run svr with different cache sizes'
    for i in range(1, 11) :
        start = time.clock()
        clf = svm.SVR(cache_size=200*i, verbose=True)
        score = cross_validation.cross_val_score(clf, X, Y, cv = 5, n_jobs = n_jobs)
        print 'completed for ', i, ' and score=', np.mean(score), ' in ', time.clock() - start, ' secs at ', datetime.datetime.now()
    print 'done'
    
    
def Gini(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]
    
    # sort rows on prediction column 
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:,0].argsort()][::-1,0]
    pred_order = arr[arr[:,1].argsort()][::-1,0]
    
    # get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(0, 1, n_samples)
    
    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)
    
    # normalize to true Gini coefficient
    return G_pred/G_true

if __name__ == '__main__' :
    warnings.filterwarnings('ignore')
    print 'Loading data'
    X, Y = LoadData()
    print 'Data loaded'
#     score = cross_validation.cross_val_score(RandomForestClassifier(), X, Y, scoring = make_scorer(Gini), cv = 5, n_jobs = 5)
#     print score
#     print np.mean(score)
#     data = pd.DataFrame(data = X)
#     data.to_csv("temp.csv", index = False)
#     runSVRwithCache(X, Y, n_jobs=5)
#     compareAll(X, Y, 4)
    cmp = Compare(n_jobs=5)
    cmp.plotExtraTreeReg(X, Y, estimator_mul = 50)
    cmp.plotExtraTree(X, Y, estimator_mul = 50)
    cmp.plotKneighbours(X, Y)
    cmp.plotRandomForest(X, Y)
    cmp.plotRandomForestReg(X, Y)
    cmp.plotAdaBoost(X, Y)
    cmp.plotAdaBoostRegWithDecisionReg(X, Y, estimator_mul = 50)
    cmp.plotSVR(X, Y)
    cmp.plotSVM(X, Y)
    
    
