from os import path
from Datafomating import DataFormat
from sklearn.svm import OneClassSVM
import time


if __name__ == '__main__' :
    start = time.clock()
    df = DataFormat(path.dirname(__file__) + '/../../CompressData/')  
    df.readAll()
    print 'total running time = ', time.clock() - start
#     print 'data Formated'
#     start = time.clock()
#     clf = OneClassSVM(nu = 0.1, verbose = 1)
#     X = df.getData('TSO1')
#     clf.fit(X)
#     y = clf.predict(X)
#     print 'running time = ', time.clock() - start
#     print y