'''
Created on Apr 15, 2015

@author: Devendra
'''

import pandas as pd
from sklearn import preprocessing
from sklearn.utils import shuffle
import numpy as np

FTRAIN = 'E:/Python/Workspace/KaggleData/otto group/train.csv'
FTEST = 'E:/Python/Workspace/KaggleData/otto group/test.csv'

def LoadData(test = False) :
    fName = FTEST if test else FTRAIN
    df = pd.read_csv(fName, header = 0)
    df = df.drop(['id'], axis=1)
    if not test :
        le = preprocessing.LabelEncoder()
        df['class'] = le.fit_transform(df['target'])
        df = df.drop(['target'], axis=1)
        y = df['class'].values
        X = df.drop(['class'], axis = 1).values
#         y = y.reshape([y.shape[0], 1])
#         enc = preprocessing.OneHotEncoder()
#         y = enc.fit_transform(y)
#         y = y.toarray()
#         y = y.astype(np.int32)
        X, y = shuffle(X, y, random_state=42)
    else :
        X = df.values
        y = None
    X = X.astype(np.float32)
    return X, y

def LoadData2D(test = False) :
    X, y = LoadData(test)
    if not test :
        X = X.reshape(-1,1,1,X.shape[1])
    return X, y

def getTestId():
    df = pd.read_csv(FTEST, header = 0)
    return df['id'].values

def getColumns():
    df = pd.read_csv(FTRAIN, header = 0)
    le = preprocessing.LabelEncoder()
    le.fit(df['target'])
    return le.classes_