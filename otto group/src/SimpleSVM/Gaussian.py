'''
Created on Mar 22, 2015

@author: Devendra
'''


from sklearn import gaussian_process

class GaussianClass():
    def __init__(self, _path, trainData, testData):
        self.__path = _path
        self.__trainData = trainData
        self.__testData = testData
        
    def runGaussian(self):
        X = self.__trainData.values
        Y = X[:, -1]
        X = X[:, :-1]
        gp = gaussian_process.GaussianProcess(regr = 'quadratic')
        gp.fit(X, Y)
        print 'training done'
        print 'score = ', gp.score(X, Y)
#         Xtest = self.__testData.ix[:,'feat_1':].values
#         _id = self.__testData.ix[:,'id'].values
#         _id = _id.reshape([_id.shape[0], 1])
#         output = clf.predict_proba(Xtest)
#         output = np.hstack([_id, output])
#         outData = pd.DataFrame(output, columns=['id', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9'])
#         outData['id'] = outData['id'].apply(self.convert)
#         outData.to_csv(self.__path + '//output2.csv', index=False)
        
    def convert(self, x):
        try:
            return x.astype(int)
        except:
            return x
        