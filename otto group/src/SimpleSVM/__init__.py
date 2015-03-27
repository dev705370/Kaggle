from DataFormating import DataFormating as DF
from SVMModule import SimpleSVM
from Gaussian import GaussianClass
from Compare import Compare
from ExtraTree import ExtraTree
import time

if __name__ == '__main__' :
    _path = 'E://Python//Workspace//KaggleData//otto group'
    start = time.clock()
    trainData = DF.formatTrainData(_path)
    testData = DF.formatTestData(_path)
    print 'Data formatted'
#     svm = SimpleSVM(_path, trainData, testData)
#     svm.runSVM()
#     gc = GaussianClass(_path, trainData, testData)
#     gc.runGaussian()
    _cmp = Compare(_path, trainData)
    _cmp.run()
#     ET = ExtraTree(_path, trainData, testData)
#     ET.runExtraTree()
    print 'run time is ', time.clock() - start, 'secs'
    