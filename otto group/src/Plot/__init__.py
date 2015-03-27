from plotTwo import PlotForTwo
from plotOne import PlotOne

if __name__ == '__main__' :
#     PFT = PlotForTwo('E://Python//Workspace//KaggleData//otto group//train.csv')
#     PFT.plot('feat_13', 'feat_24')
    PO = PlotOne('E://Python//Workspace//KaggleData//otto group', 'train.csv')
    PO.plotAll()