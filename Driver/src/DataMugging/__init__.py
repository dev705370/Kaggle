from DataMugging import DataMugging


if __name__ == '__main__' :
    source = 'E://Kaggle//Drivers//drivers//'
    destination = 'E://Python//Workspace//Kaggle//Driver//resource//DataMugging//'
    dm = DataMugging(source, destination)
    dm.start()