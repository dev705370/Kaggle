from DataMugging.DataMugging import DataMugging


if __name__ == '__main__' :
    source = 'E://Kaggle//Drivers//drivers//'
    destination = 'data/'
    dm = DataMugging(source, destination)
    dm.start()
