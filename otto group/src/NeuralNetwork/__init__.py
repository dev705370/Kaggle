import loadData
from sklearn.preprocessing import OneHotEncoder
import neuralNet
import matplotlib.pyplot as pyplot
import numpy as np
import pandas


def getData(net, test = False):
    if net == 2:
        return loadData.LoadData2D(test)
    else :
        return loadData.LoadData(test)
        

def init_network(net, prob):
    max_iter = 1000
    if net == 1:
        return neuralNet.Network1(max_epochs=max_iter)
    elif net == 2:
        return neuralNet.Network2(max_epochs = 2)
    elif net == 3:
        return neuralNet.Network3(max_epochs=max_iter, prob = prob)
    elif net == 4:
        return neuralNet.Network4(max_epochs = max_iter)
    elif net == 5:
        return neuralNet.Network5(max_epochs = max_iter)
    elif net == 6:
        return neuralNet.Network6(max_epochs = max_iter)
    elif net == 7:
        return neuralNet.Network7(max_epochs = max_iter, units=1000)
    else :
        return neuralNet.Network8(max_epochs = max_iter)

def network(net, prob = 0.2):
    X, y = getData(net)
    net1 = init_network(net, prob)
    net1.fit(X, y)
    train_loss = np.array([i["train_loss"] for i in net1.train_history_])
    valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])
#     valid_accuracy[prob] = np.array([i["valid_accuracy"] for i in net1.train_history_])
    pyplot.figure(prob*100)
    pyplot.plot(train_loss, linewidth=3, label="train")
    pyplot.plot(valid_loss, linewidth=3, label="valid")
    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel("epoch")
    pyplot.ylabel("loss")
#     pyplot.yscale("log")
    pyplot.savefig('fig_net_' + str(net) + '_' + str(prob) + '.png')
    X_test, _ = loadData.LoadData(test = True)
    y_predict = net1.predict_proba(X_test)
#     if net == 7:
#         y_predict = softmax(y_predict)
    test_id = loadData.getTestId()
    test_id = test_id.reshape([test_id.shape[0], 1])
    y_predict = np.concatenate([test_id, y_predict], axis = 1)
    columns = ['id'] + list(loadData.getColumns())
    data = pandas.DataFrame(y_predict, columns=columns)
    data['id'] = data['id'].apply(convertInt)
    data.to_csv("out_network" + str(net) + '_' + str(prob) + ".csv", index=False)
    pyplot.show()
    
def softmax(y_predict):
    print y_predict.shape
    y_sum = np.sum(y_predict, axis=1)
    print y_sum.shape
    return y_predict / y_sum
    
def convertInt(x):
    try:
        return x.astype(int)
    except:
        return x
    
 
valid_accuracy = {}   
if __name__ == '__main__' :
#     network(1)
#     network(2)
#     for i in range(1, 10):
#         network(3, 0.1*i)
#     pyplot.figure(1)
#     for key, value in valid_accuracy.iteritems():
#         pyplot.plot(value, linewidth=2, label=str(key))
#     pyplot.grid()
#     pyplot.legend()
#     pyplot.xlabel("epoch")
#     pyplot.ylabel("accuracy")
#     pyplot.savefig('fig_net3_acc.png')
#     pyplot.show()
#     network(4)
#     network(5)
#     network(6)
    network(7)
#     network(8)
    