import data_loader
from matplotlib import pyplot
import pylab
import model
import math
import torch

from torch.utils.data import Dataset, DataLoader, TensorDataset



def accuracy(out, yb):
    preds = out
    return (preds == yb).float().sum()

def toTensor(x_train, y_train, x_valid, y_valid):
    return map(torch.tensor, (x_train, y_train, x_valid, y_valid))


def trainModel3(x_train, y_train, x_valid, y_valid):
    # train model
    epochs = 10
    bs = 256
    lr = 0.5
    trainDataSet = TensorDataset(x_train, y_train)
    myModel = model.MyModel()
    myModel.train(trainDataSet, epochs, lr, bs)
    preds_vaild = myModel.predict(x_valid)
    print('acc:{0}',accuracy(preds_vaild, y_valid)/y_valid.size(0))



if __name__ == "__main__":
    print("main function")
    loader = data_loader.DataLoader("./data/mnist/mnist.pkl.gz")
    x_train, y_train, x_valid, y_valid = loader.getDataSet()
    x_train, y_train, x_valid, y_valid = toTensor(x_train, y_train, x_valid, y_valid)
    # train model
    trainModel3(x_train, y_train, x_valid, y_valid)
