import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


class MyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(784, 10)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, trainX):
        preds = self.softmax(self.linear(trainX))
        return preds

    def loss(self, preds, labels):
        return torch.nn.functional.cross_entropy(preds, labels)

    def train(self, dataSet, epochs, lr, batchSize):
        trainLoader = DataLoader(dataset=dataSet, batch_size=batchSize, shuffle=True)
        opt = torch.optim.SGD(self.parameters(), lr=lr)
        for epoch in range(epochs):
            for xb, yb in trainLoader:
                preds = self.forward(xb)
                loss = self.loss(preds, yb)
                loss.backward()
                opt.step()
                opt.zero_grad()
        return self;

    def predict(self, X):
        return torch.argmax(self.forward(X), dim=1)

    @classmethod
    def getModel(cls, lr):
        model = MyModel()
        return model, torch.optim.SGD(model.parameters(), lr=lr)


