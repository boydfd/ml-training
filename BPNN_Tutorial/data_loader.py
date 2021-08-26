import pickle
import gzip

class DataLoader:

    def __init__(self, path):
        self.path = path
        self.url = "http://deeplearning.net/data/mnist/"
        self.fileName = "mnist.pkl.gz"
        self.downloadDataSet()

    def getDataSet(self):
        with gzip.open(self.path, "rb") as f:
            ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
        return x_train, y_train, x_valid, y_valid

    def downloadDataSet(self):
        import requests
        from pathlib import Path
        if not (Path(self.path)).exists():
            content = requests.get(self.url + self.fileName).content
            Path(self.path).open("wb").write(content)


if __name__ == "__main__":
    print("testing function")
    loader = DataLoader("./data/mnist/mnist.pkl.gz")
    x_train, y_train, x_valid, y_valid = loader.getDataSet()

    from matplotlib import pyplot
    import pylab

    pyplot.imshow(x_train[1].reshape((28, 28)), cmap="gray")
    pylab.show()

    print("finish")
