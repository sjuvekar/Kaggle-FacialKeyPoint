import numpy
from matplotlib import pyplot

class BasicNNet(object):
    
    def __init__(self):
        self.net = None

    def fit(self, X, y):
        self.net.fit(X, y)

    def predict(self, x):
        return self.net.predict(X)

    def plot(self):
        """
        Plot train and validation losses from history
        """
        train_loss = numpy.array([i["train_loss"] for i in self.net.train_history_])
        valid_loss = numpy.array([i["valid_loss"] for i in self.net.train_history_])
        pyplot.plot(train_loss, linewidth=3, label="train")
        pyplot.plot(valid_loss, linewidth=3, label="valid")
        pyplot.grid()
        pyplot.legend()
        pyplot.xlabel("epoch")
        pyplot.ylabel("loss")
        pyplot.ylim(1e-3, 1e-2)
        pyplot.yscale("log")
        pyplot.show()

    def pickle(self):
        with open("models/{}.pickle", str(self.__class__.__name__), "wb") as f:
            pickle.dump(self.net, f, -1)
