import numpy
from matplotlib import pyplot
import load_data
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

class SingleLayerNNet():

    def __init__(self):
        self.net = NeuralNet(
            layers = [ # Three layers, one hidden
                ("input", layers.InputLayer),
                ("hidden", layers.DenseLayer),
                ("output", layers.DenseLayer)
                ],
            input_shape=(None, 96 * 96), # 96 * 96 input pixels per batch
            hidden_num_units=100,
            output_nonlinearity=None, #Output layer uses identity function
            output_num_units=30, # 30 target values
        
            #Optimization method
            update=nesterov_momentum,
            update_learning_rate=0.01,
            update_momentum=0.9,
            
            regression=True, # Flag to indicate that this is regression, not classification
            max_epochs=400,
            verbose=1)


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


if __name__ == "__main__":
    nnet = SingleLayerNNet()
    X,y = load_data.load()
    nnet.fit(X, y)
    nnet.plot()
