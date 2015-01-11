import load_data
from basic_nnet import BasicNNet
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

class SingleLayerNNet(BasicNNet):

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



if __name__ == "__main__":
    nnet = SingleLayerNNet()
    X,y = load_data.load()
    nnet.fit(X, y)
    nnet.plot()
    nnet.pickle()
