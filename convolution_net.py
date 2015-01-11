from basic_nnet import BasicNNet
import load_data
from lasagne import layers
from lasagne.layers.conv import Conv2DLayer
from lasagne.layers.pool import MaxPool2DLayer
from nolearn.lasagne import NeuralNet

class ConvolutionNet(object):
    
    def __init__(self):
        self.net = NeuralNet(
            layers = [
                ("input", layers.InputLayer),
                ("conv1", Conv2DLayer),
                ("pool1", MaxPool2DLayer),
                ("conv2", Conv2DLayer),
                ("pool2", MaxPool2DLayer),
                ("conv3", Conv2DLayer),
                ("pool3", MaxPool2DLayer),
                ("hidden1", layers.DenseLayer),
                ("hidden2", layers.DenseLayer),
                ("output", layers.DenseLayer)
                ],
            input_shape=(None, 1, 96, 96),
            conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_ds=(2, 2),
            conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_ds=(2, 2),
            conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_ds=(2, 2),
            
            hidden1_num_units=500,
            hidden2_num_units=500,
            output_num_units=30,
            output_nonlinearity=None,
            
            update_learning_rate=0.01,
            update_momentum=0.9,
            
            regression=True,
            max_epochs=1000,
            verbose=1)


if __name__ == "__main__":
    nnet = ConvolutionNet()
    X,y = load_data.load2d() 
    nnet.fit(X, y)
    nnet.lot()
    nnet.pickle()
