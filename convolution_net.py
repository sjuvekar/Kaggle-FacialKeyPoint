import theano
from basic_nnet import BasicNNet
import load_data
from lasagne import layers
from lasagne.layers.conv import Conv2DLayer
from lasagne.layers.pool import MaxPool2DLayer
from nolearn.lasagne import NeuralNet
from flip_batch_iterator import FlipBatchIterator
from adjust_variable import AdjustVariable
from util import float32

Conv2DLayer = layers.cuda_convnet.Conv2DCCLayer
MaxPool2DLayer = layers.cuda_convnet.MaxPool2DCCLayer

class ConvolutionNet(BasicNNet):
    
    def __init__(self):
        self.net = NeuralNet(
            layers = [
                ("input", layers.InputLayer),
                ("conv1", Conv2DLayer),
                ("pool1", MaxPool2DLayer),
		('dropout1', layers.DropoutLayer),
                ("conv2", Conv2DLayer),
                ("pool2", MaxPool2DLayer),
		('dropout2', layers.DropoutLayer),
                ("conv3", Conv2DLayer),
                ("pool3", MaxPool2DLayer),
                ('dropout3', layers.DropoutLayer),
		("hidden1", layers.DenseLayer),
		('dropout4', layers.DropoutLayer),
                ("hidden2", layers.DenseLayer),
                ("output", layers.DenseLayer)
                ],
            input_shape=(None, 1, 96, 96),
            conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_ds=(2, 2),
            conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_ds=(2, 2),
            conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_ds=(2, 2),
           
	    dropout1_p=0.1,
	    dropout2_p=0.2,
  	    dropout3_p=0.3,
	    dropout4_p=0.5,
 
            hidden1_num_units=1000,
            hidden2_num_units=1000,
            output_num_units=30,
            output_nonlinearity=None,

            #update_learning_rate=0.01,
            #update_momentum=0.9,
            update_learning_rate=theano.shared(float32(0.03)),
            update_momentum=theano.shared(float32(0.9)),
            on_epoch_finished=[
                AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
                AdjustVariable('update_momentum', start=0.9, stop=0.999),
                ],

            regression=True,
            batch_iterator_train=FlipBatchIterator(batch_size=128),
            max_epochs=10000,
            verbose=1)


if __name__ == "__main__":
    nnet = ConvolutionNet()
    X,y = load_data.load2d() 
    nnet.fit(X, y)
    try:
        nnet.plot()
    except:
        pass
    nnet.pickle()
