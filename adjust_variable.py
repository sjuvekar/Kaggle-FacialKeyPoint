from util import float32

class AdjustVariable(object):

    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start = start
        self.stop = stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls in None:
            self.ls = nn.linspace(self.start, self.stop, nn.max_epochs)
        
        epoch = train_history[-1]["epoch"]
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)
