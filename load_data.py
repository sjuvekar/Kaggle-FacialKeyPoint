import os
import numpy
import pandas
from sklearn.utils import shuffle

TRAIN_FILE = "Data/training.csv"
TEST_FILE = "Data/test.csv"

def load(test=False, cols=None):
    fname = TEST_FILE if test else TRAIN_FILE
    df = pandas.read_csv(fname)

    df["Image"] = df["Image"].apply(lambda im: numpy.fromstring(im, sep = " "))
    
    if cols:
        df = df[list(cols) + ["Image"]]

    print df.count()
    df = df.dropna()

    X = numpy.vstack(df["Image"].values) / 255. #Scale image values to be between [0, 1]
    X = X.astype(numpy.float32)

    if not test:
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48 # Scale target coordinates to be between [0, 1]
        X, y = shuffle(X, y, random_state = 42) # Shuffle train data
        y = y.astype(numpy.float32)
    else:
        y = None

    return X,y

if __name__ == "__main__":
    X,y = load()
    print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
           X.shape, X.min(), X.max()))
    print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
           y.shape, y.min(), y.max()))
