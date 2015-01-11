import numpy
from nolearn.lasagne import BatchIterator

class FlipBatchIterator(BatchIterator):
    
    flip_indices = [
        (0, 2), (1, 3),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (22, 24), (23, 25),
        ]

    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        # Flip half of the images
        batch_size = Xb.shape[0]
        indices = numpy.random.choice(batch_size, batch_size / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        if yb is not None:
            # Horizontally flip marker positions
            yb[indices, ::2] = yb[indices, ::] * -1

        # Swap places of left and right
        for a, b in self.flip_indices:
            yb[indices, a], yb[indices, b] = (yb[indices, b], yb[indices , a])

        return Xb, yb
            
