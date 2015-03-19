from theano import tensor

from blocks.bricks import Softmax, Linear, Rectifier, MLP
from blocks.bricks.cost import CategoricalCrossEntropy

from blocks.initialization import IsotropicGaussian, Constant, Orthogonal

from cuboid.bricks import BrickSequence

class ModelHelper(object):
    def __init__(self):
        x = tensor.matrix('features')
        y = tensor.ivector('targets')

        seq = BrickSequence(input_dim=93, bricks=[
            Linear(input_dim = 93, output_dim = 9,
                weights_init=IsotropicGaussian(0.01),
                biases_init = Constant(0))
            ])

        seq.initialize()
        o = seq.apply(x)

        probs = Softmax().apply(o)
        self.cost = CategoricalCrossEntropy().apply(y, probs)
