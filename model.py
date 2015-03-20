from theano import tensor

from blocks.bricks import Softmax, Linear, Rectifier, MLP, WEIGHT
from blocks.bricks.cost import CategoricalCrossEntropy

from blocks.initialization import IsotropicGaussian, Constant, Orthogonal

from cuboid.bricks import BrickSequence, Dropout, LeakyRectifier

from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph

class ModelHelper(object):
    def __init__(self):
        x = tensor.matrix('features')
        y = tensor.ivector('targets')

        seq = BrickSequence(input_dim=93, bricks=[
            Linear(output_dim = 128,
                weights_init=IsotropicGaussian(0.01),
                biases_init = Constant(0))
            , LeakyRectifier(a=0.3)

            , Dropout(p_drop=0.2)
            , Linear(output_dim = 256,
                weights_init=IsotropicGaussian(0.01),
                biases_init = Constant(0))
            , LeakyRectifier(a=0.3)

            , Dropout(p_drop=0.2)
            , Linear(output_dim = 256,
                weights_init=IsotropicGaussian(0.01),
                biases_init = Constant(0))
            , LeakyRectifier(a=0.3)

            , Linear(output_dim = 9,
                weights_init=IsotropicGaussian(0.01),
                biases_init = Constant(0))
            ])

        seq.initialize()
        o = seq.apply(x)

        probs = Softmax().apply(o)
        self.cost = CategoricalCrossEntropy().apply(y, probs)
        self.cost.name = "cost"

        o = seq.apply_inference(x)
        self.inference_probs= Softmax().apply(o)
        self.inference_cost = CategoricalCrossEntropy().apply(y, self.inference_probs)
        self.inference_cost.name  = "inference_cost"

        cg = ComputationGraph([self.cost])
        weights = VariableFilter(roles=[WEIGHT])(cg.variables)
        self.reg_cost = self.cost + sum([1e-6 * (w ** 2).sum() for w in weights])
        self.reg_cost.name = "reg_cost"

