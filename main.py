from dataset import DatasetHelper
import theano
from theano import tensor
from blocks.main_loop import MainLoop
from blocks.algorithms import GradientDescent, Adam
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from cuboid.algorithms import NAG

from model import ModelHelper

m = ModelHelper()

cg = ComputationGraph([m.cost])
print cg.parameters

algorithm = GradientDescent(
    cost=m.cost, params=cg.parameters,
    #step_rule=Adam(learning_rate=0.002))
    step_rule=NAG(lr=0.01, momentum=0.9))

h = DatasetHelper()
stream = h.get_train_stream()

main_loop = MainLoop(
        algorithm,
        h.get_train_stream(),
        model = Model(m.cost),
        extensions = [Timing()
            , TrainingDataMonitoring(
                [m.cost],
                prefix="train",
                after_epoch=True)
            , DataStreamMonitoring(
                [m.cost],
                h.get_test_stream(),
                prefix="test")
            , Printing()])

main_loop.run()
