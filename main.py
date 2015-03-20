from dataset import DatasetHelper
import theano
from theano import tensor
from blocks.main_loop import MainLoop
from blocks.algorithms import GradientDescent, Adam
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.extensions.plot import Plot
from blocks.extensions.saveload import Dump, Checkpoint

from cuboid.algorithms import NAG

from cuboid.evaluators import DatasetMapEvaluator
from model import ModelHelper
import pandas as pd
import numpy as np

m = ModelHelper()

cg = ComputationGraph([m.cost])
print cg.parameters

algorithm = GradientDescent(
    cost=m.reg_cost, params=cg.parameters,
    #step_rule=Adam(learning_rate=0.002))
    step_rule=NAG(lr=0.01, momentum=0.9))

h = DatasetHelper()

stream = h.get_train_stream()

main_loop = MainLoop(
        algorithm,
        h.get_train_stream(),
        model = Model(m.reg_cost),
        extensions = [Timing()
            , TrainingDataMonitoring(
                [m.cost, m.reg_cost],
                prefix="train",
                after_epoch=True)
            , DataStreamMonitoring(
                [m.cost, m.inference_cost],
                h.get_test_stream(),
                prefix="test")
            , Plot('otto',
                channels=[['test_cost', 'train_cost'], ['test_inference_cost']])
            #, FinishAfter(after_n_epochs=90)
            , Printing()])

main_loop.run()
print "Writing"

evaluator = DatasetMapEvaluator(m.inference_probs)
result = evaluator.evaluate(h.get_leaderboard_stream())
ids = np.reshape(range(1, result.shape[0]+1), (-1, 1))
#result = np.hstack((ids, result))
classes = ["Class_%s"%i for i in range(1, 10)]
frame = pd.DataFrame(result, columns=classes)
frame['id'] = ids
frame.to_csv("out.csv", index=False, columns=['id'] + classes)
