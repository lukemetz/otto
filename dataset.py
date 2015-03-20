import theano
import numpy as np
import pandas as pd
import os

from fuel.datasets import IndexableDataset
from fuel.schemes import ShuffledScheme, SequentialScheme

from fuel.streams import DataStream
from collections import OrderedDict
from sklearn.cross_validation import train_test_split

from fuel import config

from sklearn import preprocessing, feature_extraction

class OttoDataset(IndexableDataset):
    provides_sources = ('features', 'targets')
    folder = 'otto'
    def __init__(self, which_set, **kwargs):
        if which_set not in ('train', 'test', 'leaderboard'):
            raise ValueError("Otto dataset only has train and test and leaderboard")
        self.which_set = which_set

        indexables = OrderedDict(zip(self.provides_sources, self._load_otto()))
        super(OttoDataset, self).__init__(indexables, **kwargs)

    def _load_otto(self):
        base_path = os.path.join(config.data_path, self.folder)
        train =  pd.read_csv(os.path.join(base_path, "train.csv"))
        classes_str = train['target']
        targets = np.zeros_like(classes_str.values, dtype="int32")
        for i in range(1, 10):
            targets[classes_str.values == ("Class_%d"%i)] = i
        features = train[["feat_%d"%x for x in range(1, 94)]].values.astype(theano.config.floatX)

        print "Preprocessing"
        scalar = preprocessing.StandardScaler(copy=False)
        features = scalar.fit_transform(features)
        #tfidf = feature_extraction.text.TfidfTransformer()
        #features = tfidf.fit_transform(features).toarray()

        if self.which_set == "leaderboard":
            test =  pd.read_csv(os.path.join(base_path, "test.csv"))
            features = test[["feat_%d"%x for x in range(1, 94)]].values.astype(theano.config.floatX)

            features = scalar.transform(features)
            print features.shape

        targets -= 1
        features = features.astype(theano.config.floatX)

        if self.which_set == "train" or self.which_set == "test":
            tr_features, te_features, tr_targets, te_targets =\
                    train_test_split(features, targets, test_size=0.3, random_state=42)

            if self.which_set == "train":
                return tr_features, tr_targets
            elif self.which_set == "test":
                return te_features, te_targets

        elif self.which_set == "leaderboard":
            return features, np.zeros((features.shape[0], ))

class DatasetHelper(object):
    def __init__(self):
        pass

    def get_train_stream(self):
        dataset = OttoDataset('train')
        scheme = ShuffledScheme(examples=range(dataset.num_examples), batch_size=128)
        datastream = DataStream(dataset=dataset, iteration_scheme=scheme)
        return datastream

    def get_test_stream(self):
        dataset = OttoDataset('test')
        scheme = ShuffledScheme(examples=range(dataset.num_examples), batch_size=128)
        datastream = DataStream(dataset=dataset, iteration_scheme=scheme)
        return datastream

    def get_leaderboard_stream(self):
        dataset = OttoDataset('leaderboard')
        scheme = SequentialScheme(examples=range(dataset.num_examples), batch_size=128)
        datastream = DataStream(dataset=dataset, iteration_scheme=scheme)
        return datastream


if __name__ == "__main__":
    h = DatasetHelper()
    print h.get_train_stream().get_epoch_iterator().next()

