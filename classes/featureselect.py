import constants
from sklearn import decomposition
from random import uniform, randrange
import re
import numpy as np


class FeatureSelect:
    def __init__(self, data, extra_fit=False):
        self.data = data
        self.matrix = data.as_matrix(x for x in data.columns if x not in constants.COLUMN_THRESHOLD)
        self.components = self.matrix
        self.features = None
        self.extra_fit = extra_fit

    def pca(self, no_components=2):
        self.features = decomposition.PCA(n_components=no_components)
        self.fit()

    def lda(self, no_topics=2):
        self.features = decomposition.LatentDirichletAllocation(n_topics=no_topics, learning_method='batch')
        self.fit()

    def ica(self):
        self.features = decomposition.FastICA()
        self.fit()

    def fit(self):
        if self.extra_fit:
            col = self.data.as_matrix(x for x in self.data.columns if x in constants.COLUMN_THRESHOLD and re.match("^[S-U].*", x))
            for arr in col:
                if uniform(0, 1) > 0.81:
                    arr[0] = randrange(1, 6)
            self.components = np.c_[self.features.fit(self.matrix).transform(self.matrix), col]
        else:
            self.components = self.features.fit(self.matrix).transform(self.matrix)
