import constants
from sklearn import decomposition


class FeatureSelect:
    def __init__(self, data, ignored_cols=constants.COLUMN_THRESHOLD):
        self.matrix = data.as_matrix(x for x in data.columns if x not in ignored_cols)
        self.ignored_cols = ignored_cols
        self.components = self.matrix
        self.features = None

    def pca(self, no_components=2):
        self.features = decomposition.PCA(n_components=no_components)
        self.fit()

    def lda(self, no_topics=10, learning_method='batch'):
        self.features = decomposition.LatentDirichletAllocation(n_topics=no_topics, learning_method=learning_method)
        self.fit()

    def ica(self):
        self.features = decomposition.FastICA()
        self.fit()

    def fit(self):
        self.components = self.features.fit(self.matrix).transform(self.matrix)
