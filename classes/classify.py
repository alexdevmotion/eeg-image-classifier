from sklearn import svm
from sklearn import tree
from sklearn import ensemble
from sklearn import neural_network
from sklearn import neighbors
from sklearn import linear_model
from sklearn import discriminant_analysis


class Classify:
    def __init__(self, components, labels, mode):
        self.components = components
        self.labels = labels
        self.model = None
        self.mode = mode
        self.classify_func = self.get_classify_func(mode)

    def get_classify_func(self, mode):
        if mode == 'svm':
            return self.svm_classify
        elif mode == 'decisiontree':
            return self.decision_tree_classify
        elif mode == 'randomforest':
            return self.random_forest_classify
        elif mode == 'adaboost':
            return self.adaboost_classify
        elif mode == 'neuralnetwork':
            return self.neural_network_classify
        elif mode == 'nearestneighbors':
            return self.nearest_neighbors_classify
        elif mode == 'sgd':
            return self.sgd_classify
        elif mode == 'gradientboosting':
            return self.gradient_boosting_classify
        elif mode == 'lda':
            return self.lda_classify
        elif mode == 'bagging':
            return self.bagging_classify
        elif mode == 'voting':
            return self.voting_classify
        else:
            raise Exception('No such classification mode')

    def classify(self, *args, **kw):
        if self.mode == 'svm' or self.mode == 'bagging' or self.mode == 'voting':
            self.model = self.classify_func(*args, **kw)
        else:
            self.model = self.classify_func()
        self.model.fit(self.components, self.labels)

    def predict(self, other_components):
        return self.model.predict(other_components).ravel()

    def nearest_neighbors_classify(self):
        return neighbors.KNeighborsClassifier(n_neighbors=100)

    def random_forest_classify(self):
        return ensemble.RandomForestClassifier(n_estimators=10, criterion='entropy')

    def adaboost_classify(self):
        return ensemble.AdaBoostClassifier()

    def decision_tree_classify(self):
        return tree.DecisionTreeClassifier()

    def gradient_boosting_classify(self):
        return ensemble.GradientBoostingClassifier(n_estimators=100)

    def svm_classify(self, C=1.0, gamma='auto'):
        # remember @optunity
        return svm.SVC(C=C, gamma=gamma)

    def neural_network_classify(self):
        return neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

    def sgd_classify(self):
        return linear_model.SGDClassifier()

    def lda_classify(self):
        return discriminant_analysis.LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')

    def bagging_classify(self, base):
        base_estimator = self.get_classify_func(base)
        return ensemble.BaggingClassifier(base_estimator())

    def voting_classify(self, estimators, voting='hard', weights=None):
        estimators_arr = []
        for estimator_str in estimators:
            estimators_arr.append((estimator_str, self.get_classify_func(estimator_str)()))
        return ensemble.VotingClassifier(estimators_arr, voting=voting, weights=weights)
