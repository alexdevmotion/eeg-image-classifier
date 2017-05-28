from classes.input import Input
from classes.plot import Plot
from classes.preprocess import Preprocess
from classes.featureselect import FeatureSelect
from classes.helpers import Helpers
from classes.classify import Classify
import classes.constants as constants

ignore_cols = []
no_components = 4

input = Input('input/out_Daniel_25012017_noduplicates.csv')
input.read_csv_emotiv()

preprocess = Preprocess(input.data)
preprocess.remove_dc_offset()
# preprocess.notch_filter(50)
preprocess.bandpass_filter(1, 50)

[data_train, data_test] = Helpers.split_by_column_into_train_test(preprocess.data)


featureselect_train = FeatureSelect(data_train)
featureselect_train.pca(no_components=no_components)

featureselect_test = FeatureSelect(data_test)
featureselect_test.pca(no_components=no_components)

labels_train = Helpers.extract_labels_from_dataframe(data_train)

classify = Classify(featureselect_train.components, labels_train)
classify.svm_classify()

labels_test = Helpers.extract_labels_from_dataframe(data_test)
predicted_labels_test = classify.svm_predict(featureselect_test.components)

print labels_test
print predicted_labels_test
print Helpers.compute_precision(labels_test, predicted_labels_test)
