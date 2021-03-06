from classes.input import Input
from classes.plot import Plot
from classes.preprocess import Preprocess
from classes.featureselect import FeatureSelect
from classes.helpers import Helpers
from classes.classify import Classify
from classes.precision import Precision
import classes.constants as constants


@constants.timeit
def main():
    ignore_cols = [constants.COLUMN_TIMESTAMP, constants.COLUMN_SENSOR_1, constants.COLUMN_SENSOR_2,
                   constants.COLUMN_SENSOR_11, constants.COLUMN_SENSOR_16]

    input = Input('input/__Alexandru Constantin_5s_10_graveyardbeach_27052017_224455.csv', ignore_cols=ignore_cols)
    input.read_csv()
    input.make_column_uniform()
    input.replace_column_with_thresholds()

    # @constants.timeit
    def preprocess(data):
        preprocess = Preprocess(data)
        preprocess.remove_dc_offset()
        preprocess.resample(100)
        preprocess.detrend()
        # preprocess.notch_filter(50)
        preprocess.bandpass_filter(1, 50)
        # preprocess.discard_datapoints_below_or_over()
        # preprocess.discard_datapoints_by_ratio_to_median()
        # preprocess.fft()
        preprocess.min_max_scale()
        return preprocess.data

    preprocessed_data = preprocess(input.data)
    Plot.plot_without_threshold(preprocessed_data)

    [data_train, data_test] = Helpers.split_by_column_into_train_test(preprocessed_data)

    featureselect_train = FeatureSelect(data_train)
    featureselect_train.pca()

    featureselect_test = FeatureSelect(data_test)
    featureselect_test.pca()

    labels_train = Helpers.extract_labels_from_dataframe(data_train)
    labels_test = Helpers.extract_labels_from_dataframe(data_test)

    # @constants.timeit
    def classify_and_compute_precision(C=1.0, gamma='auto'):
        classify = Classify(featureselect_train.components, labels_train, mode='randomforest')
        classify.classify(C, gamma)

        params_string = '[C=' + str(C) + '][gamma=' + str(gamma) + ']'

        predicted_labels_test = classify.predict(featureselect_test.components)

        Plot.plot_lists([
            {'data': labels_test, 'label': 'Expected' + params_string},
            {'data': predicted_labels_test, 'label': 'Predicted' + params_string}
        ])
        precision_obj = Precision(real_labels=labels_test, predicted_labels=predicted_labels_test)
        raw_precision = precision_obj.compute_raw_precision()
        cat_precision = precision_obj.compute_per_category_median_precision()
        print 'raw_precision = ', raw_precision
        print 'cat_precision = ', cat_precision

        return cat_precision

    # classify_and_compute_precision()
    max_precision = 0
    for c_pow in xrange(7, 15, 2):
        for gamma_pow in xrange(-8, -4, 2):
            C = 10 ** c_pow
            gamma = 10 ** gamma_pow
            precision = classify_and_compute_precision(C, gamma)
            if precision > max_precision:
                max_precision = precision

    print 'max(precision) = ', max_precision


main()
