from classes.input import Input
from classes.plot import Plot
from classes.preprocess import Preprocess
from classes.featureselect import FeatureSelect
from classes.helpers import Helpers
from classes.classify import Classify
from classes.precision import Precision
import classes.constants as constants


# @constants.timeit
def main():
    path1 = 'input/session_3/__Alexandru Constantin_2s_1_colors_simple_15062017_222039.csv'
    path2 = 'input/session_3/__Alexandru Constantin_2s_1_colors_simple_reordered_15062017_221958.csv'

    ignore_cols = [constants.COLUMN_TIMESTAMP, constants.COLUMN_SENSOR_1, constants.COLUMN_SENSOR_2,
                   constants.COLUMN_SENSOR_9, constants.COLUMN_SENSOR_10]

    def read_and_preprocess(path, mapping=None):
        input = Input(ignore_cols=ignore_cols)
        input.read_csv(path)
        input.make_column_uniform()
        if mapping is None:
            mapping = input.replace_column_with_thresholds()
        else:
            input.replace_column_with_thresholds(mapping=mapping)

        # Plot.plot_without_threshold(input.data)

        # @constants.timeit
        def preprocess(data):
            preprocess = Preprocess(data)
            preprocess.remove_dc_offset()
            preprocess.resample(100)
            preprocess.detrend()
            n_channels = preprocess.discard_columns_by_ratio_to_median()
            # preprocess.notch_filter(50)
            preprocess.bandpass_filter(1, 50)
            preprocess.discard_datapoints_below_or_over()
            # preprocess.discard_datapoints_by_ratio_to_median()
            # preprocess.fft()
            preprocess.min_max_scale()
            return preprocess.data, n_channels

        preprocessed_data, n_channels = preprocess(input.data)
        # Plot.plot_without_threshold(preprocessed_data)
        return preprocessed_data, mapping, n_channels

    data_train, mapping, n_channels = read_and_preprocess(path1)
    data_test, useless, useless = read_and_preprocess(path2, mapping)

    # [data_train, data_test] = Helpers.split_by_column_into_train_test(preprocessed_data)

    featureselect_train = FeatureSelect(data_train, n_channels <= constants.CHANNEL_TRESHOLD)
    featureselect_train.pca()

    featureselect_test = FeatureSelect(data_test, n_channels <= constants.CHANNEL_TRESHOLD)
    featureselect_test.pca()

    labels_train = Helpers.extract_labels_from_dataframe(data_train)
    labels_test = Helpers.extract_labels_from_dataframe(data_test)

    # @constants.timeit
    def classify_and_compute_precision(C=1.0, gamma='auto', with_plot=True):
        mode = 'voting'
        classify = Classify(featureselect_train.components, labels_train, mode=mode)
        classify.classify(['nearestneighbors', 'adaboost', 'randomforest'], 'soft', [3, 2, 1])

        params_string = ''
        if mode == 'svm':
            params_string = '[C=' + str(C) + '][gamma=' + str(gamma) + ']'

        predicted_labels_test = classify.predict(featureselect_test.components)

        if with_plot: Plot.plot_lists([
            {'data': labels_test, 'label': 'Expected' + params_string},
            {'data': predicted_labels_test, 'label': 'Predicted' + params_string}
        ])
        precision_obj = Precision(real_labels=labels_test, predicted_labels=predicted_labels_test)
        raw_precision = precision_obj.compute_raw_precision()
        cat_precision = precision_obj.compute_per_category_median_precision()
        print 'raw_precision', params_string, ' = ', raw_precision
        print 'cat_precision', params_string, ' = ', cat_precision

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
