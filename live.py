from classes.OpenBciTasks import OpenBciThreadedTasks
from classes import ImageHelpers
from Tkinter import *
import Queue
import numpy as np
from classes.input import Input
from classes.plot import Plot
from classes.preprocess import Preprocess
from classes.featureselect import FeatureSelect
from classes.helpers import Helpers
from classes.classify import Classify
from classes.precision import Precision
import classes.constants as constants
from time import sleep

DIR = 'experiments/colors'
IMAGE_INTERVAL = 2
CROP = True
IGNORE_COLS = [constants.COLUMN_TIMESTAMP, constants.COLUMN_SENSOR_1, constants.COLUMN_SENSOR_2]

openbci = OpenBciThreadedTasks()

openbci.waitStreamReady().join()
print 'PYTHON: Stream is ready'


def get_most_common_prediction(predicted_labels):
    counts = np.bincount(predicted_labels)
    return np.argmax(counts)


def gather_data(subdir):
    resultQueue = Queue.Queue()
    trainLogThread = []

    def startLogging():
        print 'PYTHON: Starting displaying images and logging data'
        imageWindow.handleNextImage()
        trainLogThread.append(openbci.startLoggingToArray(resultQueue))

    def start():
        startLogging()
        duration = 3000 + len(images) * IMAGE_INTERVAL * 1000
        tk.after(duration, stop)

    def stop():
        tk.destroy()

    tk = Tk()
    tk.withdraw()
    trainDir = DIR + '/' + subdir
    images = ImageHelpers.getImagesInDirectory(trainDir)
    imageWindow = ImageHelpers.ImageWindow(trainDir, images, IMAGE_INTERVAL,
                                           openbci, CROP)
    imageWindow.tkAndWindow(tk)
    start()
    tk.mainloop()
    trainLogThread[0].join()

    return resultQueue.get()


def input_preprocess_featureselect_arr(header, train_rows):
    def read_and_preprocess(header, train_rows):
        input = Input(ignore_cols=IGNORE_COLS)
        input.from_array(header, train_rows)
        input.make_column_uniform()
        threshold_to_filename = input.replace_column_with_thresholds()

        Plot.plot_without_threshold(input.data)

        def preprocess(data):
            preprocess = Preprocess(data)
            preprocess.remove_dc_offset()
            preprocess.resample(100)
            preprocess.detrend()
            preprocess.discard_columns_by_ratio_to_median()
            # preprocess.notch_filter(50)
            preprocess.bandpass_filter(1, 50)
            # preprocess.discard_datapoints_below_or_over()
            # preprocess.discard_datapoints_by_ratio_to_median()
            # preprocess.fft()
            preprocess.min_max_scale()
            return preprocess.data

        preprocessed_data = preprocess(input.data)
        # Plot.plot_without_threshold(preprocessed_data)
        return preprocessed_data, threshold_to_filename

    data, threshold_to_filename = read_and_preprocess(header, train_rows)

    featureselect = FeatureSelect(data)
    featureselect.pca()

    return data, featureselect.components, threshold_to_filename


def train(header, train_rows):

    data_train, train_components, threshold_to_filename = input_preprocess_featureselect_arr(header, train_rows)

    labels_train = Helpers.extract_labels_from_dataframe(data_train)

    mode = 'voting'
    classify = Classify(train_components, labels_train, mode=mode)
    classify.classify(['nearestneighbors', 'adaboost', 'randomforest'], 'soft', [3, 2, 1])
    return classify, threshold_to_filename


def predict(header, test_rows, classification_model):
    data_test, test_components, useless = input_preprocess_featureselect_arr(header, test_rows)

    labels_test = Helpers.extract_labels_from_dataframe(data_test)

    predicted_labels_test = classification_model.predict(test_components)
    precision_obj = Precision(real_labels=labels_test, predicted_labels=predicted_labels_test)
    raw_precision = precision_obj.compute_raw_precision()
    cat_precision = precision_obj.compute_per_category_median_precision()

    Plot.plot_lists([
        {'data': labels_test, 'label': 'Expected'},
        {'data': predicted_labels_test, 'label': 'Predicted'}
    ])

    print 'raw_precision', ' = ', raw_precision
    print 'cat_precision', ' = ', cat_precision


class LivePredict:
    def __init__(self, classification_model, threshold_to_filename):
        self.classification_model = classification_model
        self.threshold_to_filename = threshold_to_filename
        self.resultQueue = Queue.Queue()
        self.test_log_thread = [None]

        testDir = DIR + '/test'
        images = ImageHelpers.getImagesInDirectory(testDir)
        self.imageWindow = ImageHelpers.ImageWindow(testDir, images, IMAGE_INTERVAL,
                                               openbci, CROP)
        self.handleNextImage()

    def startLogging(self):
        imagesRemaining = self.imageWindow.handleNextImage(keep_going=False)
        if imagesRemaining:
            self.test_log_thread[0] = openbci.startLoggingToArray(self.resultQueue)
        return imagesRemaining

    def start(self):
        imagesRemaining = self.startLogging()
        if imagesRemaining:
            duration = 100 + IMAGE_INTERVAL * 1000
            self.tk.after(duration, self.stop)
        return imagesRemaining

    def stop(self):
        self.tk.destroy()

    def handleNextImage(self):
        self.tk = Tk()
        self.tk.withdraw()
        self.imageWindow.tkAndWindow(self.tk)
        imagesRemaining = self.start()
        if imagesRemaining:
            self.tk.mainloop()
            self.test_log_thread[0].join()
            header, test_rows = self.resultQueue.get()
            self.doPredict(header, test_rows)
            self.handleNextImage()

    def doPredict(self, header, test_rows):
        data, components_test, useless = input_preprocess_featureselect_arr(header, test_rows)
        predicted_labels = self.classification_model.predict(components_test)
        print predicted_labels
        most_common_prediction = get_most_common_prediction(predicted_labels)
        print 'Prediction:', self.threshold_to_filename[most_common_prediction]


header, train_rows = gather_data('train')
print 'PYTHON: Training data gathering done'
classification_model, threshold_to_filename = train(header, train_rows)
print 'PYTHON: Training model done'
sleep(5)
header, test_rows = gather_data('test')
print 'PYTHON: Test data gathering done'
# LivePredict(classification_model, threshold_to_filename)
predict(header, test_rows, classification_model)

openbci.kill()
