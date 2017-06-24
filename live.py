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
TRAIN_IMAGE_INTERVAL = 2
TEST_IMAGE_INTERVAL = 2
CROP = True
IGNORE_COLS = [constants.COLUMN_TIMESTAMP, constants.COLUMN_SENSOR_1, constants.COLUMN_SENSOR_2]

openbci = OpenBciThreadedTasks()
openbci.waitStreamReady().join()
print 'PYTHON: Stream is ready'


def get_most_common_prediction(predicted_labels):
    counts = np.bincount(predicted_labels)
    return np.argmax(counts)


def gather_data(subdir, randomize, duplicates):
    resultQueue = Queue.Queue()
    trainLogThread = []
    imageInterval = TRAIN_IMAGE_INTERVAL
    if subdir == 'test':
        imageInterval = TEST_IMAGE_INTERVAL

    def startLogging():
        print 'PYTHON: Starting displaying images and logging data'
        imageWindow.handleNextImage()
        trainLogThread.append(openbci.startLoggingToArray(resultQueue))

    def start():
        startLogging()
        duration = int(len(images) * imageInterval * 1500)
        tk.after(duration, stop)

    def stop():
        tk.destroy()

    tk = Tk()
    tk.withdraw()
    trainDir = DIR + '/' + subdir
    images = ImageHelpers.getImagesInDirectory(trainDir, randomize, duplicates)
    imageWindow = ImageHelpers.ImageWindow(trainDir, images, imageInterval, openbci, CROP)
    imageWindow.tkAndWindow(tk)
    start()
    tk.mainloop()
    trainLogThread[0].join()

    return resultQueue.get()


def input_preprocess_featureselect_arr(header, train_rows, mapping=None):
    def read_and_preprocess(header, train_rows):
        input = Input(ignore_cols=IGNORE_COLS)
        input.from_array(header, train_rows)
        input.make_column_uniform()
        threshold_to_filename = input.replace_column_with_thresholds(mapping=mapping)

        # Plot.plot_without_threshold(input.data)

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


def predict(header, test_rows, classification_model, threshold_to_filename):
    data_test, test_components, useless = input_preprocess_featureselect_arr(header, test_rows, mapping=threshold_to_filename)

    labels_test = Helpers.extract_labels_from_dataframe(data_test)

    predicted_labels_test = classification_model.predict(test_components)
    precision_obj = Precision(real_labels=labels_test, predicted_labels=predicted_labels_test)

    Plot.plot_lists([
        {'data': labels_test, 'label': 'Expected'},
        {'data': predicted_labels_test, 'label': 'Predicted'}
    ])

    raw_precision = precision_obj.compute_raw_precision()
    cat_precision = precision_obj.compute_per_category_median_precision()
    print 'raw_precision =', raw_precision
    print 'cat_precision =', cat_precision


class LivePredict:
    def __init__(self, classification_model, threshold_to_filename, randomize=False):
        self.classification_model = classification_model
        self.threshold_to_filename = threshold_to_filename
        self.resultQueue = Queue.Queue()
        self.test_log_thread = [None]
        self.randomize = randomize

        testDir = DIR + '/test'
        images = ImageHelpers.getImagesInDirectory(testDir, randomize=self.randomize)
        self.imageWindow = ImageHelpers.ImageWindow(testDir, images, TEST_IMAGE_INTERVAL,
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
            duration = int(100 + TRAIN_IMAGE_INTERVAL * 1000)
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


header, train_rows = gather_data('train', randomize=True, duplicates=3)
print 'PYTHON: Training data gathering done'
classification_model, threshold_to_filename = train(header, train_rows)
print 'PYTHON: Training model done'
sleep(5)
useless, test_rows = gather_data('test', randomize=True, duplicates=1)
print 'PYTHON: Test data gathering done'
# LivePredict(classification_model, threshold_to_filename, randomize=True)
predict(header, test_rows, classification_model, threshold_to_filename)

openbci.kill()
