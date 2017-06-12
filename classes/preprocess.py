import numpy as np
import constants
import pandas
from scipy import signal
from scipy import fftpack
from sklearn import preprocessing
from collections import OrderedDict
from helpers import Helpers


class Preprocess:
    def __init__(self, data, ignored_cols=constants.COLUMN_THRESHOLD):
        self.data = data
        self.ignored_cols = ignored_cols
        self.freq = constants.FREQ_HZ

    def remove_dc_offset(self, order=2):
        hp_cutoff_Hz = 1.0
        b, a = signal.butter(order, hp_cutoff_Hz / (self.freq / 2.0), 'highpass')
        for column in (x for x in self.data if x not in self.ignored_cols):
            self.data[column] = signal.lfilter(b, a, self.data[column], 0)
        if constants.verbose: print 'Removed DC offset'

    def bandpass_filter(self, start, stop, order=3):
        bp_Hz = np.array([start, stop])
        b, a = signal.butter(order, bp_Hz / (self.freq / 2.0), 'bandpass')
        for column in (x for x in self.data if x not in self.ignored_cols):
            self.data[column] = signal.lfilter(b, a, self.data[column], 0)
        if constants.verbose: print 'Bandpass filtered at:', start, stop

    def notch_filter(self, freq, order=2):
        bp_stop_Hz = np.array([freq - 1, freq + 1])  # set the stop band
        b, a = signal.butter(order, bp_stop_Hz / (self.freq / 2.0), 'bandstop')
        for column in (x for x in self.data if x not in self.ignored_cols):
            self.data[column] = signal.lfilter(b, a, self.data[column], 0)
        if constants.verbose: print 'Notch filtered at:', freq

    def resample(self, freq, column_name=constants.COLUMN_THRESHOLD):

        def do_resample(column, is_ignored):
            secs = len(column) / self.freq
            new_no_samples = int(secs * freq)
            if not is_ignored:
                return signal.resample(column, new_no_samples)
            return column[0:new_no_samples]

        new_data = OrderedDict()
        for cur_threshold in self.data[column_name].unique():
            indices = Helpers.get_indices_from_value(self.data, cur_threshold)
            if indices is None:
                break
            start_index = indices[0]
            end_index = indices[-1]

            i = 0
            for column in self.data:
                if column not in new_data:
                    new_data[column] = []
                batch_data = do_resample(self.data.iloc[start_index:end_index, i], column in self.ignored_cols)
                new_data[column].extend(batch_data)
                i += 1

        self.data = pandas.DataFrame(new_data)
        self.freq = freq

    def discard_columns_by_ratio_to_median(self, ratio=1.1):
        medians = {}
        for column in (x for x in self.data if x not in self.ignored_cols):
            medians[column] = self.data[column].abs().median()
        overall_average = np.mean(medians.values())
        for column, median in medians.iteritems():
            if median/overall_average > ratio:
                if constants.verbose: print 'Removed poorly recorded data for electrode', column
                self.data.drop(column, axis=1, inplace=True)

    def detrend(self):
        for column in (x for x in self.data if x not in self.ignored_cols):
            self.data[column] = signal.detrend(self.data[column])
        if constants.verbose: print 'Detrended'

    def convert_to_matrix(self):
        self.data = self.data.as_matrix(x for x in self.data.columns if x not in self.ignored_cols)

    def discard_datapoints_below_or_over(self, max_val=constants.FIFTY_MICROVOLTS):
        indexes_to_remove = []
        for index, row in self.data.iterrows():
            if any(val >= max_val or val <= -max_val for val in row[1:]):
                indexes_to_remove.append(index)

        self.data = self.data.drop(self.data.index[indexes_to_remove]).reset_index(drop=True)
        if constants.verbose: print 'Discarded datapoints below or over', max_val

    def discard_datapoints_by_ratio_to_median(self, ratio=100):
        indexes_to_remove = []
        no_columns = self.data.shape[1]
        min_value = [0] * no_columns
        for i in xrange(1, no_columns):
            min_value[i] = self.data.ix[:, i].abs().median()
        for index, row in self.data.iterrows():
            if any(row[i] >= min_value[i] * ratio or row[i] <= -min_value[i] * ratio for i in xrange(1, no_columns)):
                indexes_to_remove.append(index)
        # print indexes_to_remove
        # print len(indexes_to_remove)
        # print self.data.shape[0]
        # exit(0)
        self.data = self.data.drop(self.data.index[indexes_to_remove]).reset_index(drop=True)
        if constants.verbose: print 'Discarded data points by ratio', ratio, 'to median'

    def min_max_scale(self):
        result = self.data.copy()
        for column in (x for x in self.data if x not in self.ignored_cols):
            max_value = self.data[column].max()
            min_value = self.data[column].min()
            result[column] = (self.data[column] - min_value) / (max_value - min_value)
        self.data = result
        if constants.verbose: print 'Min max scaled between [0, 1]'

    def fft(self):
        for column in (x for x in self.data if x not in self.ignored_cols):
            self.data[column] = fftpack.fft(self.data[column])
        if constants.verbose: print 'Applied FFT'

    # def welch(self):
    #     for column in (x for x in self.data if x not in self.ignored_cols):
    #         import matplotlib.pyplot as plt
    #         f, Pxx_den = signal.welch(self.data[column])
    #         plt.semilogy(f, Pxx_den)
    #         plt.xlabel('frequency [Hz]')
    #         plt.ylabel('PSD [V**2/Hz]')
    #         plt.show()
    #         sdf
    #         self.data[column] = welched
    #     if constants.verbose: print 'Applied Welch spectral density'
