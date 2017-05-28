import pandas
import constants
import numpy as np
import re
from helpers import Helpers


class Input:
    def __init__(self, path, ignore_cols=[], index_col=None):
        self.path = path
        self.ignore_cols = ignore_cols
        self.index_col = index_col
        self.data = None

    def read_csv(self):
        self.data = pandas.read_csv(self.path, index_col=self.index_col, memory_map=True,
                                    usecols=[col for col in constants.ALL_COLUMNS if col not in self.ignore_cols])

    def make_column_uniform(self, column_name=constants.COLUMN_FILENAME):
        self.data[column_name] = self.data[column_name].apply(lambda filename: re.sub(r' *\(.*', '', filename))
        if constants.verbose: print 'Made', column_name, 'column uniform'

    def read_csv_emotiv(self):
        self.data = pandas.read_csv(self.path, memory_map=True)

    def replace_column_with_thresholds(self, column_name=constants.COLUMN_FILENAME, start_threshold=1,
                                       new_column_name=constants.COLUMN_THRESHOLD, threshold_value=None):

        if threshold_value:
            new_column = np.full([self.data.shape[0], 1], threshold_value)
            self.data.insert(0, new_column_name, new_column)
            self.data = self.data.drop(column_name, 1)
        else:
            cur_threshold = start_threshold
            for cur_val in self.data[column_name].unique():
                indices = Helpers.get_indices_from_value(self.data, cur_val, column_name=constants.COLUMN_FILENAME)
                if indices is None:
                    break
                for index in indices:
                    self.data.set_value(index, column_name, cur_threshold)
                cur_threshold += 1
            self.data = self.data.rename(columns={column_name: new_column_name})

        if constants.verbose: print 'Replaced', column_name, 'column with', new_column_name, 'column'
