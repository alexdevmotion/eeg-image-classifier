import pandas
import constants
import numpy as np
import re
from helpers import Helpers
from collections import OrderedDict


class Input:
    def __init__(self, ignore_cols=[]):
        self.ignore_cols = ignore_cols
        self.data = None

    def read_csv(self, path, index_col=None):
        self.path = path
        self.index_col = index_col
        self.data = pandas.read_csv(self.path, index_col=self.index_col, memory_map=True,
                                    usecols=[col for col in constants.ALL_COLUMNS if col not in self.ignore_cols])

    def from_array(self, header, rows):
        dfObject = OrderedDict()
        for i in xrange(0, len(header)):
            colName = header[i]
            if colName in self.ignore_cols:
                continue
            column = []
            for row in rows:
                column.append(row[i])
            dfObject[colName] = column
        self.data = pandas.DataFrame(dfObject)

    def make_column_uniform(self, column_name=constants.COLUMN_FILENAME):
        self.data[column_name] = self.data[column_name].apply(lambda filename: re.sub(r' *[\(\.].*', '', filename))
        if constants.verbose: print 'Made', column_name, 'column uniform'

    def read_csv_emotiv(self):
        self.data = pandas.read_csv(self.path, memory_map=True)

    def replace_column_with_thresholds(self, column_name=constants.COLUMN_FILENAME, start_threshold=1,
                                       new_column_name=constants.COLUMN_THRESHOLD, threshold_value=None):

        threshold_to_filename = {}
        if threshold_value:
            new_column = np.full([self.data.shape[0], 1], threshold_value)
            self.data.insert(0, new_column_name, new_column)
            self.data = self.data.drop(column_name, 1)
            #TODO nu cred ca mai chemam ramura asta, daca o chemam trebuie pus in threshold to filename
        else:
            cur_threshold = start_threshold
            uniqe_filenames = self.data[column_name].unique()
            for cur_val in uniqe_filenames:
                indices = Helpers.get_indices_from_value(self.data, cur_val, column_name=constants.COLUMN_FILENAME)
                if indices is None:
                    break
                for index in indices:
                    self.data.set_value(index, column_name, cur_threshold)
                threshold_to_filename[cur_threshold] = cur_val
                cur_threshold += 1
            self.data = self.data.rename(columns={column_name: new_column_name})

        if constants.verbose: print 'Replaced', column_name, 'column with', new_column_name, 'column'
        return threshold_to_filename
