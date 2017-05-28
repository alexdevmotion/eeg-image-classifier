import constants
import pandas


class Helpers:
    def __init__(self):
        pass

    @staticmethod
    # the supplied column name should have the same value for multiple consecutive rows
    # ratio = train / all
    def split_by_column_into_train_test(data, column_name=constants.COLUMN_THRESHOLD, ratio=2.0 / 3):
        last_val = ''
        start_index = 0
        data_train = pandas.DataFrame()
        data_test = pandas.DataFrame()
        for i, val in enumerate(data[column_name]):
            if val != last_val:
                last_val = val
                end_index = i - 1
                if end_index > 0:
                    [data_train, data_test] = Helpers.do_single_split(data, start_index, end_index,
                                                                      ratio, data_train, data_test)
                start_index = i
        if start_index > 0:  # do the same for the last frame; shape[0] = number of rows
            [data_train, data_test] = Helpers.do_single_split(data, start_index, data.shape[0] - 1,
                                                              ratio, data_train, data_test)
        return [data_train, data_test]

    @staticmethod
    def do_single_split(data, start_index, end_index, ratio, data_train, data_test):
        no_rows = end_index - start_index
        no_rows_train = int(no_rows * ratio)
        data_train = data_train.append(data.iloc[start_index:start_index + no_rows_train, :], ignore_index=True)
        data_test = data_test.append(data.iloc[start_index + no_rows_train + 1:end_index, :], ignore_index=True)
        return [data_train, data_test]

    @staticmethod
    def concatenate(data_1, data_2):
        return pandas.concat([data_1, data_2], ignore_index=True)

    @staticmethod
    def extract_labels_from_dataframe(data, use_col=constants.COLUMN_THRESHOLD):
        return data.as_matrix([use_col]).ravel()

    @staticmethod
    def get_indices_from_value(data, value, column_name=constants.COLUMN_THRESHOLD):
        indices = [i for i, x in enumerate(data[column_name]) if x == value]
        if len(indices):
            return indices
        return None