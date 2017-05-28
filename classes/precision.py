import constants
import numpy as np


class Precision:
    def __init__(self, real_labels, predicted_labels):
        self.real_labels = real_labels
        self.predicted_labels = predicted_labels

    def compute_raw_precision(self):
        no_differences = sum(1 for i, j in zip(self.real_labels, self.predicted_labels) if i != j)
        size = len(self.real_labels)
        return (size - no_differences) / (size * 1.0)

    def get_most_common_prediction_for_category(self, treshold):
        indices = [i for i, x in enumerate(self.real_labels) if x == treshold]
        if len(indices) == 0:
            return None
        counts = np.bincount(self.predicted_labels[indices])
        return np.argmax(counts)

    def compute_per_category_median_precision(self, start_threshold=1):
        cur_threshold = start_threshold
        good_predictions = 0
        while 1:
            predicted_treshold = self.get_most_common_prediction_for_category(cur_threshold)
            if predicted_treshold is None:
                break
            elif predicted_treshold == cur_threshold:
                good_predictions += 1
            cur_threshold += 1
        return (good_predictions * 1.0) / (cur_threshold - start_threshold)
