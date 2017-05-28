import matplotlib.pyplot as plt


class Plot:
    def __init__(self):
        pass

    @staticmethod
    def plot_without_threshold(data):
        data.iloc[:, 1:].plot.line()
        plt.show()

    @staticmethod
    def plot_matrix(matrix):
        T = range(matrix.shape[0])

        for i in range(matrix.shape[1]):
            plt.plot(T, matrix[:, i])

        plt.show()

    @staticmethod
    def plot_lists(plot_obj_list):
        lines = []
        for plot_obj in plot_obj_list:
            lines.append(plt.plot(plot_obj['data'], label=plot_obj['label']))
        # plt.xlabel('frequency [Hz]')
        # plt.ylabel('PSD [V**2/Hz]')
        plt.legend()
        plt.show()
