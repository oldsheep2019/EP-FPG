import sys

import numpy as np
from numpy.random import uniform
from numpy.linalg import pinv

from neural_network_tools import get_activation_function


class ELM(object):
    def __init__(self, input_num, hidden_num, output_num, act_func='sig', cor='c'):
        self.input_num = input_num
        self.hidden_num = hidden_num
        self.output_num = output_num

        self.input_weights = uniform(low=-1, high=1, size=(input_num, hidden_num))
        self.bias_list = uniform(low=-1, high=1, size=hidden_num)
        self.output_weights = np.empty((hidden_num, output_num))

        self.act_func = get_activation_function(act_func)

        self.cor = cor

    # label should be in one-hot form
    def fit(self, data, label):
        if data.shape[1] != self.input_num:
            raise ValueError('the dimension of input feature data is not consistent with ELM input_num !')

        # n_samples = data.shape[0]
        # n_classes = label.shape[1]

        P, T = data, label

        if self.cor == 'c':
            T = T * 20 - 10  # map [0, 1] -> [-10, 10]

        # the matrix-row_vector addition operation broadcasting has been done
        # bias_list will be added to every row to the former matrix
        H = np.dot(P, self.input_weights) + self.bias_list

        # activate H
        H = self.act_func(H)

        self.output_weights = np.dot(pinv(H), T)

    def predict(self, data):
        if data.shape[1] != self.input_num:
            raise ValueError('the dimension of input feature data is not consistent with ELM input_num !')

        P = data
        H = np.dot(P, self.input_weights) + self.bias_list
        H = self.act_func(H)
        Y = np.dot(H, self.output_weights)
        predict_label = np.argmax(Y, axis=1) if self.cor == 'c' else Y

        # from data_tools import dense_to_one_hot
        # predict_label = dense_to_one_hot(predict_label, class_num=self.output_num)
        return predict_label

    def test_acc(self, data, label):
        predict_label = self.predict(data)
        if self.cor == 'c':
            acc = np.sum(predict_label == np.argmax(label, axis=1)) / len(label)
            return acc
        else:
            err_list = np.sum((predict_label - label) ** 2, axis=1)
            mean_err = np.sum(err_list) / len(label)
            return mean_err


def test_elm(data_set_name, specified_hidden_num=None):
    # load and split data
    if data_set_name == 'WIL':
        from data_tools import WIL_load_data
        from constants import WIL_FEATURE_DIMENSION, WIL_CLASS_NUM, WIL_HIDDEN_SIZE

        norm_train_data, norm_test_data = WIL_load_data()
        feature_num, hidden_num, class_num = WIL_FEATURE_DIMENSION, WIL_HIDDEN_SIZE, WIL_CLASS_NUM
        cor = 'c'
    else:
        from data_tools import BLE_load_data
        from constants import BLE_FEATURE_DIMENSION, BLE_LABEL_DIMENSION, BLE_HIDDEN_SIZE

        norm_train_data, norm_test_data = BLE_load_data()
        feature_num, hidden_num, class_num = BLE_FEATURE_DIMENSION, BLE_HIDDEN_SIZE, BLE_LABEL_DIMENSION
        cor = 'r'

    if specified_hidden_num is not None:
        hidden_num = specified_hidden_num

    # get train / test data
    train_data, train_label = norm_train_data[:, :feature_num], norm_train_data[:, feature_num:]
    test_data, test_label = norm_test_data[:, :feature_num], norm_test_data[:, feature_num:]

    elm = ELM(input_num=feature_num, hidden_num=hidden_num, output_num=class_num, cor=cor)
    elm.fit(data=train_data, label=train_label)

    train_acc = elm.test_acc(data=train_data, label=train_label)
    test_acc = elm.test_acc(data=test_data, label=test_label)
    print('train:', train_acc)
    print('test', test_acc)


if __name__ == '__main__':
    for hidden_num_ in (5, 10, 20, 100):
        test_elm('WIL', specified_hidden_num=hidden_num_)
        test_elm('BLE', specified_hidden_num=hidden_num_)

    sys.exit(0)
