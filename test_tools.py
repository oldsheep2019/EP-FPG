import sys

import copy
from statistics import mean
import numpy as np

from constants import \
    WIL_FEATURE_DIMENSION, WIL_HIDDEN_SIZE, WIL_CLASS_NUM, \
    BLE_FEATURE_DIMENSION, BLE_HIDDEN_SIZE, BLE_LABEL_DIMENSION, \
    PRETRAINED_PARTICLE_NUM

from data_tools import WIL_load_data, BLE_load_data
from FPG import FPG, cal_wb_mat_size, encode_pretrained_weights
from ELM import ELM


def test(data_set_name, particle_num, max_iter, pretrain=False):
    if data_set_name == 'WIL':
        feature_dimension, hidden_size, class_num = WIL_FEATURE_DIMENSION, WIL_HIDDEN_SIZE, WIL_CLASS_NUM
        cor = 'c'  # classification
        activate_output = True
        norm_train_data, norm_test_data = WIL_load_data()
    else:
        feature_dimension, hidden_size, class_num = BLE_FEATURE_DIMENSION, BLE_HIDDEN_SIZE, BLE_LABEL_DIMENSION
        cor = 'r'  # regression
        activate_output = False
        norm_train_data, norm_test_data = BLE_load_data()

    n_rows, n_cols = cal_wb_mat_size(feature_dimension, hidden_size, class_num)

    train_data, train_label = norm_train_data[:, :feature_dimension], norm_train_data[:, feature_dimension:]
    test_data, test_label = norm_test_data[:, :feature_dimension], norm_test_data[:, feature_dimension:]

    pretrained_wb_mats = None
    elm_acc_arr = None
    if pretrain:
        pretrained_wb_mats = np.empty((n_rows, n_cols, PRETRAINED_PARTICLE_NUM))

        # train elm(s)
        elm_acc_arr = np.empty((0, 2))
        for i in range(PRETRAINED_PARTICLE_NUM):
            elm = ELM(input_num=feature_dimension, hidden_num=hidden_size, output_num=class_num, cor=cor)
            elm.fit(train_data, train_label)
            elm_train_acc = elm.test_acc(data=train_data, label=train_label)
            elm_test_acc = elm.test_acc(data=test_data, label=test_label)

            elm_acc_arr = np.append(elm_acc_arr, [[elm_train_acc, elm_test_acc]], axis=0)

            if cor == 'c':
                print(
                    'pre-training ELM %02d, train acc = %.2f%%, test acc = %.2f%%' %
                    (i, elm_train_acc * 100, elm_test_acc * 100)
                )
            else:
                print(
                    'pre-training ELM %02d, train mean error = %.2f, test mean error = %.2f' %
                    (i, elm_train_acc, elm_test_acc)
                )

            pretrained_wb_mats[:, :, i] = encode_pretrained_weights(
                elm.input_weights, elm.bias_list, elm.output_weights,
                [0] * class_num
            )

        # print total avg/max/min train/test acc for elm(s)
        if cor == 'c':
            print(
                'train acc -- avg: %.2f%%, max: %.2f%%, min: %.2f%%' %
                (mean(elm_acc_arr[:, 0]) * 100, max(elm_acc_arr[:, 0]) * 100, min(elm_acc_arr[:, 0]) * 100)
            )
            print(
                'test acc -- avg: %.2f%%, max: %.2f%%, min: %.2f%%' %
                (mean(elm_acc_arr[:, 1] * 100), max(elm_acc_arr[:, 1]) * 100, min(elm_acc_arr[:, 1]) * 100)
            )
        else:
            print(
                'train mean error -- avg: %.2f, max: %.2f, min: %.2f' %
                (mean(elm_acc_arr[:, 0]), max(elm_acc_arr[:, 0]), min(elm_acc_arr[:, 0]))
            )
            print(
                'test mean error -- avg: %.2f, max: %.2f, min: %.2f' %
                (mean(elm_acc_arr[:, 1]), max(elm_acc_arr[:, 1]), min(elm_acc_arr[:, 1]))
            )

    # create FPG (with pretrained_wb_mats if it is not None)
    fpg = FPG(
        particle_num=particle_num,
        input_num=feature_dimension, hidden_num=hidden_size, output_num=class_num,
        activate_output=activate_output,
        cor=cor,
        pretrained_wb_mats=pretrained_wb_mats
    )

    # train fpg
    gbest_particle = fpg.fit(norm_train_data, max_iter=max_iter, test_data_list=norm_test_data)

    train_acc = fpg.test_acc(norm_train_data, particle=gbest_particle)
    test_acc = fpg.test_acc(norm_test_data, particle=gbest_particle)

    if cor == 'c':
        print('train acc = %.3f%%' % (train_acc * 100))
        print('test acc = %.3f%%' % (test_acc * 100))
    else:
        print('train mean error = %.3f' % train_acc)
        print('test mean error = %.3f' % test_acc)

    train_result = copy.deepcopy(fpg.history)
    train_result['elm_acc_arr'] = elm_acc_arr if pretrain else None  # add pretrain elm acc info
    return train_result


if __name__ == '__main__':
    sys.exit(0)
