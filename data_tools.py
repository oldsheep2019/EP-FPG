import os
import sys
import csv
from copy import copy

import numpy as np

from constants import \
    WIL_FEATURE_DIMENSION, WIL_CLASS_NUM, WIL_HIDDEN_SIZE, \
    BLE_FEATURE_DIMENSION, BLE_LABEL_DIMENSION, BLE_SAMPLE_NUM, BLE_HIDDEN_SIZE


def dense_to_one_hot(label, class_num=None):
    if isinstance(label, int):
        one_hot_label = np.zeros(class_num)
        one_hot_label[label] = 1
    else:
        classes, y = np.unique(label, return_inverse=True)
        if class_num is None:
            class_num = len(classes)

        sample_num = len(y)
        one_hot_label = np.zeros((sample_num, class_num))
        for i in range(sample_num):
            one_hot_label[i, int(y[i])] = 1

    return one_hot_label


# WIL functions
def WIL_load_data(split=False, extend=False):
    data_list = WIL_load_data_file(split=split)
    train_data, test_data = WIL_split_train_test(data_list)

    norm_train_data = normalize_data(train_data, feature_dimension=WIL_FEATURE_DIMENSION, extend=extend)
    norm_test_data = normalize_data(test_data, feature_dimension=WIL_FEATURE_DIMENSION, extend=extend)

    norm_train_data[:, -1] -= 1
    norm_test_data[:, -1] -= 1

    train_label_one_hot = dense_to_one_hot(norm_train_data[:, -1])
    test_label_one_hot = dense_to_one_hot(norm_test_data[:, -1])

    norm_train_data = np.append(norm_train_data[:, :-1], train_label_one_hot, axis=1)
    norm_test_data = np.append(norm_test_data[:, :-1], test_label_one_hot, axis=1)

    return norm_train_data, norm_test_data


def WIL_load_data_file(split=False):
    FILE_PATH = 'data/wifi_localization.txt'
    if not os.path.exists(FILE_PATH):
        raise ValueError('data file path: %s is not exists!' % FILE_PATH)
    with open(FILE_PATH) as f_handle:
        lines = f_handle.readlines()
    f_handle.close()

    data_list = [[int(data) for data in line.split('\t')] for line in lines]
    data_list = list(filter(lambda data: 1 <= data[WIL_FEATURE_DIMENSION] <= WIL_CLASS_NUM, data_list))
    data_list = np.array(data_list)

    if split:
        features, labels = data_list[:, 0:WIL_FEATURE_DIMENSION], data_list[:, WIL_FEATURE_DIMENSION]
        return features, labels
    else:
        return data_list


def WIL_split_train_test(data_list):
    labels = data_list[:, WIL_FEATURE_DIMENSION]

    classes, y = np.unique(labels, return_inverse=True)
    class_num = len(classes)

    train_data, test_data = np.empty((0, WIL_FEATURE_DIMENSION + 1)), np.empty((0, WIL_FEATURE_DIMENSION + 1))
    for c_idx in range(class_num):
        class_data_list = copy(data_list[y == c_idx, :])

        sample_num = len(class_data_list)
        train_num = int(sample_num / 2)

        np.random.shuffle(class_data_list)

        train_data = np.append(train_data, class_data_list[:train_num], axis=0)
        test_data = np.append(test_data, class_data_list[train_num:], axis=0)

    np.random.shuffle(train_data)
    np.random.shuffle(test_data)

    return train_data, test_data


def normalize_data(data_list, feature_dimension, extend=False):
    ret_data_list = copy(data_list)
    features = ret_data_list[:, :feature_dimension]

    for j in range(feature_dimension):
        min_f, max_f = min(features[:, j]), max(features[:, j])
        if max_f > min_f:
            features[:, j] = (features[:, j] - min_f) / (max_f - min_f)

    # map [0, 1] -> [-1, +1]
    if extend:
        features = features * 2 - 1

    ret_data_list[:, :feature_dimension] = features
    return ret_data_list


# BLE functions
def BLE_load_data(split=False, extend=False):
    data_list = BLE_load_data_file(split=split)
    train_data, test_data = BLE_split_train_test(data_list)
    norm_train_data = normalize_data(train_data, feature_dimension=BLE_FEATURE_DIMENSION, extend=extend)
    norm_test_data = normalize_data(test_data, feature_dimension=BLE_FEATURE_DIMENSION, extend=extend)
    return norm_train_data, norm_test_data


def BLE_split_train_test(data_list):
    data_dict = {}
    for i in range(len(data_list)):
        key = tuple(data_list[i, -BLE_LABEL_DIMENSION:])
        if key in data_dict:
            data_dict[key] = np.append(data_dict[key], np.array([data_list[i]]), axis=0)
        else:
            data_dict[key] = np.array([data_list[i]])

    train_data_list, test_data_list = np.empty((0, BLE_FEATURE_DIMENSION + BLE_LABEL_DIMENSION)), \
                                      np.empty((0, BLE_FEATURE_DIMENSION + BLE_LABEL_DIMENSION))
    balance_flag = 'train'  # used to balance the number of samples in train and test data set
    for key, sample_list in data_dict.items():
        # randomly shuffle samples
        np.random.shuffle(sample_list)
        if len(sample_list) % 2 == 1:
            if balance_flag == 'train':
                train_data_list = np.append(train_data_list, sample_list[-1:], axis=0)
                balance_flag = 'test'
            else:
                test_data_list = np.append(test_data_list, sample_list[-1:], axis=0)
                balance_flag = 'train'
            # truncate sample_list
            sample_list = sample_list[:-1]

        train_data_list = np.append(train_data_list, sample_list[0::2], axis=0)
        test_data_list = np.append(test_data_list, sample_list[1::2], axis=0)

    np.random.shuffle(train_data_list)
    np.random.shuffle(test_data_list)

    return train_data_list, test_data_list


def BLE_load_data_file(split=False):
    file_path = 'data/iBeacon_RSSI_Labeled.csv'
    if not os.path.exists(file_path):
        raise ValueError('data file path: %s is not exists!' % file_path)
    with open(file_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(csv_reader)
        data_list = np.empty((BLE_SAMPLE_NUM, BLE_FEATURE_DIMENSION + BLE_LABEL_DIMENSION))
        for i, row in enumerate(csv_reader):
            # print(', '.join(row))
            loc_str = row[0]
            x, y = BLE_parse_location_string(loc_str)
            beacon_list = row[2:]
            data_list[i] = np.array([beacon_list + [x, y]])

    if split:
        features, labels = data_list[:, :BLE_FEATURE_DIMENSION], data_list[:, BLE_FEATURE_DIMENSION:]
        return features, labels
    else:
        return data_list


def BLE_parse_location_string(loc_str):
    x = ord(loc_str[0]) - ord('D')
    y = int(loc_str[1:]) - 1
    return x, y


# for both WIL and BLE
def get_data(data_set_name):
    if data_set_name == 'WIL':
        feature_dimension, hidden_size, class_num = WIL_FEATURE_DIMENSION, WIL_HIDDEN_SIZE, WIL_CLASS_NUM
        norm_train_data, norm_test_data = WIL_load_data()
    else:
        feature_dimension, hidden_size, class_num = BLE_FEATURE_DIMENSION, BLE_HIDDEN_SIZE, BLE_LABEL_DIMENSION
        norm_train_data, norm_test_data = BLE_load_data()

    train_data, train_label = norm_train_data[:, :feature_dimension], norm_train_data[:, feature_dimension:]
    test_data, test_label = norm_test_data[:, :feature_dimension], norm_test_data[:, feature_dimension:]

    return train_data, train_label, test_data, test_label


if __name__ == '__main__':
    # WIL
    # norm_train_data_, norm_test_data_ = WIL_load_data()

    # BLE
    norm_train_data_, norm_test_data_ = BLE_load_data()
    sys.exit(0)
