import sys

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from data_tools import get_data


if __name__ == '__main__':

    data_set_name = 'WIL'
    # data_set_name = 'BLE'

    test_num = 3
    for neighbor_num in (1, 3, 5, 11):
        if data_set_name == 'WIL':
            acc_list = []
            for k in range(test_num):
                train_data, train_label, test_data, test_label = get_data(data_set_name)
                neigh = KNeighborsClassifier(n_neighbors=neighbor_num)
                neigh.fit(train_data, train_label)
                predict_label = neigh.predict(test_data)

                test_sample_num = test_data.shape[0]
                err_cnt = 0
                for i in range(test_sample_num):
                    if np.argmax(predict_label[i]) != np.argmax(test_label[i]):
                        err_cnt += 1

                acc = 1 - (err_cnt / test_sample_num)
                print('kNN test accuracy = %.1f%%' % (acc * 100))
                acc_list.append(acc)
            print(
                '\nkNN mean test accuracy (for k = %d) = %.2f%%\n' %
                (neighbor_num, sum(acc_list) / len(acc_list) * 100)
            )
        else:
            mse_list = []
            for k in range(test_num):
                train_data, train_label, test_data, test_label = get_data(data_set_name)
                neigh = KNeighborsRegressor(n_neighbors=3)
                neigh.fit(train_data, train_label)

                predict_label = neigh.predict(test_data)
                # error = predict_label - test_label
                mse = np.sum((predict_label - test_label) ** 2) / len(test_label)
                print('kNN test mse = %.3f' % mse)
                mse_list.append(mse)
            print(
                '\nkNN mean test mse (for k = %d) = %.2f\n' %
                (neighbor_num, sum(mse_list) / len(mse_list))
            )

    sys.exit(0)
