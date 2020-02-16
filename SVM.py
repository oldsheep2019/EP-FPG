import sys

import numpy as np

from sklearn.svm import SVC, SVR

from data_tools import get_data


if __name__ == '__main__':
    # data_set_name = 'WIL'
    data_set_name = 'BLE'

    test_rounds = 3

    if data_set_name == 'WIL':
        for parameters in \
                ({'kernel': 'linear'},
                 {'kernel': 'poly', 'degree': 8, 'gamma': 'scale'},
                 {'kernel': 'rbf', 'gamma': 'scale'},
                 {'kernel': 'sigmoid', 'gamma': 'scale'}):
            acc_list = []
            for k in range(test_rounds):
                svc = SVC(**parameters)

                train_data, train_label, test_data, test_label = get_data(data_set_name)
                train_label = np.argmax(train_label, axis=1)
                test_label = np.argmax(test_label, axis=1)

                svc.fit(train_data, train_label)
                predict_label = svc.predict(test_data)

                err_cnt = 0
                for i in range(len(test_label)):
                    if predict_label[i] != test_label[i]:
                        err_cnt += 1
                acc = 1 - (err_cnt / len(test_label))
                print('svc test accuracy = %.1f%%' % (acc * 100))
                acc_list.append(acc)
            print(
                '\nsvc mean test accuracy (for %s) = %.2f%%\n' %
                (parameters, sum(acc_list) / len(acc_list) * 100)
            )
    else:
        for parameters in \
                ({'kernel': 'linear'},
                 {'kernel': 'poly', 'degree': 8, 'gamma': 'scale'},
                 {'kernel': 'rbf', 'gamma': 'scale'},
                 {'kernel': 'sigmoid', 'gamma': 'scale'}):
            mse_list = []

            for k in range(test_rounds):
                svr_x = SVR(**parameters)
                svr_y = SVR(**parameters)

                train_data, train_label, test_data, test_label = get_data(data_set_name)
                train_x_label, train_y_label = train_label[:, 0], train_label[:, 1]

                svr_x.fit(train_data, train_x_label)
                svr_y.fit(train_data, train_y_label)
                predict_label = np.vstack((svr_x.predict(test_data), svr_y.predict(test_data))).T

                # error = predict_label - test_label
                mse = np.sum((predict_label - test_label) ** 2) / len(test_label)
                print('svr test mse = %.3f' % mse)
                mse_list.append(mse)
            print(
                '\nsvr mean test mse (for %s) = %.2f\n' %
                (parameters, sum(mse_list) / len(mse_list))
            )
    sys.exit(0)
