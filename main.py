import sys

import time
import pickle

from test_tools import test
from BP import test_bp
from plot_tools import my_plot

PARTICLE_NUM = 200
epoch_num = 30
t0 = range(epoch_num)
t1 = range(epoch_num + 1)


def save_history(history):
    time_str = time.strftime("%Y%m%d-%H%M%S")
    file_path = 'history/%s' % time_str
    with open(file_path, 'wb') as output_file:
        pickle.dump(history, output_file, pickle.HIGHEST_PROTOCOL)
    return time_str


def contrast_test(data_set_name):
    fpg_res = test(data_set_name, particle_num=PARTICLE_NUM, max_iter=epoch_num, pretrain=False)
    pretrained_fpg_res = test(data_set_name, particle_num=PARTICLE_NUM, max_iter=epoch_num, pretrain=True)

    bp_metrics_name = 'val_accuracy' if data_set_name == 'WIL' else 'val_mean_squared_error'
    bp_res = test_bp(data_set_name, epochs=epoch_num).history[bp_metrics_name]
    bp_res = bp_res[:1] + bp_res  # add one more element to  bp_res to make it have the same length with fpg results
    if data_set_name == 'WIL':
        bp_res = [ele * 100 for ele in bp_res]  # get acc percentage

    all_history = {'dataset': 'WIL', 'fpg': fpg_res, 'pfpg': pretrained_fpg_res, 'bp': bp_res}
    time_str = save_history(all_history)

    my_plot(data_set_name, all_history, x_range=t1, epoch_num=epoch_num, time_str=time_str, separated_figure=False)


if __name__ == '__main__':

    # 11: WIL pure FPG
    # 12: WIL pretrained FPG
    # 13: WIL BP

    # 21: BLE pure FPG
    # 22: BLE pretrained FPG
    # 23: BLE BP

    test_flag = 'WIL'
    # test_flag = 'BLE'
    # test_flag = 'read history'

    if test_flag == 11:
        test('WIL', particle_num=200, max_iter=5, pretrain=False)
    elif test_flag == 12:
        test('WIL', particle_num=200, max_iter=20, pretrain=True)
    elif test_flag == 13:
        train_result = test_bp('WIL')
        test_acc_list = train_result.history['val_accuracy']
    elif test_flag == 21:
        test('BLE', particle_num=300, max_iter=50, pretrain=False)
    elif test_flag == 22:
        test('BLE', particle_num=200, max_iter=20, pretrain=True)
    elif test_flag == 23:
        train_result = test_bp('WIL')
        test_acc_list = train_result.history['val_mean_squared_error']
    elif test_flag == 'WIL':
        contrast_test('WIL')
    elif test_flag == 'BLE':
        contrast_test('BLE')
    else:  # read history and plot
        # data_set_name_ = 'WIL'
        data_set_name_ = 'BLE'
        separated_figure = False

        file_path_ = 'history/WIL__20191123-155303' if data_set_name_ == 'WIL' else 'history/BLE__20191123-142119'
        with open(file_path_, 'rb') as f_handle:
            all_history_ = pickle.load(f_handle)

        my_plot(data_set_name_, all_history_, x_range=t1, epoch_num=epoch_num, separated_figure=True)

    sys.exit(0)
