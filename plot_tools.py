import sys

import matplotlib.pyplot as plt


def my_plot(data_set_name, all_history, x_range, epoch_num, time_str='', separated_figure=True):

    # all_history = {'dataset': 'WIL', 'fpg': fpg_res, 'pfpg': pretrained_fpg_res, 'bp': bp_res}
    # time_str = save_history(all_history)

    x_label = 'training iteration'
    y_label = 'accuracy (%)' if data_set_name == 'WIL' else 'mean squared error'

    # fpg vs pfpg vs bp
    if separated_figure:
        plt.figure(dpi=1200)
    else:
        plt.subplot(1, 2, 1)

    plt.plot(x_range, all_history['bp'], 'b--', label='BP')
    plt.plot(x_range, all_history['fpg'].get('test_acc'), 'g:', label='FPG')
    plt.plot(x_range, all_history['pfpg'].get('test_acc'), 'r-', label='EP-FPG')

    plt.legend()

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if separated_figure:
        f_name = 'figures/%s/fpg_vs_bp__%s.png' % (data_set_name, time_str)
        #plt.savefig(f_name, bbox_inches='tight', pad_inches=0)
        plt.show()


    # pfpg vs elm
    elm_acc_arr = all_history['pfpg'].get('elm_acc_arr')
    best_elm_metrics = max(elm_acc_arr[:, 1]) * 100 if data_set_name == 'WIL' else min(elm_acc_arr[:, 1])

    if separated_figure:
        plt.figure(dpi=1200)
    else:
        plt.subplot(1, 2, 2)

    plt.plot(x_range, [best_elm_metrics] * (epoch_num + 1), 'b--', label='ELM')
    plt.plot(x_range, all_history['pfpg'].get('test_acc'), 'r-', label='EP-FPG')

    plt.legend()

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if separated_figure:
        f_name = 'figures/%s/pfpg_vs_elm__%s.png' % (data_set_name, time_str)
    else:
        f_name = 'figures/%s/merged__%s.png' % (data_set_name, time_str)
    plt.savefig(f_name, bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    sys.exit(0)
