import numpy as np
from math import ceil, floor

from constants import init_pos_min, init_pos_max


class weight_bias_matrix(object):
    def __init__(self, input_num, hidden_num, output_num):
        self.input_num = input_num
        self.hidden_num = hidden_num
        self.output_num = output_num

        n_rows = hidden_num
        n_cols = self.input_num + 1 + self.output_num + ceil(self.output_num / self.hidden_num)
        self.wb_mat = np.random.uniform(low=init_pos_min, high=init_pos_max, size=(n_rows, n_cols))

    def decode_wb_mat(self, name, beg, end=None, update_value=None):
        beg -= 1
        if end is not None:
            end -= 1

        if name in ('iw', 'inputweight'):
            row_idx, col_idx = end, beg
        elif name in ('hb', 'hiddenbias'):
            row_idx, col_idx = beg, self.input_num
        elif name in ('ow', 'outputweight'):
            row_idx, col_idx = beg, self.input_num + 1 + end
        elif name in ('ob', 'outputbias'):
            t_row, t_col = beg % self.hidden_num, floor(beg / self.hidden_num)
            row_idx, col_idx = t_row, self.input_num + 1 + self.output_num + t_col

        ret = self.wb_mat[row_idx][col_idx]
        if update_value is not None:
            self.wb_mat[row_idx][col_idx] = update_value

        return ret
