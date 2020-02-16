import sys

import numpy as np
from numpy import exp, sin


def get_activation_function(func_name='sig'):
    func = None
    if func_name in ('sig', 'sigm', 'sigmoid'):
        def sig(x):
            return 1 / (1 + exp(-x))
            # return 1 / (1 + exp(-np.clip(x, -10, 10, out=x)))

        func = sig
    elif func_name in ('sin', ):
        func = sin

    return func


if __name__ == '__main__':
    sys.exit(0)
