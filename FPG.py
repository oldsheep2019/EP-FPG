import numpy as np
from numpy import exp
from numpy.random import uniform

from math import ceil, floor

from copy import copy

from constants import *
from neural_network_tools import get_activation_function


# functions for feedforward neural network
def erase_extra_dimension(mat, particle_num, hidden_num, output_num):
    extra_num = (hidden_num - (output_num % hidden_num)) % hidden_num
    for p_i in range(particle_num):
        for e_i in range(extra_num):
            mat[-(e_i + 1), -1, p_i] = 0


def feed_forward(feature, weight_mat, bias_vec, act_func=None):
    # append a +1 to the end of feature list for multiplying bias
    extended_feature = np.append(feature, [1])

    # concatenate biases to the weight matrix
    wb_mat = np.append(weight_mat, bias_vec.T, axis=0)

    output = np.dot(extended_feature, wb_mat)
    if act_func is not None:
        output = act_func(output)
    return output


# functions for PSO GSA
def get_inertia_weight(iter_idx, max_iter):
    return iw_beg + iter_idx / max_iter * (iw_end - iw_beg)


def cal_wb_mat_size(input_num, hidden_num, output_num):
    n_rows = hidden_num
    n_cols = input_num + 1 + output_num + ceil(output_num / hidden_num)
    return n_rows, n_cols


def encode_pretrained_weights(input_weights, hidden_biases, output_weights, output_biases):
    input_num, hidden_num = input_weights.shape
    output_num = output_weights.shape[1]

    n_rows, n_cols = cal_wb_mat_size(input_num, hidden_num, output_num)
    wb_mat = np.zeros((n_rows, n_cols))

    # encode NN weights and biases to wb_mat
    wb_mat[:, :input_num] = input_weights.T
    # wb_mat[:, input_num] = np.reshape(hidden_biases, (hidden_num, 1))
    wb_mat[:, input_num] = hidden_biases
    wb_mat[:, (input_num + 1):(input_num + 1 + output_num)] = output_weights

    for k in range(output_num):
        i, j = k % hidden_num, floor(k / hidden_num)
        wb_mat[i, input_num + 1 + output_num + j] = output_biases[k]

    return wb_mat


class FPG(object):
    def __init__(
            self,
            particle_num,
            input_num, hidden_num, output_num, af='sig',
            activate_output=True,
            cor='c',
            pretrained_wb_mats=None
    ):
        self.input_num = input_num
        self.hidden_num = hidden_num
        self.output_num = output_num
        self.particle_num = particle_num

        self.act_func = get_activation_function(af)

        self.activate_output = activate_output
        self.cor = cor

        # record cur_best_fitness, gbest_fitness, train_acc, test_acc for each iteration
        self.history = {'cur_best_err': [], 'gbest_err': [], 'train_acc': [], 'test_acc': []}

        self.n_rows, self.n_cols = cal_wb_mat_size(input_num, hidden_num, output_num)
        self.weight_bias_matrices = uniform(
            low=init_pos_min, high=init_pos_max, size=(self.n_rows, self.n_cols, self.particle_num)
        )
        erase_extra_dimension(self.weight_bias_matrices, self.particle_num, self.hidden_num, self.output_num)

        # calculate vel_mats
        self.vel_mats = uniform(low=init_vel_min, high=init_vel_max, size=self.weight_bias_matrices.shape)

        if pretrained_wb_mats is not None:
            pmat_num = pretrained_wb_mats.shape[2]
            self.weight_bias_matrices[:, :, :pmat_num] = pretrained_wb_mats
            self.vel_mats[:, :, :pmat_num] /= PVD

    def predict(self, feature, p_idx=0, particle=None):
        if particle is not None:
            wb_mat = copy(particle)
        else:
            wb_mat = self.weight_bias_matrices[:, :, p_idx]

        input_weights = wb_mat[:, :self.input_num].T
        hidden_biases = np.reshape(wb_mat[:, self.input_num], (self.hidden_num, 1))
        output_weights = wb_mat[:, (self.input_num + 1):(self.input_num + 1 + self.output_num)]

        output_biases = np.empty((self.output_num, 1))
        for k in range(self.output_num):
            i, j = k % self.hidden_num, floor(k / self.hidden_num)
            output_biases[k] = wb_mat[i, self.input_num + 1 + self.output_num + j]

        H_output = feed_forward(feature, input_weights, hidden_biases, self.act_func)
        O_output = feed_forward(H_output, output_weights, output_biases,
                                self.act_func if self.activate_output else None
                                )

        return O_output

    # classification or regression
    def test_acc(self, data_list, particle):
        # fitness = self.fitness(data_list, particle=particle)
        sample_num = len(data_list)
        err_cnt = 0
        for sample_idx in range(sample_num):
            feature, label = data_list[sample_idx, :self.input_num], \
                             data_list[sample_idx, self.input_num:]
            predict_label = self.predict(feature=feature, particle=particle)
            if self.cor == 'c':
                if np.argmax(predict_label) != np.argmax(label):
                    err_cnt += 1
            else:
                err_cnt += sum((predict_label - label) ** 2)
        if self.cor == 'c':
            return 1 - err_cnt / sample_num
        else:
            return err_cnt / sample_num

    def fitness(self, data_list, p_idx=0, particle=None):
        sample_num = len(data_list)
        error_list = np.zeros(sample_num)
        for k in range(sample_num):
            feature, label = data_list[k, :self.input_num], data_list[k, self.input_num:]
            output = self.predict(feature, p_idx=p_idx, particle=particle)
            desired_output = label  # one_hot_label
            # calculate error
            error_list[k] = sum((output - desired_output) ** 2)
        total_error = sum(error_list) / sample_num
        fitness_ = -1 * total_error
        return fitness_

    def fit(self, data_list, max_iter=10, test_data_list=None):
        # initialization
        vel_mats = self.vel_mats
        erase_extra_dimension(vel_mats, self.particle_num, self.hidden_num, self.output_num)

        fitness_list = np.zeros(self.particle_num)
        gbest_particle, gbest_fitness = None, None

        mass_list = np.zeros(self.particle_num)
        R_mat = np.zeros((self.particle_num, self.particle_num))
        F_mat = np.zeros((self.n_rows, self.n_cols, self.particle_num, self.particle_num))
        # JF_mat = np.zeros((self.n_rows, self.n_cols, self.particle_num))
        accel_mat = np.zeros((self.n_rows, self.n_cols, self.particle_num))

        for iter_idx in range(max_iter + 1):
            # calculate fitness for each particle
            for p_idx in range(self.particle_num):
                fitness_list[p_idx] = self.fitness(data_list=data_list, p_idx=p_idx)

            # update gbest fitness and corresponding particle position
            best_p_idx, worst_p_idx = np.argmax(fitness_list), np.argmin(fitness_list)

            # for test
            # temp = self.fitness(data_list=data_list, p_idx=best_p_idx)

            cur_best_fitness, cur_worst_fitness = fitness_list[best_p_idx], fitness_list[worst_p_idx]
            if (gbest_fitness is None) or (cur_best_fitness > gbest_fitness):
                gbest_fitness = cur_best_fitness
                gbest_particle = copy(self.weight_bias_matrices[:, :, best_p_idx])

            # calculate training acc each round
            train_data_acc = self.test_acc(data_list, gbest_particle)
            format_str = '%d: %.3f %.3f'
            if self.cor == 'c':
                format_str += ' %.1f%%'
                train_data_acc *= 100
            else:
                format_str += ' %.3f'

            print_list = [iter_idx, -gbest_fitness, -cur_best_fitness, train_data_acc]

            test_data_acc = None
            if test_data_list is not None:
                test_data_acc = self.test_acc(test_data_list, gbest_particle)
                format_str += ' %.3f'
                if self.cor == 'c':
                    test_data_acc *= 100
                    format_str += '%%'
                print_list.append(test_data_acc)

            print(format_str % tuple(print_list))
            # save history
            self.history['cur_best_err'].append(-cur_best_fitness)
            self.history['gbest_err'].append(-gbest_fitness)
            self.history['train_acc'].append(train_data_acc)
            self.history['test_acc'].append(test_data_acc)

            if abs(gbest_fitness) < 0.01:
                break

            if iter_idx == max_iter:
                break

            # particles move
            self.weight_bias_matrices += vel_mats

            # calculate accelerations
            G = G0 * exp(-alpha * iter_idx / max_iter)
            # calculate distances between particles
            for i in range(self.particle_num):
                for j in range(self.particle_num):
                    R_mat[i, j] = np.linalg.norm(
                        self.weight_bias_matrices[:, :, i] - self.weight_bias_matrices[:, :, j]
                    )
            # calculate masses of particles
            for p_idx in range(self.particle_num):
                mass_list[p_idx] = (fitness_list[p_idx] - cur_worst_fitness) / (cur_best_fitness - cur_worst_fitness)
            total_mass = np.sum(mass_list)
            mass_list /= total_mass

            # calculate Gravitational Forces between particles
            for i in range(self.particle_num):
                for j in range(self.particle_num):
                    # the actual mass for particle with index=worst_p_idx is 0
                    # and later we will need to do F / mass
                    # so we need to use a fake mass here to offset this 0 mass in (F / mass)
                    # mass_i = fake_worst_mass if i == worst_p_idx else mass_list[i]
                    mass_i = 1
                    F_mat[:, :, i, j] = \
                        G * \
                        mass_i * mass_list[j] / (R_mat[i, j] + epsilon) * \
                        (self.weight_bias_matrices[:, :, j] - self.weight_bias_matrices[:, :, i])
            # joint force
            JF_mat = np.dot(F_mat, uniform(size=self.particle_num))
            # JF_mat = JF_mat.reshape((self.n_rows, self.n_cols, self.particle_num))

            # calculate accelerations of particles
            for p_idx in range(self.particle_num):
                # to deal with the mass = 0 for the worst particle
                # mass_i = fake_worst_mass if p_idx == worst_p_idx else mass_list[p_idx]
                mass_i = 1
                accel_mat[:, :, p_idx] = JF_mat[:, :, p_idx] / mass_i

            # calculate new velocities
            new_vel_mat = np.empty(vel_mats.shape)
            iw_cur = get_inertia_weight(iter_idx, max_iter)
            for p_idx in range(self.particle_num):
                new_vel_mat[:, :, p_idx] = iw_cur * vel_mats[:, :, p_idx] + \
                                           Cp1 * uniform() * accel_mat[:, :, p_idx] + \
                                           Cp2 * uniform() * (gbest_particle - self.weight_bias_matrices[:, :, p_idx])
            vel_mats = new_vel_mat

        return gbest_particle
