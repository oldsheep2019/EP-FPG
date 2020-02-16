
# for WIL
# data dimension
WIL_FEATURE_DIMENSION = 7
WIL_CLASS_NUM = 4

# for BLE
# data dimension
BLE_FEATURE_DIMENSION = 13
BLE_LABEL_DIMENSION = 2
BLE_SAMPLE_NUM = 1420
BLE_X_MIN, BLE_X_MAX = 0, 19  # 'D' -- 'W'
BLE_Y_MIN, BLE_Y_MAX = 0, 14  # 1 -- 15


# neural network
WIL_HIDDEN_SIZE = 5
PRETRAINED_PARTICLE_NUM = 10

BLE_HIDDEN_SIZE = 7

# for all
iw_beg, iw_end = 0.9, 0.4
init_vel_min, init_vel_max = -1, 1
init_pos_min, init_pos_max = -1, 1
PVD = 30  # pretrained particles initial velocity damping factor

# PSO
C1, C2 = 2, 2
# r1_min, r1_max = 0, 1
# r2_min, r2_max = 0, 1

# GSA
alpha = 20
G0 = 1
epsilon = 0.1
fake_worst_mass = 1
init_accel, init_mass = 0, 0

# PSOGSA
Cp1, Cp2 = 1, 1
