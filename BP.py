import sys

# import numpy as np
# from math import sqrt

from keras.layers import Dense, Activation
from keras.models import Sequential

from constants import WIL_FEATURE_DIMENSION, WIL_HIDDEN_SIZE, WIL_CLASS_NUM, \
                      BLE_FEATURE_DIMENSION, BLE_HIDDEN_SIZE, BLE_LABEL_DIMENSION
from data_tools import WIL_load_data, BLE_load_data


def generate_classification_model(hidden_size=None):
    if hidden_size is None:
        hidden_size = WIL_HIDDEN_SIZE

    classification_model = Sequential([
        Dense(hidden_size, input_shape=(WIL_FEATURE_DIMENSION,)),
        Activation('sigmoid'),
        Dense(WIL_CLASS_NUM),
        Activation('softmax'),
    ])

    classification_model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return classification_model


def generate_regression_model(hidden_size=None):
    if hidden_size is None:
        hidden_size = BLE_HIDDEN_SIZE

    regression_model = Sequential([
        Dense(hidden_size, input_shape=(BLE_FEATURE_DIMENSION,)),
        Activation('sigmoid'),
        Dense(BLE_LABEL_DIMENSION),
        # Activation('softmax'),
    ])

    regression_model.compile(
        optimizer='rmsprop',
        loss='mean_squared_error',
        metrics=['mean_squared_error']
    )

    return regression_model


def test_bp(data_set_name, epochs=30, hidden_size=None):
    if data_set_name == 'WIL':
        feature_dim = WIL_FEATURE_DIMENSION
        norm_train_data, norm_test_data = WIL_load_data()
    else:
        feature_dim = BLE_FEATURE_DIMENSION
        norm_train_data, norm_test_data = BLE_load_data()

    train_data, train_labels = norm_train_data[:, :feature_dim], norm_train_data[:, feature_dim:]
    test_data, test_labels = norm_test_data[:, :feature_dim], norm_test_data[:, feature_dim:]

    if data_set_name == 'WIL':
        classification_model = generate_classification_model(hidden_size=hidden_size)
        history = classification_model.fit(
            train_data, train_labels,
            batch_size=32, shuffle=True, epochs=epochs,
            validation_data=(test_data, test_labels)
        )
    else:
        regression_model = generate_regression_model(hidden_size=hidden_size)
        history = regression_model.fit(
            train_data, train_labels,
            batch_size=32, shuffle=True, epochs=epochs,
            validation_data=(test_data, test_labels)
        )

        # pl = regression_model.predict(test_data)
        # err_list = np.sum((pl - test_labels) ** 2, axis=1)
        # se_list = [sqrt(e) for e in err_list]

    return history


if __name__ == '__main__':

    # data_set_name_ = 'WIL'
    data_set_name_ = 'BLE'

    for hidden_size_ in (5, 10, 20, 100):
        history_ = test_bp(data_set_name_, hidden_size=hidden_size_)

    sys.exit(0)
