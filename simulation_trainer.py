# -*- coding: utf-8 -*-
from io import BytesIO
from tensorflow.python.lib.io import file_io
from tqdm.auto import tqdm
from configuration import Config
from Simulation.model import LuxuryDiceSimulationMdn as Model
from Simulation.loss import MdnLoss

import os
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

#
# Configuration Loading
# ----------------------------------------------------------------------------------------------------------------------
config = Config(os.path.join(os.getcwd(), "Simulation/config.yaml"))

# Set GPU as available physical device
if gpus := tf.config.experimental.list_physical_devices(device_type='GPU'):
    tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')

if __name__ == "__main__":
    # Data Preparation
    # x = np.random.rand(32, 15, 3)
    # x = tf.cast(x, tf.float32)
    # time_encode = np.random.rand(32, 2)
    # time_encode = tf.cast(time_encode, tf.float32)
    # y = np.random.rand(32, 13)
    # y = tf.cast(y, tf.float32)
    f = BytesIO(file_io.read_file_to_string(config.data.data_path, binary_mode=True))
    data = np.load(f)
    x, time_encode, y = data["record"], data["time_code"], data["y"]
    x = tf.cast(x, tf.float32)
    time_encode = tf.cast(time_encode, tf.float32)
    y = tf.cast(y, tf.float32)

    k = int(config.data.train_test_split * x.shape[0])
    x_train, x_test = x[:k, :, :], x[k:, :, :]
    time_encode_train, time_encode_test = time_encode[:k, :], time_encode[k:, :]
    y_train, y_test = y[:k, :], y[k:, :]

    training_data = tf.data.Dataset.from_tensor_slices((x_train, time_encode_train, y_train))
    training_batch = training_data.batch(config.train.batch_size)
    testing_data = tf.data.Dataset.from_tensor_slices((x_test, time_encode_test, y_test))
    testing_batch = testing_data.batch(config.train.batch_size)

    #
    # Create model (BetSimulation)
    # ----------------------------------------------------------------------------------------------------------------------
    model = Model()
    # model = tf.keras.models.load_model('weights/best_luxury_dice_simulation_model_backup.tf')
    model.rnn.kernel_regularizer.l2 = 0.01

    if config.optimizer.method == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=float(config.optimizer.learning_rate))
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=float(config.optimizer.learning_rate))

    #
    # Loss
    # ----------------------------------------------------------------------------------------------------------------------
    # kld = tf.keras.losses.KLDivergence()
    mdn_loss = MdnLoss(reduce=False)
    # mse = tf.keras.losses.MeanSquaredError()

    #
    # Train Model
    # ----------------------------------------------------------------------------------------------------------------------
    best_train_loss = float("inf")
    best_test_loss = float("inf")
    for e in range(config.train.epoch):
        train_loss_cache = []
        # train_mse_cache = []
        test_loss_cache = []
        # test_mse_cache = []
        for x, time, y in tqdm(training_batch, desc="Training"):
            with tf.GradientTape() as tape:
                y_p1, y_p2 = model(x, time, training=True)
                neg_log_pdf_normal = mdn_loss(y_p1[:, :1], y_p2[:, :1], y[:, :1], "Normal")
                neg_log_pdf_beta = mdn_loss(y_p1[:, 1:], y_p2[:, 1:], y[:, 1:], "Beta")
                neg_log_pdf_normal = tf.math.reduce_mean(neg_log_pdf_normal)
                neg_log_pdf_beta = tf.math.reduce_mean(neg_log_pdf_beta)
                focal_weight = config.model.focal_weight if neg_log_pdf_normal > 0 else 1 / config.model.focal_weight
                train_loss = neg_log_pdf_normal + neg_log_pdf_beta

            gradients = tape.gradient(train_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss_cache.append(train_loss.numpy())

        for x, time, y in tqdm(testing_batch, desc="Testing"):
            y_p1, y_p2 = model(x, time)
            neg_log_pdf_normal = mdn_loss(y_p1[:, :1], y_p2[:, :1], y[:, :1], "Normal")
            neg_log_pdf_beta = mdn_loss(y_p1[:, 1:], y_p2[:, 1:], y[:, 1:], "Beta")
            neg_log_pdf_normal = tf.math.reduce_mean(neg_log_pdf_normal)
            neg_log_pdf_beta = tf.math.reduce_mean(neg_log_pdf_beta)
            focal_weight = config.model.focal_weight if neg_log_pdf_normal > 0 else 1 / config.model.focal_weight
            test_loss = neg_log_pdf_normal + neg_log_pdf_beta
            test_loss_cache.append(test_loss.numpy())

        train_loss_epoch = np.mean(train_loss_cache)
        test_loss_epoch = np.mean(test_loss_cache)

        if test_loss_epoch < best_test_loss:
            # model.save('weights/best_luxury_dice_simulation_model.tf', save_format='tf')
            best_test_loss = test_loss_epoch
        print('Epoch: {}/{}\ttrain_loss: {:.6f}\ttest_loss: {:.6f}'.
              format(e + 1, config.train.epoch, train_loss_epoch, test_loss_epoch))
