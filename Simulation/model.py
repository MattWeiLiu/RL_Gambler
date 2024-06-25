from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


class LuxuryDiceSimulation(keras.Model):

    def __init__(self, units=13):
        super(LuxuryDiceSimulation, self).__init__()

        self.rnn = layers.LSTM(10, kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0, l2=0.3))
        self.fc = layers.Dense(units, activation='relu')

    def call(self, x, time_encode, training=False):
        """
        :param x.shape: (batch_size, sequential_length, vector_dim)
        :param time: [sin, cosine]
        :return y shape: (batch_size, num_gaussian*2 + population)
        """
        x = self.rnn(x)
        x = tf.concat([x, time_encode], axis=1)
        y = self.fc(x)
        return tf.concat([y[:, :1], tf.nn.softmax(y[:, 1:])], 1)


class LuxuryDiceSimulationMdn(keras.Model):

    def __init__(self, units=13):
        super(LuxuryDiceSimulationMdn, self).__init__()

        self.min_sigma = 1e-7
        self.units = units
        self.rnn = layers.LSTM(20, kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0, l2=0.05))
        self.fc = layers.Dense(units * 2, activation='relu')

    def call(self, x, time_encode, training=False):
        """
        :param x.shape: (batch_size, sequential_length, vector_dim)
        :param time: [sin, cosine]
        :return y shape: (batch_size, num_gaussian*2 + population)
        """
        x = self.rnn(x)
        x = tf.concat([x, time_encode], axis=1)
        y = self.fc(x)
        y = tf.where(y < self.min_sigma, self.min_sigma, y)
        return y[:, :self.units], y[:, self.units:]
