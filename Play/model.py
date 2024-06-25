from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


class DiceMaster(keras.Model):

    def __init__(self):
        super(DiceMaster, self).__init__()

        self.fc = keras.Sequential(
            [
                layers.Dense(102, activation="relu", name="layer1"),
                layers.Dense(102, activation="relu", name="layer2"),
                layers.Dense(3, activation="sigmoid", name="layer3"),
            ]
        )

    def call(self, x, training=False):
        y = self.fc(x)
        max_values = tf.reduce_max(y, axis=1)
        mask = tf.equal(y, max_values[:, tf.newaxis])
        return tf.where(mask, y, tf.zeros_like(y))  # (batch_size, [small, leopard, Large])
