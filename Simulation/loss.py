from tensorflow import keras
from tensorflow_probability import distributions
import tensorflow as tf


class MdnLoss(keras.Model):
    def __init__(self, reduce=True):
        super(MdnLoss, self).__init__()
        self.reduce = reduce
        self.min_boundary = 1e-7

    def call(self, mu, sigma, target, dist_type="Normal"):
        if dist_type == "Normal":
            dist = distributions.Normal(mu, sigma)
        elif dist_type == "Beta":
            dist = distributions.Beta(mu, sigma)
            target = tf.clip_by_value(target, self.min_boundary, 1 - self.min_boundary)
        if self.reduce:
            return tf.math.reduce_mean(-dist.log_prob(target))
        else:
            return -dist.log_prob(target)
