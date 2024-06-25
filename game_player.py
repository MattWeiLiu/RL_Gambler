#!/usr/bin/env python3
from Play.ENV import LuxuryDiceGame
from Play.model import DiceMaster as Net
from collections import OrderedDict
from scipy import stats
from lib.configuration import Config
from lib.utils import *

import os
import numpy as np
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")

config = Config(os.path.join(os.getcwd(), "Play/config.yaml"))

# Set GPU as available physical device
if gpus := tf.config.experimental.list_physical_devices(device_type='GPU'):
    tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')


def calc_loss(batch):
    states, actions, rewards, next_states = batch
    states_v = tf.constant(states)
    next_states_v = tf.constant(next_states)
    # actions_v = tf.transpose(tf.constant(actions))
    rewards_v = tf.transpose(tf.constant(rewards))

    with tf.GradientTape() as tape:
        state_action_values = net(states_v)
        next_state_values = tgt_net(next_states_v)
        expected_state_action_values = next_state_values * config.RL_PARAMETER.GAMMA + rewards_v[:, tf.newaxis]
        loss_t = tf.keras.losses.MSE(state_action_values, expected_state_action_values)
    gradients = tape.gradient(loss_t, net.trainable_variables)
    optimizer.apply_gradients(zip(gradients, net.trainable_variables))
    return loss_t.numpy()


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self.state = None
        self._reset()

    def _reset(self):
        self.state = self.env._reset()

    def play_step(self, net, epsilon=0.0):
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array(list(self.env.states["Observation"]) + list(self.env.states["TimeCode"][0]), copy=False)
            state_v = tf.constant(state_a.reshape(1, -1))
            act_v = net(state_v)
            action = {"Bet": tf.reduce_sum(act_v) * 1e5, "Category": int(tf.argmax(act_v, axis=1))}

        # do step in the environment
        new_state, reward, call, prob = self.env.step(action)
        exp = Experience(list(self.state["Observation"]) + list(self.state["TimeCode"].numpy()[0]),
                         [float(action["Bet"]), int(action["Category"])],
                         reward,
                         list(new_state["Observation"]) + list(new_state["TimeCode"].numpy()[0]))
        self.exp_buffer.append(exp)
        self.state = new_state
        return net, reward, action, call, prob


if __name__ == "__main__":
    env = LuxuryDiceGame(config)
    net = Net()
    tgt_net = Net()

    buffer = ExperienceBuffer(config.RL_PARAMETER.REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = config.RL_PARAMETER.EPSILON_START

    optimizer = tf.keras.optimizers.Adam(learning_rate=float(config.optimizer.learning_rate))
    round_idx = 0
    best_mean_reward = None

    loss_cache = deque([], maxlen=10000)
    reward_cache = deque([], maxlen=10000)
    while True:
        round_idx += 1
        epsilon = max(config.RL_PARAMETER.EPSILON_FINAL, config.RL_PARAMETER.EPSILON_START - round_idx
                      / float(config.RL_PARAMETER.EPSILON_DECAY_LAST_EPOCH))

        net, reward, action, call, prob = agent.play_step(net, epsilon)
        reward_cache.append(reward)
        
        if len(buffer) < config.RL_PARAMETER.REPLAY_START_SIZE:
            continue

        batch = buffer.sample(config.RL_PARAMETER.BATCH_SIZE)
        loss_cache.append(np.mean(calc_loss(batch)))
        LastAvgRwd = stats.trim_mean(reward_cache, 0.1)
        LastAvgLoss = np.mean(loss_cache)
        if round_idx % 500 == 0:
            if isinstance(action, OrderedDict):
                bet = action["Bet"][0]
            else:
                bet = int(action["Bet"])
            select = selection_map[action["Category"]]
            print(f"Round: {round_idx}, LastAvgLoss: {LastAvgLoss:.3f}, LastAvgRwd: {LastAvgRwd:.3f}, CurrRwd: {reward}, eps: {epsilon:.2f}, Prob: {prob}, Call: {call}, Bet: {bet}, Select: {select}")

        if best_mean_reward is None or best_mean_reward < LastAvgRwd:
            net.save("weights/best_LuxuaryDiceModel.tf", save_format='tf')
            if best_mean_reward is not None:
                print(f"Best mean reward updated {best_mean_reward:.3f} -> {LastAvgRwd:.3f}, model saved")
            best_mean_reward = LastAvgRwd

        if round_idx % config.RL_PARAMETER.SYNC_TARGET_FRAMES == 0:
            # tgt_net = tf.keras.models.clone_model(net)
            tgt_net.set_weights(net.get_weights())
