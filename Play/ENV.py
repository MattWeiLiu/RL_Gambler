from io import BytesIO
from tensorflow.python.lib.io import file_io
from collections import deque
from gym.spaces import Dict, Discrete, Box
from Play.GAME import LuxuryDice as Game
from lib.utils import updateTimeVec
from collections import OrderedDict
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import gym


class LuxuryDiceGame(gym.Env):

    def __init__(self, config, observation_size=100):
        self.config = config
        self.action_space = Dict({"Bet": Box(low=0.0, high=10000),
                                  "Category": Discrete(3)
                                  })
        self.balance = float(config.RL_PARAMETER.INIT_BALANCE)
        self.observation_size = observation_size
        self.selection_map = {0: "small", 1: "leopard", 2: "large"}
        self.odds = config.odds

        self.game = Game(config)
        self.starting_data_path = config.starting_data_path
        self._reset()
        self.simulator = tf.keras.models.load_model(config.simulation_model_path)

    def _reset(self):
        self.game.reset()
        self.sample_starting_data()
        self.states = {"Observation": deque(self.game.queue, maxlen=self.observation_size),
                       "TimeCode": self.time_encode}
        return self.states

    def sample_starting_data(self):
        f = BytesIO(file_io.read_file_to_string(self.starting_data_path, binary_mode=True))
        data = np.load(f)
        x, time_encode = data["record"], data["time_code"]
        k = np.random.randint(low=0.8 * x.shape[0], high=x.shape[0])
        self.x = tf.cast(x[k:k + 1, :, :], tf.float32)
        self.time_encode = tf.cast(time_encode[k:k + 1, :], tf.float32)

    def step(self, action):
        call, reward, prob = self.roll_the_dice(action)
        self.balance += reward
        if call == "large":
            self.states["Observation"].extend([1, 0, 0])
        elif call == "leopard":
            self.states["Observation"].extend([0, 1, 0])
        else:  # call == "small"
            self.states["Observation"].extend([0, 0, 1])
        return self.states, reward, call, prob

    def roll_the_dice(self, action):
        if isinstance(action, OrderedDict):
            bet = action["Bet"][0]
        else:
            bet = int(action["Bet"])
        selection = action["Category"]
        odd = self.odds[self.selection_map[selection]] - 1
        alpha, beta = self.simulator(self.x, self.time_encode)
        log_numBet = tfp.distributions.Normal(loc=alpha[:, :1], scale=beta[:, :1]).sample()
        numBet = int(100 ** log_numBet.numpy()[0][0])
        bet_sample = tfp.distributions.Beta(alpha[:, 1:], beta[:, 1:]).sample() * numBet
        bet_sample = bet_sample.numpy()
        respective_bet = [0, 0, 0]  # small, leopard, large
        for i in range(3):
            for j in range(4):
                respective_bet[i] += int(bet_sample[0, i * 4 + j] * (10 ** (j + 2)))
        if selection == 0:  # small
            s = 7
        elif selection == 1:  # leopard
            s = 3
        else:  # selection == "large"
            s = 11
        bet = int(bet)
        dummy_bet = int(bet)
        for k, divisor in enumerate([1e5, 1e4, 1e3, 1e2]):
            q, r = divmod(dummy_bet, divisor)
            numBet += q
            bet_sample[0, s - k] += q
            dummy_bet = r

        small, leopard, large = respective_bet
        call, prob = self.game.draw(small, leopard, large)
        if call == "large":
            tmp = [1, 0, 0]
        elif call == "leopard":
            tmp = [0, 1, 0]
        else:
            tmp = [0, 0, 1]
        sin, cos = self.time_encode.numpy()[0]
        self.time_encode = tf.cast([updateTimeVec(sin, cos)], tf.float32)
        numBet = max(numBet, 1)
        last_status = [numBet] + list(bet_sample[0] / numBet) + tmp
        self.x = tf.concat([self.x[:, 1:, :], tf.cast([[last_status]], tf.float32)], axis=1)
        reward = bet * odd if call == self.selection_map[selection] else -bet
        return call, reward, prob
