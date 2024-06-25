from collections import deque
import random
import numpy as np

# config = Config(os.path.join(os.getcwd(), "Play/luxury_dice.yaml"))


class LuxuryDice:

    def __init__(self, config):
        self.overflow = config.overflow
        self.probabilityTable = config.probabilityTable
        self.odds = config.odds
        self.tax = config.tax
        self.safe_line = config.safeline
        self.loaded_gate = config.loadedgate
        self.index_round = config.index_round

        self.balance = random.randint(config.safeline, config.overflow.size)
        self.round = 0
        self.previous_k_jackpot = 0
        self.queue = deque([self.generate_random_bit() for _ in range(self.index_round)])

    def draw(self, small, leopard, large):
        self.round += 1
        bet = large + leopard + small
        tax = bet * self.tax.fair
        self.balance += bet
        prob = self.probabilityTable[min(self.previous_k_jackpot, len(self.probabilityTable) - 1)].copy()
        emergency_flag = False
        loaded_flag = False
        #
        # Check if it is affordable and set the probability of unaffordable call to zero
        # -------------------------------------------------------------------------------
        if large * self.odds.large > self.balance - tax or \
           leopard * self.odds.leopard > self.balance - tax or \
           small * self.odds.small > self.balance - tax:
            if small * self.odds.small < self.balance - tax:
                prob["large"] = 0
                prob["leopard"] = 0
                prob["small"] = 10000
            elif large * self.odds.large < self.balance - tax:
                prob["large"] = 10000
                prob["leopard"] = 0
                prob["small"] = 0
            else:  # leopard * self.odds.leopard < self.balance
                prob["large"] = 0
                prob["leopard"] = 10000
                prob["small"] = 0
            emergency_flag = True

        if not emergency_flag and self.balance < self.safe_line and bet > self.loaded_gate:
            loaded_flag = True
            prob = self.loaded_probability_adjust(prob, bet, large, leopard, small)

        calls = list(prob.keys())
        probability = list(prob.values()) / np.sum(list(prob.values()))
        call = np.random.choice(calls, p=probability)
        if call == "large":
            self.balance -= large * self.odds.large
            self.queue.append(0)
        elif call == "leopard":
            self.balance -= leopard * self.odds.leopard
            self.queue.append(1)
            self.previous_k_jackpot += 1
        else:  # call == "small"
            self.balance -= small * self.odds.small
            self.queue.append(0)

        if self.queue.popleft() == 1:
            self.previous_k_jackpot -= 1

        #
        # Levying taxes
        # -------------------------------------------------------------------------------
        if emergency_flag:
            self.balance += (1 - self.tax.emergency) * bet
        elif loaded_flag:
            self.balance += (1 - self.tax.loaded) * bet
        else:  # Fair status
            self.balance += (1 - self.tax.fair) * bet

        #
        # Check overflow
        # -------------------------------------------------------------------------------
        if self.round == self.overflow.sequence:
            self.balance = min(self.balance, self.overflow.size)
            self.round = 0

        return call, prob

    def loaded_probability_adjust(self, basic_prob, bet, large, leopard, small):
        bias = min(2 * (self.safe_line - self.balance) / self.safe_line, 1)
        coef = self.round_down(bias * basic_prob["leopard"] / 6667, 3)
        bias_ratio = [0, 0, 0]
        bias_ratio[0] = int(coef * (3333 - (10000 * small / bet)))
        bias_ratio[1] = int(coef * (3333 - (10000 * leopard / bet)))
        bias_ratio[2] = 0 - bias_ratio[0] - bias_ratio[1]
        prob_table = {"large": 0, "leopard": 0, "small": 0}
        prob_table["small"] = basic_prob["small"] + bias_ratio[0]
        prob_table["leopard"] = basic_prob["leopard"] + bias_ratio[1]
        prob_table["large"] = 10000 - prob_table["small"] - prob_table["leopard"]
        return prob_table

    @staticmethod
    def round_down(num, decimals):
        factor = 10 ** decimals
        return int(num * factor) / factor

    def reset(self):
        self.balance = random.randint(self.safe_line, self.overflow.size)
        self.round = 0
        self.previous_k_jackpot = 0
        self.queue = deque([self.generate_random_bit() for _ in range(self.index_round)])

    def generate_random_bit(self):
        random_number = random.random()
        if random_number < 0.051:
            self.previous_k_jackpot += 1
            return 1
        else:
            return 0
