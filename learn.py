import random

"""
La node ne change pas les paramètre de la Node
Sert également de classe de base pour la hiérachie des object qui permettent le choix des paramètres
"""
class Static:
    def chooseParameter(self, power: int, SF: int, lostPacket: bool, validCombination: list, nbSend: int):
        return SF, power


"""Si collision, les paramètre sont tirés aléatoirement dans les paramètres valides"""
class RandChoise(Static):
    def chooseParameter(self, power, SF, lostPacket, validCombination, nbSend):
        if lostPacket:
            return random.choice(validCombination)
        else:
            return SF, power


import numpy as np
import math
import random
from scipy.stats import beta
from scipy import special as sp


class UCB1:
    def __init__(self, counts=[], values=[], n_arms=0):
        self.counts = [0 for col in range(n_arms)]
        self.values = [0.0 for col in range(n_arms)]
        self.n_arms = n_arms

    def select_arm(self, lr):
        """ Selectionne le bras avec la valeur de l'estimateur la plus haute"""
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        ucb_values = [0.0 for arm in range(self.n_arms)]
        total_counts = sum(self.counts)
        for arm in range(self.n_arms):
            bonus = math.sqrt((math.log(total_counts)) / float(self.counts[arm]))
            ucb_values[arm] = self.values[arm] + 0.15 * bonus

        value_max = max(ucb_values)
        return ucb_values.index(value_max)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value


class Exp3:
    def __init__(self, counts=[], values=[], n_arms=0):

        self.n_arms = n_arms
        self.counts = [0 for col in range(n_arms)]
        self.G = [0 for col in range(n_arms)]
        init_proba = float(1 / float(n_arms))
        self.weights = [1 for col in range(n_arms)]
        self.values = [0 for col in range(n_arms)]
        self.t = 0

    def chooseParameter(self, power: int, SF: int, lostPacket: bool, validCombination: list, nbSend: int):
        eta = 0.1
        def tirage_aleatoire_avec_proba(proba_vect):
            valeur_test = random.uniform(0, 1)
            arm_chosen = -1
            i = 0
            sum_partiel = 0
            while i <= len(proba_vect) and arm_chosen == -1:
                sum_partiel += (proba_vect[i])
                if sum_partiel > valeur_test:
                    arm_chosen = i
                i += 1
            return arm_chosen

        self.recalcul_proba = False
        self.proba_vect = [0 for col in range(self.n_arms)]

        #####################################
        # ALGO CALCUL DU Pi
        max_G = max(self.G)
        sum_exp_eta_x_Gjt_max_G = 0
        self.t += 1

        for i in range(len(self.G)):
            sum_exp_eta_x_Gjt_max_G += math.exp(eta * (self.G[i] - max_G))
        for i in range(len(self.proba_vect)):
            self.proba_vect[i] = math.exp(eta * (self.G[i] - max_G)) / float(sum_exp_eta_x_Gjt_max_G)

        ######################################

        arm_chosen = tirage_aleatoire_avec_proba(self.proba_vect)

        return arm_chosen

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        if not self.recalcul_proba:
            if self.counts[chosen_arm] != 1:
                if self.proba_vect[chosen_arm] != 0:
                    if self.proba_vect[chosen_arm] < 0.01:  # Pour eviter les problemes de limite de calcul
                        self.G[chosen_arm] = float(self.G[chosen_arm]) + (float(reward) / 0.01)
                    else:
                        self.G[chosen_arm] = float(self.G[chosen_arm]) + (
                                float(reward) / float(self.proba_vect[chosen_arm]))
            else:
                self.G[chosen_arm] = reward
        else:
            if self.counts[chosen_arm] != 1:
                if self.proba_vect[chosen_arm] != 0:
                    if self.proba_vect[chosen_arm] < 0.01:
                        self.G[chosen_arm] = float(self.G[chosen_arm]) + (float(reward) / 0.01)
                    else:
                        self.G[chosen_arm] = float(self.G[chosen_arm]) + (
                                float(reward) / float(self.proba_vect_2[chosen_arm]))
            else:
                self.G[chosen_arm] = reward


class ThompsonSampling():
    def __init__(self, counts=[], values=[], n_arms=0):
        self.a = [1 for arm in range(n_arms)]
        self.b = [1 for arm in range(n_arms)]
        self.counts = [0 for i in range(n_arms)]
        self.index = 0
        self.n_arms = n_arms
        self.previous_reward = 0
        self.modified = True
        self.all_draws = [0 for i in range(n_arms)]
        self.all_means = []
        self.counter = 10

    # Thompson Sampling selection of arm for each round
    def select_arm(self, useless):
        # n_arms = len(self.counts)

        if self.modified:
            # Pair up all beta params of a and b for each arm
            beta_params = zip(self.a, self.b)
            # Perform random draw for all arms based on their params (a,b)
           # x = random.random()
            """for i in beta_params:
                self.all_draws[i] = beta.rvs(i[0], i[1], size=1) #(self.a[i] + self.b[i]) + np.random.normal(0, 0.001) #self.all_draws[i] = ((x**(self.a[i]-1)) * ((1-x)**(self.b[i]-1))) / sp.beta(self.a[i], self.b[i])  #beta.rvs(self.a[i], self.b[i], size=1) #self.a[i] /
            """
            all_draws = []
                # self.all_means.append(self.all_draws[i] - beta.rvs(self.a[i], self.b[i], size=1000).mean())
            all_draws = [beta.rvs(i[0], i[1], size=1) for i in beta_params]
            self.all_draws = all_draws
            # self.index = all_draws.index(max(all_draws))
            # return index of arm with the highest draw

        return np.argmax(self.all_draws)  # self.all_draws.index((max(self.all_draws)))

        """else:
            return random.randint(0, self.n_arms - 1)
"""

    def estimate(self, x, gamma):
        return x

    def delta(self, x):
        return x - self.previous_reward

    # Choose to update chosen arm and reward
    def update(self, chosen_arm, reward, epsilon=0.5):
        # update counts pulled for chosen arm

        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
        # Update average/mean value/reward for chosen arm
        # value = self.values[chosen_arm]
        # new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        # self.values[chosen_arm] = new_value
        # Update a and b
        # a is based on total counts of rewards of arm
        epsilon = 0.5 #1 / (0.0000005 * n)
        omega = random.random()
        if omega <= epsilon:
            self.a[chosen_arm] = self.a[chosen_arm] + self.estimate(reward, 1)
            # b is based on total counts of failed rewards on arm
            self.b[chosen_arm] = self.b[chosen_arm] + self.estimate(1 - reward, 1)
            self.modified = True
        else:
            self.modified = False
        self.previous_reward = reward


class AB:
    def __init__(self, k=250, n_arms=0):
        self.index = 0
        self.k = k
        self.utilities = [0.0 for col in range(n_arms)]
        self.n_arms = n_arms

    def select_arm(self, lr):
        """ Selectionne le bras avec la valeur de l'estimateur la plus haute"""
        old_index = self.index
        self.index = self.index + 1
        if old_index < self.k:
            return old_index % self.n_arms
        else:
            return self.utilities.index(max(self.utilities))

    def update(self, chosen_arm, reward):
        self.utilities[chosen_arm] += reward


class EXP3BIS:

    def __init__(self, k=250, n_arms=0.0):
        self.Bandit = Bt.Exp3(nbArms=n_arms, gamma=0.2)

    def select_arm(self, lr, rank=1):
        return self.Bandit.choiceWithRank(rank)

    def update(self, chosen_arm, reward):
        self.Bandit.getReward(chosen_arm, reward)


class MyExp3:
    def __init__(self, gamma=0.1, n_arms=0.0):
        self.weights = [1.0 for i in range(0, n_arms)]
        self.proba = [1.0 for i in range(0, n_arms)]
        self.n_arms = n_arms
        self.gamma = gamma

    def select_arm(self, lr=0):
        for i in range(0, self.n_arms):
            self.proba[i] = (1 - self.gamma) * (self.weights[i] / np.sum(self.weights)) + self.gamma / self.n_arms
        tmpSum = 0
        randomVal = random.random()
        for i in range(self.n_arms):
            tmpSum += self.proba[i]
            if randomVal <= tmpSum:
                return i
        return self.proba[self.n_arms - 1]

    def update(self, chosen_arm, reward):
        estimated = reward / self.proba[chosen_arm]
        self.weights[chosen_arm] = self.weights[chosen_arm] * np.exp((self.gamma / self.n_arms) * estimated)



class ADR:

    def __init__(self, counts=[], values=[], n_arms=0):
        self.n_arms = n_arms
        self.action = None

    def select_arm(self, lr):
        return self.action

    def update(self, chosen_arm,reward):
        if not reward :
            self.action += 1




