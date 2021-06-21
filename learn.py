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

# Un ADR approximatif qui augmente la puissance en priorité, puis la sf si la puissance a atteint son maximum
class ADR(Static):
    def chooseParameter(self, power, SF, lostPacket, validCombination, nbSend):
        if lostPacket:
            if power < 20:
                power += 1
            else:
                SF += 1
        else:
            if SF > 7:
                SF -= 1
            elif power > 2:
                power -= 1
        return SF, power


import numpy as np
import math
import random
from scipy.stats import beta


class UCB1:
    def __init__(self, counts=[], values=[], n_arms=0):
        self.counts = [0 for col in range(n_arms)]
        self.values = [0.0 for col in range(n_arms)]
        self.n_arms = n_arms

    def select_arm(self, lr):
        """ Selectionne le bras avec la valeur de l'estimateur la plus haute"""
        # Parcours des choix qui n'ont pas encore été visités
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        ucb_values = [0.0 for arm in range(self.n_arms)]
        total_counts = sum(self.counts)
        # Calcul des valeurs d'UCB1 pour les différents choix
        for arm in range(self.n_arms):
            bonus = math.sqrt((math.log(total_counts)) / float(self.counts[arm]))
            ucb_values[arm] = self.values[arm] + 0.15 * bonus
        # On choisit celui avec la valeur la plus élevée
        value_max = max(ucb_values)
        return ucb_values.index(value_max)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value


# Version d'Exp3 trouvée sur internet et qui donne des résultats différents de ma version
class Exp3:
    def __init__(self, counts=[], values=[], n_arms=0):

        self.n_arms = n_arms
        self.counts = [0 for col in range(n_arms)]
        self.G = [0 for col in range(n_arms)]
        init_proba = float(1 / float(n_arms))
        self.weights = [1 for col in range(n_arms)]
        self.values = [0 for col in range(n_arms)]
        self.t = 0
        self.old_arm = 0

    def select_arm(self, eta):
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
        self.old_arm = arm_chosen
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
        self.n_arms = n_arms
        self.modified = True
        self.all_draws = [0 for i in range(n_arms)]

    def select_arm(self, useless):
        """Thompson Sampling : sélection de bras"""
        if self.modified:
            # On groupe toutes les paires de a et b et on tire une valeur selon distribution béta
            beta_params = zip(self.a, self.b)
            all_draws = [beta.rvs(i[0], i[1], size=1) for i in beta_params]
            self.all_draws = all_draws
        return np.argmax(self.all_draws)

    def update(self, chosen_arm, reward):
        """Choix du bras à mettre à jour"""
        # epsilon correpsond à la probabilité de mettre à jour a et b (basé sur Adaptative rate Thomspon Sampling de Basu et Ghosh)
        epsilon = 0.5
        omega = random.random()
        if omega <= epsilon:
            # a est basé sur les succès
            self.a[chosen_arm] = self.a[chosen_arm] + reward
            # b est basé sur les échecs
            self.b[chosen_arm] = self.b[chosen_arm] + (1 - reward)
            self.modified = True
        else:
            self.modified = False


class AB:
    def __init__(self, k=0, n_arms=0):
        # bras actuel
        self.index = 0
        # taille du batch
        self.k = k
        self.utilities = [0.0 for col in range(n_arms)]
        self.n_arms = n_arms

    def select_arm(self, lr):
        """ Selectionne le bras avec la valeur de recommandation la plus haute"""
        old_index = self.index
        self.index = self.index + 1
        if old_index < self.k:
            return old_index % self.n_arms
        else:
            return self.utilities.index(max(self.utilities))

    def update(self, chosen_arm, reward):
        self.utilities[chosen_arm] += reward


class MyExp3:
    def __init__(self, gamma=0.1, n_arms=0.0):
        # poids attribués par bras
        self.weights = [1.0 for i in range(0, n_arms)]
        # probabilité d'utiliser
        self.proba = [1.0 for i in range(0, n_arms)]
        # nombre de bras
        self.n_arms = n_arms
        # paramètre à régler
        self.gamma = gamma

    def select_arm(self, lr=0):
        # Mise à jour de la probabilité des bras
        for i in range(0, self.n_arms):
            self.proba[i] = (1 - self.gamma) * (self.weights[i] / np.sum(self.weights)) + self.gamma / self.n_arms
        # Sélection d'un bras selon la distribution
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


# Amélioration supposée de Thompson Sampling, visant à renvoyer soit la meilleure, soit la deuxième meilleure action
class TopTwoThompsonSampling():
    def __init__(self, counts=[], values=[], n_arms=0):
        self.a = [1 for arm in range(n_arms)]
        self.b = [1 for arm in range(n_arms)]
        self.index = 0
        self.n_arms = n_arms
        self.previous_reward = 0
        self.modified = True
        self.all_draws = [0 for i in range(n_arms)]

    #
    def select_arm(self, useless):
        # n_arms = len(self.counts)

        if self.modified:
            # Tirage aléatoire basée sur les paramètres a et b
            self.all_draws = np.random.beta(self.a, self.b,
                                            size=(1, self.n_arms))
            self.all_draws = np.concatenate(self.all_draws, axis=0)

        # On retourne la meilleure ou la deuxième meilleure action selon une loi uniforme
        return np.random.choice(self.all_draws.argsort()[::-1]
                                [:2])

    def update(self, chosen_arm, reward):
        epsilon = 1
        omega = random.random()
        if omega <= epsilon:
            self.a[chosen_arm] = self.a[chosen_arm] + self.estimate(reward, 1)
            self.b[chosen_arm] = self.b[chosen_arm] + self.estimate(1 - reward, 1)
            self.modified = True
        else:
            self.modified = False
        self.previous_reward = reward


class qlearning:

    def __init__(self, n_arms=0, n_reemit=8):
        self.q_matrix = []
        self.n_arms = n_arms
        one_vector = [0 for i in range(n_arms)]
        for i in range(n_reemit):
            self.q_matrix.append(one_vector)
        self.state = 0

    def select_arm(self, state, epsilon=0.5):
        action = np.argmax(self.q_matrix[state])
        if (random.uniform(0, 1) < epsilon):
            action = random.randint(0, self.n_arms - 1)
        return action

    def update(self, reward, state, action, newstate):
        gamma = 0.6
        alpha = 0.1
        self.state = state
        newaction = np.argmax(self.q_matrix[newstate])
        self.q_matrix[state][action] = self.q_matrix[state][action] + alpha * (
                reward + gamma * self.q_matrix[newstate][newaction] - self.q_matrix[state][action])

