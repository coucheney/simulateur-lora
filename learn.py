import collections
import random

"""
La node ne change pas les paramètre de la Node
Sert également de classe de base pour la hiérachie des object qui permettent le choix des paramètres
"""
class Static:
    def __init__(self):
        self.nStrat = 0

    def start(self, validCombination):
        self.nStrat = len(validCombination)

    def chooseParameter(self, power: int, SF: int, lostPacket: bool, validCombination: list, nbSend: int, energyCost: float):
        return SF, power


# Si collision, les paramètre sont tirés aléatoirement dans les paramètres valides
class RandChoise(Static):
    def chooseParameter(self, power: int, SF: int, lostPacket: bool, validCombination: list, nbSend: int, energyCost: float):
        if lostPacket:
            return random.choice(validCombination)
        else:
            return SF, power

# Un ADR approximatif qui augmente la puissance en priorité, puis la sf si la puissance a atteint son maximum
class ADR(Static):

    def __init__(self):
        super().__init__()
        self.power = 14
        self.sf = 12
        self.twenty = False
        self.snr = collections.deque(maxlen=20)
        self.margin = 10
        self.required = [-7.5, -10, -12.5, -15, -17.5, -20]
        self.noise = 2  # bruit définit au hasard
        self.counter = 0

    # Calcul du Signal To Noise Ratio
    def computeSNR(self, power,SF):
        val = 20 * np.log(power+2*SF / self.noise)
        self.snr.append(val)

    def computemargin(self, SF):
        return max(self.snr) - self.required[SF % 7] - self.margin

    def chooseParameter(self, power=0, SF=0, lostPacket=False, validCombination=None, nbSend=0, state=0, energyCost=0):
        if not lostPacket:
            self.computeSNR(power, SF)
            Nstep = round(self.computemargin(SF) / 3)
            if Nstep < 0:
                if power < 14:
                    power += 3
            elif SF > 7:
                SF -= 1
            elif power > 0:
                power -= 1

        self.counter += 1
        if self.counter == 64:
            self.counter = 0
            if lostPacket and SF < 12:
                SF += 1
        return SF, power


import numpy as np
import math
import random
from scipy.stats import beta


class UCB1(Static):
    def __init__(self):
        super().__init__()
        self.counts = []
        self.values = []

    def start(self, validCombination):
        self.nStrat = len(validCombination)
        self.counts = [0 for col in range(self.nStrat)]
        self.values = [0.0 for col in range(self.nStrat)]


    def chooseParameter(self, power: int, SF: int, lostPacket: bool, validCombination: list, nbSend: int, energyCost: float):
        if lostPacket:
            reward = 0
        else:
            reward = -energyCost # 1-(self.packet.energyCost/0.5)
        try:
            self.update(validCombination.index([SF, power]), reward)
        except ValueError:
            self.update(0, reward)
        arm = self.select_arm(0.1)
        return validCombination[arm][0], validCombination[arm][1]

    def select_arm(self, lr):
        """ Selectionne le bras avec la valeur de l'estimateur la plus haute"""
        # Parcours des choix qui n'ont pas encore été visités
        for arm in range(self.nStrat):
            if self.counts[arm] == 0:
                return arm
        ucb_values = [0.0 for arm in range(self.nStrat)]
        total_counts = sum(self.counts)
        # Calcul des valeurs d'UCB1 pour les différents choix
        for arm in range(self.nStrat):
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
class Exp3(Static):
    def __init__(self):
        super().__init__()
        self.nStrat = None
        self.counts = None
        self.G = None
        self.weights = None
        self.values = None
        self.t = 0
        self.old_arm = 0
        self.recalcul_proba = False

    def start(self, validCombination):
        self.nStrat = len(validCombination)
        self.counts = [0 for col in range(self.nStrat)]
        self.G = [0 for col in range(self.nStrat)]
        self.weights = [1 for col in range(self.nStrat)]
        self.values = [0 for col in range(self.nStrat)]

    def chooseParameter(self, power: int, SF: int, lostPacket: bool, validCombination: list, nbSend: int, energyCost: float):
        if lostPacket:
            reward = 0
        else:
            reward = -energyCost # 1-(self.packet.energyCost/0.5)
        try:
            self.update(validCombination.index([SF, power]), reward)
        except ValueError:
            self.update(0, reward)
        arm = self.select_arm(0.1)
        return validCombination[arm][0], validCombination[arm][1]

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
        self.proba_vect = [0 for col in range(self.nStrat)]

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


class ThompsonSampling(Static):
    def __init__(self):
        super().__init__()
        self.a = None
        self.b = None
        self.modified = True
        self.all_draws = None
        self.old_arm = 0

    def start(self, validCombination):
        self.nStrat = len(validCombination)
        self.a = [1 for arm in range(self.nStrat)]
        self.b = [1 for arm in range(self.nStrat)]
        self.all_draws = [0 for i in range(self.nStrat)]

    def chooseParameter(self, power: int, SF: int, lostPacket: bool, validCombination: list, nbSend: int, energyCost: float):
        if lostPacket:
            reward = 0
        else:
            reward = -energyCost # 1-(self.packet.energyCost/0.5)
        try:
            self.update(validCombination.index([SF, power]), reward)
        except ValueError:
            self.update(0, reward)
        arm = self.select_arm(0.1)
        return validCombination[arm][0], validCombination[arm][1]

    def select_arm(self, useless):
        """Thompson Sampling : sélection de bras"""
        if self.modified:
            # On groupe toutes les paires de a et b et on tire une valeur selon distribution béta
            beta_params = zip(self.a, self.b)
            all_draws = [beta.rvs(i[0], i[1], size=1) for i in beta_params]
            self.all_draws = all_draws
            self.old_arm = np.argmax(self.all_draws)
        return self.old_arm

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



class AB(Static):
    def __init__(self, k=0):
        super().__init__()
        self.index = 0
        self.k = k
        self.utilities = None
        self.nStrat = None

    def start(self, validCombination):
        self.nStrat = len(validCombination)
        self.utilities = [0.0 for col in range(self.nStrat)]

    def chooseParameter(self, power: int, SF: int, lostPacket: bool, validCombination: list, nbSend: int, energyCost: float):
        if lostPacket:
            reward = 0
        else:
            reward = -energyCost # 1-(self.packet.energyCost/0.5)
        try:
            self.update(validCombination.index([SF, power]), reward)
        except ValueError:
            self.update(0, reward)
        arm = self.select_arm(0.1)
        return validCombination[arm][0], validCombination[arm][1]

    def select_arm(self, lr):
        """ Selectionne le bras avec la valeur de recommandation la plus haute"""
        old_index = self.index
        self.index = self.index + 1
        if old_index < self.k:
            return old_index % self.nStrat
        else:
            return self.utilities.index(max(self.utilities))

    def update(self, chosen_arm, reward):
        self.utilities[chosen_arm] += reward


class MyExp3(Static):
    def __init__(self, gamma=0.1):
        super().__init__()
        self.nStrat = None
        self.weights = None
        self.proba = None
        self.gamma = None
        self.old_arm = 0
        self.gamma = gamma

    def start(self, validCombination):
        self.nStrat = len(validCombination)
        self.weights = [1 for i in range(0, self.nStrat)]
        self.proba = [1 for i in range(0, self.nStrat)]

    def chooseParameter(self, power: int, SF: int, lostPacket: bool, validCombination: list, nbSend: int, energyCost: float):
        if lostPacket:
            reward = 0
        else:
            reward = -energyCost # 1-(self.packet.energyCost/0.5)
        try:
            self.update(validCombination.index([SF, power]), reward)
        except ValueError:
            self.update(0, reward)
        arm = self.select_arm()
        return validCombination[arm][0], validCombination[arm][1]

    def select_arm(self):
        # Mise à jour de la probabilité des bras
        for i in range(0, self.nStrat):
            self.proba[i] = (1 - self.gamma) * (self.weights[i] / np.sum(self.weights)) + self.gamma / self.nStrat
        # Sélection d'un bras selon la distribution
        tmpSum = 0
        randomVal = random.random()
        for i in range(self.nStrat):
            tmpSum += self.proba[i]
            if randomVal <= tmpSum:
                self.old_arm = i
                return i
        self.old_arm = self.proba[self.nStrat - 1]
        return self.proba[self.nStrat - 1]

    def update(self, chosen_arm, reward):
        estimated = reward / self.proba[chosen_arm]
        self.weights[chosen_arm] = self.weights[chosen_arm] * np.exp((self.gamma / self.nStrat) * estimated)


# Amélioration supposée de Thompson Sampling, visant à renvoyer soit la meilleure, soit la deuxième meilleure action
class TopTwoThompsonSampling(Static):
    def __init__(self):
        super().__init__()
        self.a = None
        self.b = None
        self.index = 0
        self.previous_reward = 0
        self.modified = True
        self.all_draws = None

    def start(self, validCombination):
        self.nStrat = len(validCombination)
        self.a = [1 for arm in range(self.nStrat)]
        self.b = [1 for arm in range(self.nStrat)]
        self.all_draws = [0 for i in range(self.nStrat)]

    def chooseParameter(self, power: int, SF: int, lostPacket: bool, validCombination: list, nbSend: int, energyCost: float):
        if lostPacket:
            reward = 0
        else:
            reward = -energyCost # 1-(self.packet.energyCost/0.5)
        try:
            self.update(validCombination.index([SF, power]), reward)
        except ValueError:
            self.update(0, reward)
        arm = self.select_arm()
        return validCombination[arm][0], validCombination[arm][1]

    def select_arm(self):
        # nStrat = len(self.counts)

        if self.modified:
            # Tirage aléatoire basée sur les paramètres a et b
            self.all_draws = np.random.beta(self.a, self.b,
                                            size=(1, self.nStrat))
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
    def __init__(self, n_reemit=8):
        self.q_matrix = []
        self.nStrat = None
        self.n_reemit = n_reemit
        self.state = 0

    def start(self, validCombination):
        self.nStrat = len(validCombination)
        one_vector = [0 for i in range(self.nStrat)]
        for i in range(self.n_reemit):
            self.q_matrix.append(one_vector)

    def chooseParameter(self, power: int, SF: int, lostPacket: bool, validCombination: list, nbSend: int, energyCost: float):
        if lostPacket:
            reward = 0
        else:
            reward = -energyCost # 1-(self.packet.energyCost/0.5)
        try:
            self.update(validCombination.index([SF, power]), reward)
        except ValueError:
            self.update(0, reward)
        arm = self.select_arm()
        return validCombination[arm][0], validCombination[arm][1]

    def select_arm(self, state, epsilon=0.5):
        action = np.argmax(self.q_matrix[state])
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, self.nStrat - 1)
        return action

    def update(self, reward, state, action, newstate):
        gamma = 0.6
        alpha = 0.1
        self.state = state
        newaction = np.argmax(self.q_matrix[newstate])
        self.q_matrix[state][action] = self.q_matrix[state][action] + alpha * (
                reward + gamma * self.q_matrix[newstate][newaction] - self.q_matrix[state][action])

