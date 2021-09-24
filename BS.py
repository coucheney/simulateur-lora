import numpy as np

from Packet import Packet
from dqn_agent import DQNAgent


# objet corespondant a l'antenne
class BS:
    def __init__(self, idBS: int, coord):
        self.idBS = idBS
        self.coord = coord
        self.packetAtBS = []
        self.combinations = self.checkCombination()
        self.nbNode = 100
        self.agent = DQNAgent(gamma=0, epsilon=1, lr=0.00008, n_actions=len(self.combinations), input_dims=(7,),
                              mem_size=800, batch_size=16, eps_min=0, eps_dec=7e-6, replace=16, algo='tmp',
                              chkpt_dir='save')
        self.first = [True for i in range(self.nbNode)]
        self.previous_state = [None for i in range(self.nbNode)]
        self.previous_action = [0 for i in range(self.nbNode)]
        self.hasnotchanged = [0 for i in range(self.nbNode)]
        self.ctr = 0


    def __str__(self):
        return str(self.packetAtBS)

    # ajout d'un packet dans la liste de réception de l'antenne
    def addPacket(self, packet: Packet):
        self.packetAtBS.append(packet)

    # suppression d'un packet dans la liste de reception de l'antenne
    def removePacket(self, packet: Packet):
        self.packetAtBS.remove(packet)


######################################Partie DQN################################################


    """
           Fonction de recommandation de paramètres de la base, utilisée pour un DQN.
           Est chargée de récupérer les informations du message (RSSI , sensibilité, SF)
           Entraine le réseau de neurones et renvoit un jeu de paramètres 
    """
    def recommandation(self, param1, param2, param3, node, reward):
        done = False
        param1sin = np.sin(param1)
        param2sin = np.sin(param2)
        state = np.array([param1sin, param2sin, param1, param2, param3, np.cos(param1), np.cos(param2)])
        if self.first[node]:
            self.previous_state[node] = state
            self.previous_action[node] = self.agent.choose_action(self.previous_state[node])
            self.first[node] = False
            return self.combinations[self.previous_action[node]][0], self.combinations[self.previous_action[node]][1]
        if self.hasnotchanged[node] > 9: done = True
        self.agent.store_transition(self.previous_state[node], self.previous_action[node], reward, state, done)
        self.agent.learn()
        action = self.agent.choose_action(state)
        if action == self.previous_action[node]:
            self.hasnotchanged[node] += 1
        else:
            self.hasnotchanged[node] = 0
        self.previous_action[node] = action
        self.previous_state[node] = state
        return self.combinations[self.previous_action[node]][0], self.combinations[self.previous_action[node]][1]

    """
        Deuxième fonction de recommandation de la base. Fonctionne comme la première mais n'entraine pas le réseau. 
    """
    def recommandation2(self, param1, param2, param3, node, change):
        if self.first[node] or change:
            param1sin = np.sin(param1)
            param2sin = np.sin(param2)
            state = np.array([param1sin, param2sin, param1, param2, param3, np.cos(param1), np.cos(param2)])
            action = self.agent.choose_action(state)
            self.first[node] = False
            return self.combinations[action][0], self.combinations[action][1]

    """Voir checkCombination2 dans le fichier Node.py"""
    def checkCombination(self, sensi=[]):
        sf = [7, 8, 9, 10, 11, 12]
        pw = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        combi = []
        for i in sf:
            for j in pw:
                combi.append([i, j])
        deleteList = [2, 3, 5, 6, 7]
        toDelete = []
        for elem in combi:
            if elem[1] in deleteList:
                toDelete.append(elem)
        for elem in toDelete:
            combi.remove(elem)
        return combi

    """Sauvegarde le réseau"""
    def save(self):
        self.agent.save_models()

    """Charge le réseau"""
    def load(self):
        self.agent.load_models()