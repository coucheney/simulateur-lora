import numpy as np

from Packet import Packet
from dqn_agent import DQNAgent


# objet corespondant a l'antenne
class BS:
    def __init__(self, idBS: int, coord):
        self.idBS = idBS
        self.coord = coord
        self.packetAtBS = []
        self.combinations = [7, 8, 9, 10, 11, 12]
        self.nbNode = 100
        self.agent = DQNAgent(gamma=0, epsilon=0, lr=0, n_actions=len(self.combinations), input_dims=(2,),
                              mem_size=100, batch_size=32, eps_min=0.01, eps_dec=2e-5, replace=64, algo='tmp',
                              chkpt_dir='save')

        # gamma, epsilon, lr, n_actions, input_dims,mem_size, batch_size, eps_min=0.01, eps_dec=5e-7, replace=1000, algo=None, env_name=None, chkpt_dir='tmp/dqn'
        self.first = [True for i in range(self.nbNode)]
        self.previous_state = [None for i in range(self.nbNode)]
        self.previous_action = [0 for i in range(self.nbNode)]
        self.hasnotchanged = [0 for i in range(self.nbNode)]
        self.ctr = 0

    def recommandation(self, param1, param2, param3, param4, node, reward):
        done = False
        state = np.array([param1, param2])
        if self.first[node]:
            self.previous_state[node] = state
            self.previous_action[node] = self.agent.choose_action(self.previous_state[node])
            self.first[node] = False
            return self.combinations[self.previous_action[node]], 20
        # print("Numéro de node", node)
        if self.hasnotchanged[node] > 9: done = True
        #self.agent.store_transition(self.previous_state[node], self.previous_action[node], reward, state, done)
        #self.agent.learn()
        action = self.agent.choose_action(state)
        #print("Noeud", node, "Précédent état: ", self.previous_state[node], "Précédent action : ",
              #self.combinations[self.previous_action[node]], "Reward :", reward, "Etat actuel", state, "New action", self.combinations[action])
        if action == self.previous_action[node]:
            self.hasnotchanged[node] += 1
        else:
            self.hasnotchanged[node] = 0
        self.previous_action[node] = action
        # print("In the base station action", self.previous_action[node])
        self.previous_state[node] = state
        return self.combinations[self.previous_action[node]], 20

    def __str__(self):
        return str(self.packetAtBS)

    # ajout d'un packet dans la liste de réception de l'antenne
    def addPacket(self, packet: Packet):
        self.packetAtBS.append(packet)

    # suppression d'un packet dans la liste de reception de l'antenne
    def removePacket(self, packet: Packet):
        self.packetAtBS.remove(packet)

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

    def save(self):
        self.agent.save_models()

    def load(self):
        self.agent.load_models()
