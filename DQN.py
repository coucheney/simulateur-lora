import os

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims,fc3_dims,fc4_dims,fc5_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.fc5_dims = fc5_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.fc4 = nn.Linear(self.fc3_dims, self.n_actions)
        #self.fc4 = nn.Linear(self.fc4_dims, self.fc5_dims)
        #self.fc5 = nn.Linear(self.fc5_dims, self.fc5_dims)
        #self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)#optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,state):
        x = self.fc1(state)
        x = T.relu(self.fc2(x))
        print(x)
        actions = self.fc3(x)
        """x = (self.fc2(x))
        actions = F.relu(self.fc3(x))"""
        return actions


class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100, eps_end=0, eps_dec=1e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [ i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        if os.path.isfile("network") :
            print("Load Network")
            self.Q_eval = T.load("network")
            self.Q_eval.eval()

        else:
            self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims,
                         fc1_dims=16, fc2_dims=16, fc3_dims=6, fc4_dims=96, fc5_dims=96)
        #print("Début", self.Q_eval.state_dict())
        """DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims,
                                   fc1_dims=256, fc2_dims=256)"""

        """ T.load("network")
        self.Q_eval.eval()"""
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        #self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        #self.terminal_memory[index] = done

        self.mem_cntr +=1

    def choose_action(self,observation):
        if np.random.random() > self.epsilon:
            #print("Observation", observation)
            state = T.tensor([observation]).to(self.Q_eval.device).float()
            #print("State", state)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
            #print(actions)
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        self.Q_eval.optimizer.zero_grad()
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        #new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        #terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index,action_batch]
        #q_next = self.Q_eval.forward(new_state_batch)
        #q_next[terminal_batch] = 0.0

        q_target = reward_batch #+ self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_eval, q_target).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
                        else self.eps_min

    def saveNetwork(self):
        T.save(self.Q_eval,"network")

    def loadNetwork(self):
        model = T.load("network")
        model.eval()
