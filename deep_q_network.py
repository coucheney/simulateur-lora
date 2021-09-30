import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

"""Dueling Deep-Q-Network"""
class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.fc1 = nn.Linear(*input_dims, 96)
        self.activation1 = nn.Tanh()
        self.fc2 = nn.Linear(96, 96)
        self.activation2 = nn.SiLU()
        self.fc3 = nn.Linear(96, 96)
        self.V = nn.Linear(96, 1)
        self.A = nn.Linear(96, n_actions)
        self.optimizer = optim.AdamW(self.parameters(),lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.fc3 = nn.Linear(96, 96)
        self.fc4 = nn.Linear(96, 96)
        self.fc5 = nn.Linear(96, 96)
        self.fc6 = nn.Linear(96, 96)
        self.fc7 = nn.Linear(96, 96)
        self.fc8 = nn.Linear(96, 96)
        self.fc9 = nn.Linear(96, 96)
        self.fc10 = nn.Linear(96, 96)
        self.fc11 = nn.Linear(96, 96)

    """Prédit des valeurs de reward"""
    def forward(self, state):
        x = self.activation1(self.fc1(state))
        x = (self.fc2(x))
        x = self.activation2(self.fc3(x))
        V = self.V(x)
        A = self.A(x)
        return (V, A)

    """Sauvegarde le réseau de neurones dans le fichier network"""
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), 'network')

    """Charge le réseau de neurones à partir du fichier network"""
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load('network'))