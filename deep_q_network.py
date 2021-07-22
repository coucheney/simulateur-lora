import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.fc1 = nn.Linear(*input_dims, 64)
        self.activation1 = nn.ELU(64, 64)
        self.fc2 = nn.Linear(64, 32)
        self.activation2 = nn.ELU(32, 32)
        self.fc3 = nn.Linear(32, 24)
        self.activation3 = nn.ELU(24, 24)
        self.fc4= nn.Linear(24, 12)
        self.activation4 = nn.ELU(12, 12)
        self.fc5 = nn.Linear(12, 6)
        self.activation5 = nn.ELU(6, 6)
        self.fc6 = nn.Linear(6, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        x = self.activation1(self.fc1(state))
        x1 = self.activation2(self.fc2(x))
        x2 = self.activation3(self.fc3(x1))
        x3 = self.activation4(self.fc4(x2))
        x4 = self.activation5(self.fc5(x3))
        actions = self.fc6(x4)
        return actions

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), 'network')

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load('network'))