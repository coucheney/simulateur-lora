import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, 96)
        self.activation1 = nn.SELU()
        self.fc2 = nn.Linear(96, 96)
        self.activation2 = nn.Tanh()
        self.fc3 = nn.Linear(96, 96)
        self.activation3 = nn.Tanh()
        self.fc4 = nn.Linear(96, 96)
        self.activation4 = nn.Tanh()
        self.fc5 = nn.Linear(96, 96)
        self.activation5 = nn.Tanh()
        self.fc6 = nn.Linear(96, 96)
        self.activation6 = nn.SiLU()
        self.fc7 = nn.Linear(96, 96)
        self.activation7 = nn.SiLU()
        self.fc8 = nn.Linear(96, 96)
        self.activation8 = nn.SiLU()
        self.fc9 = nn.Linear(96, 96)
        self.activation9 = nn.SiLU()
        self.fc10 = nn.Linear(96, 96)
        self.activation10 = nn.SiLU()
        self.fc11 = nn.Linear(96, 96)
        self.pi = nn.Linear(96, n_actions)
        self.v = nn.Linear(fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x8 = self.activation1(self.fc1(state))
        #x9 = self.activation2(self.fc10(x8))
        #x10 = self.activation3(self.fc11(x9))
        pi = self.pi(x8)
        v = self.v(x8)
        return (pi, v)

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), 'network')

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load('network'))

class Agent():
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions,
                 gamma=0.99):
        self.gamma = gamma
        self.lr = lr
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.actor_critic = ActorCriticNetwork(lr, input_dims, n_actions,
                                               fc1_dims, fc2_dims)
        self.log_prob = None

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor_critic.device)
        probabilities, _ = self.actor_critic.forward(state)
        probabilities = F.softmax(probabilities, dim=1)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        self.log_prob = log_prob

        return action.item()

    def learn(self, state, reward, state_, done):
        self.actor_critic.optimizer.zero_grad()
        state = T.tensor([state], dtype=T.float).to(self.actor_critic.device)
        reward = T.tensor(reward, dtype=T.float).to(self.actor_critic.device)

        _, critic_value = self.actor_critic.forward(state)

        delta = reward - critic_value

        actor_loss = -self.log_prob*delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()
        self.actor_critic.optimizer.step()

    def save_models(self):
        self.actor_critic.save_checkpoint()

    def load_models(self):
        self.actor_critic.load_checkpoint()