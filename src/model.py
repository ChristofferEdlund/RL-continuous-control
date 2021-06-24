import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):

    def __init__(self, state_size, action_size, seed, first_fc=400, second_fc=300):
        super(Actor, self).__init__()

        # Set seed for reproducibility
        self.seed = torch.manual_seed(seed)

        self.fcs = nn.Sequential(*[
            nn.Linear(state_size, first_fc),
            nn.BatchNorm1d(first_fc),
            nn.ReLU(),
            nn.Linear(first_fc, second_fc),
            nn.ReLU(),
            nn.Linear(second_fc, action_size)
        ])
        # Perhaps wants to make a custom init of weights
        self.activation = F.tanh

        self.reset_parameters()

    def forward(self, state):
        """
        Here we want to map state to actions since this is the actor network
        """
        x = self.fcs(state)
        return self.activation(x)


    def reset_parameters(self):
        self.fcs[0].weight.data.uniform_(*hidden_init(self.fcs[0]))
        self.fcs[3].weight.data.uniform_(*hidden_init(self.fcs[3]))
        self.fcs[5].weight.data.uniform_(-3e-3, 3e-3)



class Critic(nn.Module):

    def __init__(self, state_size, action_size, seed, first_fc=400, second_fc=300):
        super(Critic, self).__init__()

        # Set seed for reproducibility
        self.seed = torch.manual_seed(seed)

        self.fc_embedd = nn.Sequential(*[
            nn.Linear(state_size, first_fc),
            nn.BatchNorm1d(first_fc),
            nn.ReLU()
        ])

        self.fc_action_map = nn.Sequential(*[
            nn.Linear(first_fc + action_size, second_fc),
            nn.ReLU(),
            nn.Linear(second_fc, 1) # <- linear output of 1 since we will be predicting a Q-value
        ])

        self.reset_parameters()
        # Perhaps wants to make a custom init of weights ?



    def forward(self, state, action):
        """
        Here we want to map state (state, action) pair to a Q-value
        """
        x = self.fc_embedd(state)
        x = torch.cat([x, action], dim=1)  # <- dim 0 is batch
        return self.fc_action_map(x)

    def reset_parameters(self):
        self.fc_embedd[0].weight.data.uniform_(*hidden_init(self.fc_embedd[0]))
        self.fc_action_map[0].weight.data.uniform_(*hidden_init(self.fc_action_map[0]))
        self.fc_action_map[2].weight.data.uniform_(-3e-3, 3e-3)
