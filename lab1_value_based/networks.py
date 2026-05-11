"""
Neural Network Architectures for DQN variants.

TDDE78 — Lab 1: Value-Based Deep RL
Linköping University, Spring 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    """
    Standard Deep Q-Network.

    A fully-connected neural network that takes a state as input
    and outputs Q-values for each possible action.

    Architecture: state_dim -> 120 -> 84 -> action_dim

    Args:
        state_dim (int): Dimension of the state/observation space.
        action_dim (int): Number of possible actions.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super(DQNNetwork, self).__init__()
        # =====================================================================
        # TODO: Define the network layers.
        #
        # Create a network with:
        #   - Linear layer: state_dim -> 120
        #   - Linear layer: 120 -> 84
        #   - Linear layer: 84 -> action_dim
        # =====================================================================
        
        self.fc1 = nn.Linear(state_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, action_dim)
        
        # raise NotImplementedError("Define DQNNetwork layers")

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state: Tensor of shape (batch_size, state_dim)

        Returns:
            Tensor of shape (batch_size, action_dim) — Q-values for each action.
        """
        # =====================================================================
        # TODO: Implement the forward pass.
        #
        # Pass the state through the layers with ReLU activations
        # between hidden layers. No activation on the output layer.
        # =====================================================================
        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values
    
        # raise NotImplementedError("Implement DQNNetwork.forward()")


class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN Network (Wang et al., 2016).

    Separates the Q-value into:
      - V(s): state value (how good is this state?)
      - A(s,a): advantage (how much better is this action than average?)

    Q(s,a) = V(s) + A(s,a) - mean(A(s,:))

    The subtraction of mean(A) ensures identifiability — without it,
    V and A are not uniquely determined.

    Architecture:
        state_dim -> 128 (shared)
        128 -> 128 -> 1      (value stream)
        128 -> 128 -> action_dim  (advantage stream)

    Args:
        state_dim (int): Dimension of the state/observation space.
        action_dim (int): Number of possible actions.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super(DuelingDQNNetwork, self).__init__()
        # =====================================================================
        # TODO: Define the network layers.
        #
        # Create:
        #   1. A shared feature layer: state_dim -> 128
        #   2. Value stream: 128 -> 128 -> 1
        #   3. Advantage stream: 128 -> 128 -> action_dim
        # =====================================================================
        
        self.shared = nn.Linear(state_dim, 128)
        
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # raise NotImplementedError("Define DuelingDQNNetwork layers")

    def forward(self, state):
        """
        Forward pass through the dueling network.

        Args:
            state: Tensor of shape (batch_size, state_dim)

        Returns:
            Tensor of shape (batch_size, action_dim) — Q-values for each action.
        """
        # =====================================================================
        # TODO: Implement the dueling forward pass.
        #
        # Steps:
        # 1. Pass state through shared feature layer (+ ReLU)
        # 2. Pass features through value stream -> V(s)  (shape: batch_size, 1)
        # 3. Pass features through advantage stream -> A(s,a) (shape: batch_size, action_dim)
        # 4. Combine: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        #
        # The mean subtraction ensures that the advantage has zero mean,
        # making V(s) represent the true state value.
        # =====================================================================
        features = F.relu(self.shared(state))
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
        
        # raise NotImplementedError("Implement DuelingDQNNetwork.forward()")
