import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from reinforcement_functions import SquashedNormal
import math
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device

        # Add layer normalization for input stabilization
        self.input_norm = nn.LayerNorm(in_channels)
        
        self.Wq = nn.Linear(self.in_channels, self.out_channels)
        self.Wk = nn.Linear(self.in_channels, self.out_channels)
        self.Wv = nn.Linear(self.in_channels, self.out_channels)

        # Initialize with smaller weights
        self._init_weights()

    def _init_weights(self):
        # Use smaller initialization to prevent large values
        nn.init.xavier_uniform_(self.Wq.weight, gain=0.1)
        nn.init.xavier_uniform_(self.Wk.weight, gain=0.1)
        nn.init.xavier_uniform_(self.Wv.weight, gain=0.1)
        
        # Initialize biases to zero
        nn.init.zeros_(self.Wq.bias)
        nn.init.zeros_(self.Wk.bias)
        nn.init.zeros_(self.Wv.bias)

    def forward(self, x):
        x = x.to(self.device)
        
        # Check for NaN in input and replace with zeros
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        # Normalize input to prevent extreme values
        x = self.input_norm(x)
        
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        
        # Add numerical stability checks
        if torch.isnan(Q).any() or torch.isnan(K).any() or torch.isnan(V).any():
            Q = torch.nan_to_num(Q, nan=0.0)
            K = torch.nan_to_num(K, nan=0.0)
            V = torch.nan_to_num(V, nan=0.0)
        
        # Scale the attention scores by sqrt(dimension)
        scaling_factor = torch.sqrt(torch.tensor(self.out_channels, dtype=torch.float, device=self.device))
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / scaling_factor
        
        # Clamp attention scores to prevent extreme values
        attention_scores = torch.clamp(attention_scores, -50.0, 50.0)
        
        # Use more numerically stable softmax
        attention_weights = F.softmax(attention_scores, dim=2)
        
        # Add small epsilon to prevent division by zero
        attention_weights = attention_weights + 1e-8
        attention_weights = attention_weights / attention_weights.sum(dim=2, keepdim=True)
        
        attention_output = torch.bmm(attention_weights, V)
        
        # Check for NaN in output and replace with zeros
        if torch.isnan(attention_output).any():
            attention_output = torch.nan_to_num(attention_output, nan=0.0)

        return attention_output
    

class AttentionGain(nn.Module):
    def __init__(
        self,
        state_dim_per_agent,
        hidden_dims_per_agent,
        action_dim_per_agent,
        n_agents,
        device,
    ):
        super().__init__()

        self.device = device
        self.state_dim_per_agent = state_dim_per_agent
        self.hidden_dims_per_agent = hidden_dims_per_agent
        self.hidden_dims_per_agent.append(action_dim_per_agent)
        self.n_agents = n_agents

        # Build layers without moving them individually
        layers = []
        layers.append(nn.Linear(self.state_dim_per_agent, self.hidden_dims_per_agent[0]))
        layers.append(nn.Tanh())
        
        for layer in range(len(self.hidden_dims_per_agent) - 1):
            layers.append(
                AttentionLayer(
                    self.hidden_dims_per_agent[layer], 
                    self.hidden_dims_per_agent[layer],
                    device  # Pass device to AttentionLayer
                )
            )
            layers.append(
                nn.Linear(
                    self.hidden_dims_per_agent[layer],
                    self.hidden_dims_per_agent[layer + 1],
                )
            )
            if layer < len(self.hidden_dims_per_agent) - 2:
                layers.append(nn.Tanh())

        # Create ModuleList and move all layers to device at once
        self.layers = nn.ModuleList(layers)
        for i, _ in enumerate(self.layers):
            self.layers[i].to(self.device)

    def forward(self, observations, laplacian_mask):
        observations = observations.squeeze(0)
        # Ensure inputs are on the correct device
        observations = observations.to(self.device)
        laplacian_mask = laplacian_mask.to(self.device)
        
        x = observations.unsqueeze(0).repeat(self.n_agents, 1, 1)
        laplacian_mask = laplacian_mask.reshape(-1, self.n_agents).unsqueeze(2)
        hidden_num = 0

        x = self.layers[0](x.reshape(-1, self.state_dim_per_agent)).reshape(
            -1, self.n_agents, self.hidden_dims_per_agent[hidden_num]
        )
        x = self.layers[1](x) * torch.kron(
            laplacian_mask,
            torch.ones(
                (1, 1, self.hidden_dims_per_agent[hidden_num]), device=self.device
            ),
        )
        for layer in range(2, len(self.layers), 3):
            x = self.layers[layer](x)
            x = self.layers[layer + 1](
                x.reshape(-1, self.hidden_dims_per_agent[hidden_num])
            ).reshape(-1, self.n_agents, self.hidden_dims_per_agent[hidden_num + 1])
            if layer + 2 < len(self.layers):
                x = self.layers[layer + 2](x) * torch.kron(
                    laplacian_mask,
                    torch.ones(
                        (1, 1, self.hidden_dims_per_agent[hidden_num + 1]),
                        device=self.device,
                    ),
                )
            hidden_num += 1

        return x
    

class SelfAttentionNonlinearPolicy(nn.Module):

    def __init__(
        self,
        state_dim_per_agent, 
        action_dim_per_agent, 
        n_agents, 
        hidden_channels_dim, 
        r_communication, 
        device
    ):
        super().__init__()

        self.device = device
        self.state_dim_per_agent = state_dim_per_agent
        self.action_dim_per_agent = action_dim_per_agent
        self.state_dim = state_dim_per_agent * n_agents
        self.action_dim = action_dim_per_agent * n_agents
        self.n_agents = n_agents
        self.r_communication = r_communication
        self.hidden_channels_dim = hidden_channels_dim

        self.std = 0.001

        self.layers = AttentionGain(
            self.state_dim_per_agent,
            self.hidden_channels_dim,
            self.action_dim_per_agent * self.state_dim_per_agent,
            self.n_agents,
            self.device,
        ).to(self.device)
    
    def forward(self, x):

        laplacian_mask = self.proximity_laplacian(x[:, :, 2:4])

        x, laplacian_mask = x.to(self.device), laplacian_mask.to(self.device)

        net_out = self.layers(x, laplacian_mask)
        K = (
            net_out.reshape(
                -1,
                self.n_agents,
                self.n_agents,
                self.action_dim_per_agent,
                self.state_dim_per_agent,
            )
            .permute(0, 1, 3, 2, 4)
            .reshape(-1, self.action_dim, self.state_dim)
        )
        action = -torch.bmm(K, x.reshape(x.shape[0], -1, 1)).squeeze(2)
        action = action.view(x.shape[0], self.n_agents, self.action_dim_per_agent)
        action = torch.tanh(action)
        return action[:, :, :2]

    def proximity_laplacian(self, x_t):
        x_t_expanded_1 = x_t.repeat(1, self.n_agents, 1)
        x_t_expanded_2 = torch.kron(
            x_t.contiguous(), torch.ones((1, self.n_agents, 1), device=self.device)
        )
        distances = (
            (x_t_expanded_1 - x_t_expanded_2)
            .norm(p=2, dim=2)
            .reshape(x_t.shape[0], self.n_agents, x_t.shape[1])
        )
        laplacian_mask = distances.le(self.r_communication).float()
        return laplacian_mask
      