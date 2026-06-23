import torch
from torch import nn
from torch import distributions as pyd
import torch.nn.functional as F
import math

class RobotDynamics():
    def __init__(self, dt, n_agents):
        self.dt = dt
        self.n_agents = n_agents
        self.n_states = 4

    def forward(self, x, u):
        x_dot = torch.zeros_like(x)
        x_dot[:, :, 0] = x[:, :, 2] * torch.cos(x[:, :, 3]) 
        x_dot[:, :, 1] = x[:, :, 2] * torch.sin(x[:, :, 3])
        x_dot[:, :, 2] = u[:, :, 0]
        x_dot[:, :, 3] = u[:, :, 1]

        return x + self.dt * x_dot


class AttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.Wq = nn.Linear(self.in_channels, self.out_channels)
        self.Wk = nn.Linear(self.in_channels, self.out_channels)
        self.Wv = nn.Linear(self.in_channels, self.out_channels)

    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x).transpose(1, 2)
        V = self.Wv(x)

        attention_weights = F.softmax(torch.bmm(Q, K), dim=2)
        attention_output = torch.bmm(attention_weights, V)

        return attention_output
    

class AttentionGain(nn.Module):

    def __init__(self, state_dim_per_agent, hidden_dims_per_agent, action_dim_per_agent, n_agents, device):
        super().__init__()

        self.device = device
        self.state_dim_per_agent = state_dim_per_agent
        self.hidden_dims_per_agent = hidden_dims_per_agent
        self.hidden_dims_per_agent.append(action_dim_per_agent)
        self.n_agents = n_agents

        # Initialized to avoid unstable training
        self.layers = [nn.Linear(self.state_dim_per_agent, self.hidden_dims_per_agent[0]).to(device)]
        self.layers.append(nn.Tanh())
        for layer in range(len(self.hidden_dims_per_agent) - 1):
            self.layers.append(AttentionLayer(self.hidden_dims_per_agent[layer], self.hidden_dims_per_agent[layer]))
            self.layers.append(nn.Linear(self.hidden_dims_per_agent[layer], self.hidden_dims_per_agent[layer + 1]).to(device))
            if layer < len(self.hidden_dims_per_agent) - 2:
                self.layers.append(nn.Tanh())

        self.layers = nn.ModuleList(self.layers)

    def forward(self, observations, laplacian_mask):
        x = observations.unsqueeze(0).repeat(self.n_agents, 1, 1)
        laplacian_mask = laplacian_mask.reshape(-1, self.n_agents).unsqueeze(2)
        hidden_num = 0

        x = self.layers[0](x.reshape(-1, self.state_dim_per_agent)).reshape(-1, self.n_agents, self.hidden_dims_per_agent[hidden_num])
        x = self.layers[1](x) * torch.kron(laplacian_mask, torch.ones((1, 1, self.hidden_dims_per_agent[hidden_num]), device=self.device))
        for layer in range(2, len(self.layers), 3):
            x = self.layers[layer](x)
            x = self.layers[layer + 1](x.reshape(-1, self.hidden_dims_per_agent[hidden_num])).reshape(-1, self.n_agents, self.hidden_dims_per_agent[hidden_num + 1])
            if layer + 2 < len(self.layers):
                x = self.layers[layer + 2](x) * torch.kron(laplacian_mask, torch.ones((1, 1, self.hidden_dims_per_agent[hidden_num + 1]), device=self.device))
            hidden_num += 1
            
        return x


class SelfAttentionNonlinearPolicy(nn.Module):

    def __init__(self, state_dim_per_agent, action_dim_per_agent, n_agents, hidden_channels_dim, r_communication, device):
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

        self.layers = AttentionGain(self.state_dim_per_agent, self.hidden_channels_dim, self.action_dim_per_agent * self.state_dim_per_agent, self.n_agents, self.device).to(self.device)

    def proximity_laplacian(self, x_t):
        x_t_expanded_1 = x_t.repeat(1, self.n_agents, 1)
        x_t_expanded_2 = torch.kron(x_t.contiguous(), torch.ones((1, self.n_agents, 1), device=self.device))
        distances = (x_t_expanded_1 - x_t_expanded_2).norm(p=2, dim=2).reshape(x_t.shape[0], self.n_agents, x_t.shape[1])
        laplacian_mask = distances.le(self.r_communication).float()
        return laplacian_mask

    def forward(self, observations, positions):
        laplacian_mask = self.proximity_laplacian(positions)
        net_out = self.layers(observations, laplacian_mask)
        K = net_out.reshape(-1, self.n_agents, self.n_agents, self.action_dim_per_agent, self.state_dim_per_agent).permute(0, 1, 3, 2, 4).reshape(-1, self.action_dim, self.state_dim)
        action = -torch.bmm(K, observations.unsqueeze(2)).squeeze(2)
        return SquashedNormal(action, self.std * torch.ones_like(action, device=self.device))
        # return action
    

class MLPPolicy(nn.Module):
    """
    Simple MLP policy for multi-agent control that replaces the self-attention architecture.
    
    This policy directly maps the flattened state vector to control actions using
    fully connected layers.
    """
    def __init__(self, state_dim_per_agent, action_dim_per_agent, n_agents, hidden_dims=[64, 64], device="cpu"):
        super().__init__()
        self.device = device
        self.state_dim_per_agent = state_dim_per_agent
        self.action_dim_per_agent = action_dim_per_agent
        self.state_dim = state_dim_per_agent * n_agents
        self.action_dim = action_dim_per_agent * n_agents
        self.n_agents = n_agents
        
        # Create MLP layers
        layers = []
        
        # Input layer
        input_dim = self.state_dim
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], self.action_dim))
        
        self.mlp = nn.Sequential(*layers).to(device)
    
    def forward(self, observations, positions=None):
        """
        Forward pass through the MLP policy.
        
        Args:
            observations: Tensor of shape [batch_size, state_dim] containing flattened states
            positions: Tensor of shape [batch_size, n_agents, space_dim] - not used in MLP but
                      kept for compatibility with the original policy interface
        
        Returns:
            Tensor of shape [batch_size, action_dim] containing actions for all agents
        """
        # Process through MLP
        actions = self.mlp(observations)
        return actions


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dims=[64, 64], device="cpu"):
        super().__init__()
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.network = nn.Sequential(*layers).to(device)
    
    def forward(self, state):
        return self.network(state)


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu
