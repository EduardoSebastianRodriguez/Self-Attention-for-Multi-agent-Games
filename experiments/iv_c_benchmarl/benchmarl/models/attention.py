from __future__ import annotations

import importlib
from dataclasses import dataclass, MISSING
from math import prod
from typing import List, Optional

import torch
from tensordict.utils import _unravel_key_to_tuple, NestedKey
from torch import nn, Tensor
from torch import distributions as pyd
import torch.nn.functional as F
import math

from benchmarl.models.common import Model, ModelConfig

_has_torch_geometric = importlib.util.find_spec("torch_geometric") is not None
if _has_torch_geometric:
    import torch_geometric
    from torch_geometric.transforms import BaseTransform

    class _RelVel(BaseTransform):
        """Transform that reads graph.vel and writes node1.vel - node2.vel in the edge attributes"""

        def __init__(self):
            pass

        def __call__(self, data):
            (row, col), vel, pseudo = data.edge_index, data.vel, data.edge_attr

            cart = vel[row] - vel[col]
            cart = cart.view(-1, 1) if cart.dim() == 1 else cart

            if pseudo is not None:
                pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
                data.edge_attr = torch.cat([pseudo, cart.type_as(pseudo)], dim=-1)
            else:
                data.edge_attr = cart
            return data


TOPOLOGY_TYPES = {"full", "empty", "from_pos"}


class AttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device

        # Layer normalization for input stabilization
        self.input_norm = nn.LayerNorm(in_channels)

        self.Wq = nn.Linear(self.in_channels, self.out_channels)
        self.Wk = nn.Linear(self.in_channels, self.out_channels)
        self.Wv = nn.Linear(self.in_channels, self.out_channels)

        self._init_weights()

    def _init_weights(self):
        # Small-gain Xavier initialization to prevent large activations
        nn.init.xavier_uniform_(self.Wq.weight, gain=0.1)
        nn.init.xavier_uniform_(self.Wk.weight, gain=0.1)
        nn.init.xavier_uniform_(self.Wv.weight, gain=0.1)

        nn.init.zeros_(self.Wq.bias)
        nn.init.zeros_(self.Wk.bias)
        nn.init.zeros_(self.Wv.bias)

    def forward(self, x):
        x = x.to(self.device)

        # Guard against NaNs in the input
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)

        # Normalize input to prevent extreme values
        x = self.input_norm(x)

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        if torch.isnan(Q).any() or torch.isnan(K).any() or torch.isnan(V).any():
            Q = torch.nan_to_num(Q, nan=0.0)
            K = torch.nan_to_num(K, nan=0.0)
            V = torch.nan_to_num(V, nan=0.0)

        # Scale the attention scores by sqrt(dimension)
        scaling_factor = torch.sqrt(
            torch.tensor(self.out_channels, dtype=torch.float, device=self.device)
        )
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / scaling_factor

        # Clamp attention scores to prevent extreme values
        attention_scores = torch.clamp(attention_scores, -50.0, 50.0)

        attention_weights = F.softmax(attention_scores, dim=2)

        # Add small epsilon to prevent division by zero, then renormalize
        attention_weights = attention_weights + 1e-8
        attention_weights = attention_weights / attention_weights.sum(dim=2, keepdim=True)

        attention_output = torch.bmm(attention_weights, V)

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
                    device,
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


class SelfAttentionNonlinearPolicy(Model):

    def __init__(
        self,
        input_spec,
        output_spec,
        agent_group,
        input_has_agent_dim,
        n_agents,
        centralised,
        share_params,
        device,
        action_spec,
        model_index,
        is_critic,
        hidden_channels_dim,
        position_key,
        velocity_key,
        exclude_pos_from_node_features,
        topology,
        self_loops,
        edge_radius,
        pos_features,
        vel_features,
    ):
        super().__init__(
            input_spec,
            output_spec,
            agent_group,
            input_has_agent_dim,
            n_agents,
            centralised,
            share_params,
            device,
            action_spec,
            model_index,
            is_critic,
        )
        self.position_key = position_key
        self.velocity_key = velocity_key
        self._full_position_key = None
        self._full_velocity_key = None
        self.exclude_pos_from_node_features = exclude_pos_from_node_features
        self.topology = topology
        self.self_loops = self_loops
        self.edge_radius = edge_radius
        self.pos_features = pos_features
        self.vel_features = vel_features
        if self.pos_features > 0:
            self.pos_features += 1  # We will add also 1-dimensional distance

        self.input_features = sum(
            [
                spec.shape[-1]
                for key, spec in self.input_spec.items(True, True)
                if _unravel_key_to_tuple(key)[-1] not in (position_key, velocity_key)
            ]
        )  # Input keys
        if self.position_key is not None and not self.exclude_pos_from_node_features:
            self.input_features += self.pos_features - 1
        if self.velocity_key is not None:
            self.input_features += self.vel_features

        self.edge_index = _get_edge_index(
            topology=self.topology,
            self_loops=self.self_loops,
            device=self.device,
            n_agents=self.n_agents,
        )

        self.action_dim_per_agent = self.output_leaf_spec.shape[-1]
        self.state_dim = self.input_features * n_agents
        self.action_dim = self.action_dim_per_agent * n_agents
        self.n_agents = n_agents
        self.r_communication = edge_radius
        self.hidden_channels_dim = hidden_channels_dim

        self.std = 0.001

        self.layers = AttentionGain(
            self.input_features,
            self.hidden_channels_dim,
            self.action_dim_per_agent * self.input_features,
            self.n_agents,
            self.device,
        ).to(self.device)

    def _forward(self, tensordict):
        # Gather in_key
        input = [
            tensordict.get(in_key)
            for in_key in self.in_keys
            if _unravel_key_to_tuple(in_key)[-1]
            not in (self.position_key, self.velocity_key)
        ]

        # Retrieve position
        if self.position_key is not None:
            if self._full_position_key is None:  # Run once to find full key
                self._full_position_key = self._get_key_terminating_with(
                    list(tensordict.keys(True, True)), self.position_key
                )
                pos = tensordict.get(self._full_position_key)
                if pos.shape[-1] != self.pos_features - 1:
                    raise ValueError(
                        f"Position key in tensordict is {pos.shape[-1]}-dimensional, "
                        f"while model was configured with pos_features={self.pos_features-1}"
                    )
            else:
                pos = tensordict.get(self._full_position_key)
            if not self.exclude_pos_from_node_features:
                input.append(pos)
        else:
            pos = None

        # Retrieve velocity
        if self.velocity_key is not None:
            if self._full_velocity_key is None:  # Run once to find full key
                self._full_velocity_key = self._get_key_terminating_with(
                    list(tensordict.keys(True, True)), self.velocity_key
                )
                vel = tensordict.get(self._full_velocity_key)
                if vel.shape[-1] != self.vel_features:
                    raise ValueError(
                        f"Velocity key in tensordict is {vel.shape[-1]}-dimensional, "
                        f"while model was configured with vel_features={self.vel_features}"
                    )
            else:
                vel = tensordict.get(self._full_velocity_key)
            input.append(vel)
        else:
            vel = None

        input = torch.cat(input, dim=-1)
        batch_size = input.shape[:-2]

        # The radius function from torch_cluster cannot handle the mps device,
        # so build the graph on cpu.
        input = input.to("cpu")
        pos = pos.to("cpu")
        graph = _batch_from_dense_to_ptg(
            x=input,
            edge_index=self.edge_index,
            pos=pos,
            vel=vel,
            self_loops=self.self_loops,
            edge_radius=self.edge_radius,
        )

        self.num_envs, _, obs_shape = graph.x.shape
        observations = graph.x.view(self.num_envs, -1)

        laplacian_mask = self._make_laplacian_mask(graph.edge_index)

        observations, laplacian_mask = observations.to(self.device), laplacian_mask.to(self.device)

        net_out = self.layers(observations, laplacian_mask)
        K = (
            net_out.reshape(
                -1,
                self.n_agents,
                self.n_agents,
                self.action_dim_per_agent,
                self.input_features,
            )
            .permute(0, 1, 3, 2, 4)
            .reshape(-1, self.action_dim, self.state_dim)
        )
        action = -torch.bmm(K, observations.unsqueeze(2)).squeeze(2)
        action = torch.tanh(action.view(self.num_envs, self.n_agents, self.action_dim_per_agent))
        tensordict.set(self.out_key, action)  # action shape: num_envs x num_agents x 4
        return tensordict

    def _make_laplacian_mask(self, edge_index):
        mask = torch.zeros(self.num_envs, self.n_agents, self.n_agents, device=self.device)
        for i, agent_id_1 in enumerate(edge_index[0]):
            agent_id_2 = edge_index[1][i]
            env_index = agent_id_1 // self.n_agents
            agent_id_1 = agent_id_1 % self.n_agents
            agent_id_2 = agent_id_2 % self.n_agents
            mask[env_index, agent_id_1, agent_id_2] = 1
        return mask

    def _get_key_terminating_with(self, keys: List[NestedKey], key: str) -> NestedKey:
        for k in keys:
            k_tuple = _unravel_key_to_tuple(k)
            if (
                k_tuple[-1] == key
                and self.agent_group in k_tuple
                and not "next" == k_tuple[0]
            ):
                return k
        raise KeyError(
            f"Key terminating with {key} and containing {self.agent_group} not found in keys: {keys}. "
            f"If you are using the GNN in a `SequenceModel` and want to use this key, it needs to be the first model."
        )

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
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


def _get_edge_index(topology: str, self_loops: bool, n_agents: int, device: str):
    if topology == "full":
        adjacency = torch.ones(n_agents, n_agents, device=device, dtype=torch.long)
        edge_index, _ = torch_geometric.utils.dense_to_sparse(adjacency)
        if not self_loops:
            edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)
    elif topology == "empty":
        if self_loops:
            edge_index = (
                torch.arange(n_agents, device=device, dtype=torch.long)
                .unsqueeze(0)
                .repeat(2, 1)
            )
        else:
            edge_index = torch.empty((2, 0), device=device, dtype=torch.long)
    elif topology == "from_pos":
        edge_index = None
    else:
        raise ValueError(f"Topology {topology} not supported")

    return edge_index


def _batch_from_dense_to_ptg(
    x: Tensor,
    edge_index: Optional[Tensor],
    self_loops: bool,
    pos: Tensor = None,
    vel: Tensor = None,
    edge_radius: Optional[float] = None,
) -> torch_geometric.data.Batch:
    batch_size = prod(x.shape[:-2])
    n_agents = x.shape[-2]
    if pos is not None:
        pos = pos.view(-1, pos.shape[-1])
    if vel is not None:
        vel = vel.view(-1, vel.shape[-1])

    b = torch.arange(batch_size, device=x.device)

    graphs = torch_geometric.data.Batch()
    graphs.ptr = torch.arange(0, (batch_size + 1) * n_agents, n_agents)
    graphs.batch = torch.repeat_interleave(b, n_agents)
    graphs.x = x
    graphs.pos = pos
    graphs.vel = vel
    graphs.edge_attr = None

    if edge_index is not None:
        n_edges = edge_index.shape[1]
        # Tensor of shape [batch_size * n_edges]
        # in which edges corresponding to the same graph have the same index.
        batch = torch.repeat_interleave(b, n_edges)
        # Edge index for the batched graphs of shape [2, n_edges * batch_size]
        # we sum to each batch an offset of batch_num * n_agents to make sure that
        # the adjacency matrices remain independent
        batch_edge_index = edge_index.repeat(1, batch_size) + batch * n_agents
        graphs.edge_index = batch_edge_index
    else:
        if pos is None:
            raise RuntimeError("from_pos topology needs positions as input")
        graphs.edge_index = torch_geometric.nn.pool.radius_graph(
            graphs.pos, batch=graphs.batch, r=edge_radius, loop=self_loops
        )

    graphs = graphs.to(x.device)
    if pos is not None:
        graphs = torch_geometric.transforms.Cartesian(norm=False)(graphs)
        graphs = torch_geometric.transforms.Distance(norm=False)(graphs)
    if vel is not None:
        graphs = _RelVel()(graphs)

    return graphs


@dataclass
class AttentionConfig(ModelConfig):
    """Dataclass config for our self-attention nonlinear policy."""

    hidden_channels_dim: int = MISSING
    position_key: str = MISSING
    velocity_key: str = MISSING
    exclude_pos_from_node_features: bool = MISSING
    topology: str = MISSING
    self_loops: bool = MISSING
    edge_radius: int = MISSING
    pos_features: int = MISSING
    vel_features: int = MISSING

    @staticmethod
    def associated_class():
        return SelfAttentionNonlinearPolicy
