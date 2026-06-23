import torch
from torch import nn
from torch import distributions as pyd
import torch.nn.functional as F
import math

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        layers = [nn.Linear(self.in_channels, self.hidden_channels[0]), nn.SiLU()]
        for i in range(len(self.hidden_channels) - 1):
            layers.append(nn.Linear(self.hidden_channels[i], self.hidden_channels[i + 1]))
            if i < len(self.hidden_channels) - 2:
                layers.append(nn.SiLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class attention(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, p, device):
        super().__init__()

        self.device = device
        self.activation_swish = nn.SiLU()
        self.activation_soft = nn.Softmax(dim=2)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.p = p

        # Initialized to avoid unstable training
        self.Aq_in = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Ak_in = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Av_in = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))

        self.Aq_out = nn.Parameter(torch.randn(self.output_dim, self.output_dim))
        self.Ak_out = nn.Parameter(torch.randn(self.output_dim, self.output_dim))

        self.mlp_in = MLP(
            input_dim,
            [hidden_dim]
        ).to(device)

        self.mlp_out = MLP(
            hidden_dim,
            [output_dim]
        ).to(device)

        self.post_processing = MLP(
            output_dim,
            [output_dim]
        ).to(device)

    def forward(self, y_t, laplacian):

        x = y_t.unsqueeze(0).repeat(self.p, 1, 1)

        x = (self.mlp_in(x.reshape(-1, self.input_dim)).reshape(-1, self.p, self.hidden_dim) *
             torch.kron(laplacian.unsqueeze(2), torch.ones((1, 1, self.hidden_dim), device=self.device)))

        Q = self.activation_swish(
            torch.bmm(self.Aq_in.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)))
        K = self.activation_swish(
            torch.bmm(self.Ak_in.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2))).transpose(1, 2)
        V = self.activation_swish(
            torch.bmm(self.Av_in.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)))

        x = self.activation_swish(torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

        x = (self.mlp_out(x.reshape(-1, self.hidden_dim)).reshape(-1, self.p, self.output_dim) *
             torch.kron(laplacian.unsqueeze(2), torch.ones((1, 1, self.output_dim), device=self.device)))

        return 0.5 * x.squeeze(2)


class attention_pg(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, p, na, device):
        super().__init__()

        self.device = device
        self.activation_swish = nn.SiLU()
        self.activation_soft = nn.Softmax(dim=2)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.p = p
        self.na = na

        # Initialized to avoid unstable training
        self.Aq_in = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Ak_in = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Av_in = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))

        self.Aq_out = nn.Parameter(torch.randn(self.output_dim, self.output_dim))
        self.Ak_out = nn.Parameter(torch.randn(self.output_dim, self.output_dim))

        self.mlp_in = MLP(
            input_dim,
            [hidden_dim]
        ).to(device)

        self.mlp_out = MLP(
            hidden_dim,
            [output_dim]
        ).to(device)

        self.post_processing = MLP(
            output_dim,
            [output_dim]
        ).to(device)

    def forward(self, y_t, laplacian):
        x = y_t.unsqueeze(0).repeat(self.p, 1, 1)

        x = (self.mlp_in(x.reshape(-1, self.input_dim)).reshape(-1, self.p, self.hidden_dim) *
             torch.kron(laplacian.unsqueeze(2), torch.ones((1, 1, self.hidden_dim), device=self.device)))

        Q = self.activation_swish(
            torch.bmm(self.Aq_in.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)))
        K = self.activation_swish(
            torch.bmm(self.Ak_in.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2))).transpose(1, 2)
        V = self.activation_swish(
            torch.bmm(self.Av_in.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x.transpose(1, 2)))

        x = self.activation_swish(torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

        x = (self.mlp_out(x.reshape(-1, self.hidden_dim)).reshape(-1, self.p, self.output_dim) *
             torch.kron(laplacian.unsqueeze(2), torch.ones((1, 1, self.output_dim), device=self.device)))

        return 0.1 * x.squeeze(2)


class NeuralNetwork(nn.Module):

    def __init__(self, m, p, device):
        super().__init__()
        self.device = device
        self.m = m
        self.p = p

        self.in_channels = 1
        self.hidden_channels = self.m * self.p
        self.output_dim = 1

        self.layers = attention(self.in_channels, self.hidden_channels, self.output_dim, self.p, self.device).to(self.device)

        self.r_communication = 0.2

    def proximity_laplacian(self, x_t):
        q1 = x_t.repeat(self.p, 1)
        q2 = torch.kron(x_t.contiguous(), torch.ones(self.p, 1, device=self.device))
        q = (q1 - q2).norm(p=2, dim=1).reshape(self.p, self.p)
        laplacian = q.le(self.r_communication).float()
        return laplacian

    def forward(self, y_t, x_t, real_laplacian):
        if real_laplacian is None:
            laplacian = self.proximity_laplacian(x_t)
        else:
            laplacian = real_laplacian
        return self.layers(y_t, laplacian), laplacian


class PolicyGradientNeuralNetwork(nn.Module):

    def __init__(self, m, p, n_agents, device):
        super().__init__()
        self.device = device
        self.m = m
        self.p = p
        self.na = n_agents

        self.in_channels = int(self.p / self.na)
        self.hidden_channels = 1
        self.output_dim = int(self.m / self.na)

        self.std = 0.0001 * torch.ones(self.m, 1, device=self.device)

        self.r_communication = 0.2

        self.layers = attention_pg(self.in_channels, self.hidden_channels, self.output_dim * self.in_channels, self.p, self.na, self.device).to(self.device)

    def proximity_laplacian(self, x_t):
        q1 = x_t.repeat(self.p, 1)
        q2 = torch.kron(x_t.contiguous(), torch.ones(self.p, 1, device=self.device))
        q = (q1 - q2).norm(p=2, dim=1).reshape(self.p, self.p)
        laplacian_bool = q.le(self.r_communication).float()
        laplacian = laplacian_bool * torch.sigmoid(-(2.0) * (q - self.r_communication))
        return laplacian, laplacian_bool

    def forward(self, y_t, x_t, real_laplacian):
        if real_laplacian is None:
            laplacian, laplacian_bool = self.proximity_laplacian(x_t)
        else:
            laplacian = real_laplacian
            laplacian_bool = real_laplacian
        mean = self.layers(y_t, laplacian) @ y_t
        return SquashedNormal(mean, self.std), laplacian_bool


def RandomPositionsProximity(N, Square, dmin, dmax, device):

    pos = [torch.zeros(2, device=device) for i in range(N)]
    for i in range(N):
        ok = False
        while not ok:
            xpos = Square * (torch.rand(1, device=device) - torch.Tensor([0.5]).to(device))
            ypos = Square * (torch.rand(1, device=device) - torch.Tensor([0.5]).to(device))
            j = 0
            ok2 = True
            ok3 = False
            while (ok2) and (j <= i - 1):
                dist = torch.sqrt((pos[j][0] - xpos) * (pos[j][0] - xpos) + (pos[j][1] - ypos) * (pos[j][1] - ypos))
                if dist < dmin:
                    ok2 = False
                if dist <= dmax:
                    ok3 = True
                if ok2:
                    j += 1
            if (ok2 and ok3) or (i == 0):
                ok = True
        pos[i][0] = xpos
        pos[i][1] = ypos

    L = torch.zeros(N, N, device=device)
    L_boolean = torch.eye(N, device=device)
    for i in range(N):
        for j in range(i + 1, N):
            dist = torch.sqrt(
                (pos[j][0] - pos[i][0]) * (pos[j][0] - pos[i][0]) + (pos[j][1] - pos[i][1]) * (pos[j][1] - pos[i][1]))
            if dist <= dmax:
                L[i, j] = -1
                L[j, i] = -1
                L_boolean[i, j] = 1
                L_boolean[j, i] = 1

    D = torch.diag(torch.sum(-L, 1))
    W = torch.zeros(N, N, device=device)

    for i in range(N):
        for j in range(N):
            if L[i, j] < 0:
                W[i, j] = 1/(1 + torch.max(D[i, i], D[j, j]))

    return pos, D + L, torch.eye(N, device=device) - torch.diag(W.sum(dim=1)) + W, L_boolean


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
        transforms = [TanhTransform(), pyd.transforms.AffineTransform(0.0, 20.0)]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu
