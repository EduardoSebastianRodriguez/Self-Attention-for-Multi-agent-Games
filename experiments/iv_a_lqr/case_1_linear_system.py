import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.linalg as la
from functions import NeuralNetwork, RandomPositionsProximity, PolicyGradientNeuralNetwork


np.random.seed(0)
torch.manual_seed(0)
device = "cpu" if not torch.has_cuda else "cuda:0"

# Parameters of the system
n = 5
N = 30
m = 5
p = 5
num_agents = 5

Pi_delta_0 = 1e-4
Pi_omega = 1e-5
Pi_v = 1e-5

# Initialization of the state, system matrices and cost matrices
mu_0 = 0.1 * torch.randn(n, device=device).unsqueeze(dim=1)

A_t = torch.eye(n, device=device)

B_t = torch.diag(torch.rand(n, device=device))

C_t = torch.eye(n, device=device)

M = torch.zeros((N + 1) * p, (N + 1) * p, device=device)
for t in range(N + 1):
    M_matrix = np.diag(np.random.uniform(1/N, 5/N, p))
    q, _ = la.qr(np.random.rand(p, p))
    Mt = torch.Tensor(q.T @ M_matrix @ q).to(device)
    M[p * t: p * (t + 1), p * t: p * (t + 1)] = Mt

R = torch.zeros(N * m, N * m, device=device)
for t in range(N):
    M_matrix = np.diag(np.random.uniform(1/N, 5/N, m))
    q, _ = la.qr(np.random.rand(m, m))
    Rt = torch.Tensor(q.T @ M_matrix @ q).to(device)
    R[m * t: m * (t + 1), m * t: m * (t + 1)] = Rt

# Generate topologies for the case of known graphs
topologies = torch.zeros((N + 1) * n, (N + 1) * n, device=device)
for t in range(N + 1):
    _, _, _, topology = RandomPositionsProximity(n,
                                                 torch.Tensor([1.0]).to(device),
                                                 torch.Tensor([0.12]).to(device),
                                                 torch.Tensor([0.17]).to(device),
                                                 device)
    topologies[n * t: n * (t + 1), n * t: n * (t + 1)] = topology

# Optimal Centralized LQR
P_riccati = M[p * N: p * (N + 1), p * N: p * (N + 1)].clone()
K_optimal_lqr = torch.zeros((N + 1) * m, (N + 1) * p, device=device)
for t in reversed(range(N)):
    M_t = M[m * t: m * (t + 1), m * t: m * (t + 1)]
    R_t = R[m * t: m * (t + 1), m * t: m * (t + 1)]
    K_riccati = -torch.pinverse(R_t + B_t.T @ P_riccati @ B_t) @ B_t.T @ P_riccati @ A_t
    K_optimal_lqr[m * t: m * (t + 1), m * t: m * (t + 1)] = K_riccati
    P_riccati = M_t + A_t.T @ P_riccati @ A_t - A_t @ P_riccati @ B_t @ torch.pinverse(R_t + B_t.T @ P_riccati @ B_t) @ B_t.T @ P_riccati @ A_t

# Furieri's solution
topologies_vec = topologies.reshape(-1)
d = int(topologies_vec.sum())
projection_matrix = torch.zeros(m * p * N * (N + 1), d, device=device)
counter = 0
for i in range(m * p * N * (N + 1)):
    if topologies_vec[i] == 1.:
        projection_matrix[i, counter] = 1.
        counter += 1
z = 0.1 * torch.rand(d, device=device)
r = 0.1

# Our approach with given topology
T = 1000000
lr = 1e-5
neural_network_fixed = NeuralNetwork(m, p, device)
neural_network_fixed.train()
optimizer_fixed = torch.optim.Adam(neural_network_fixed.parameters(), lr)

# Our approach
neural_network_state = NeuralNetwork(m, p, device)
neural_network_state.train()
optimizer_state = torch.optim.Adam(neural_network_state.parameters(), lr)

# Policy gradient state topology
gamma = 1.0
neural_network_pg_state = PolicyGradientNeuralNetwork(m, p, num_agents, device)
neural_network_pg_state.train()
optimizer_pg_state = torch.optim.Adam(neural_network_pg_state.parameters(), lr)

# Logs
log_loss_ours = np.zeros(T)
log_loss_optimal_lqr = np.zeros(T)
log_loss_furieri = np.zeros(T)
log_loss_fixed = np.zeros(T)
log_loss_pg_state = np.zeros(T)

for i in range(T):

    # Reset optimizers
    optimizer_state.zero_grad()

    noise_furieri = torch.rand(d, device=device)
    noise_furieri *= r * torch.rand(1, device=device) / noise_furieri.norm(p=2)

    optimizer_fixed.zero_grad()

    optimizer_pg_state.zero_grad()

    # Generate noise for initial state
    delta_0 = (2 * Pi_delta_0 * (torch.rand(n, device=device) - 0.5)).unsqueeze(dim=1)

    # Initialize the first state of the system
    x_t_ours = mu_0 + delta_0
    x_t_optimal_lqr = mu_0 + delta_0
    x_t_furieri = mu_0 + delta_0
    x_t_fixed = mu_0 + delta_0
    x_t_pg_state = mu_0 + delta_0

    # Initialize costs
    f_ours = torch.Tensor([0.0]).to(device)
    f_optimal_lqr = torch.Tensor([0.0]).to(device)
    f_furieri = torch.Tensor([0.0]).to(device)
    f_fixed = torch.Tensor([0.0]).to(device)
    f_pg_state = torch.Tensor([0.0]).to(device)

    cumulative_reward_pg_state = torch.zeros(N, device=device)
    cumulative_log_prob_pg_state = torch.zeros(N, device=device)
    discounted_rewards_pg_state = torch.zeros(N, device=device)
    policy_gradient_pg_state = []

    # Run rollout
    for t in range(N):
        # Generate system and measurement noise
        v_t = (2 * Pi_v * (torch.rand(p, device=device) - 0.5)).unsqueeze(dim=1)
        w_t = (2 * Pi_omega * (torch.rand(n, device=device) - 0.5)).unsqueeze(dim=1)

        # Compute outputs
        y_t_ours = C_t @ x_t_ours + v_t
        y_t_optimal_lqr = C_t @ x_t_optimal_lqr + v_t
        y_t_furieri = C_t @ x_t_furieri + v_t
        y_t_fixed = C_t @ x_t_fixed + v_t
        y_t_pg_state = C_t @ x_t_pg_state + v_t

        # Compute control gains
        gain_ours, laplacian = neural_network_state(y_t_ours.clone(), x_t_ours.clone(), None)
        gain_optimal_lqr = K_optimal_lqr[m * t: m * (t + 1), m * t: m * (t + 1)]
        gain_furieri = (projection_matrix @ (z + noise_furieri).unsqueeze(dim=1)).reshape(m * N, p * (N + 1))
        gain_fixed, _ = neural_network_fixed(y_t_fixed.clone(), x_t_fixed.clone(), topologies[n * t: n * (t + 1), n * t: n * (t + 1)].clone())

        # Compute control action
        u_t_ours = gain_ours @ y_t_ours
        u_t_optimal_lqr = gain_optimal_lqr @ y_t_optimal_lqr
        u_t_furieri = gain_furieri[m * t: m * (t + 1), p * t: p * (t + 1)] @ y_t_furieri
        u_t_fixed = gain_fixed @ y_t_fixed

        action_distribution_pg_state, laplacian_pg = neural_network_pg_state(y_t_pg_state.clone(), x_t_pg_state.clone(), None)
        u_t_pg_state = action_distribution_pg_state.rsample()
        log_prob_pg_state = action_distribution_pg_state.log_prob(u_t_pg_state)

        # Compute next state
        x_t_ours = A_t @ x_t_ours + B_t @ u_t_ours + w_t
        x_t_optimal_lqr = A_t @ x_t_optimal_lqr + B_t @ u_t_optimal_lqr + w_t
        x_t_furieri = A_t @ x_t_furieri + B_t @ u_t_furieri + w_t
        x_t_fixed = A_t @ x_t_fixed + B_t @ u_t_fixed + w_t
        x_t_pg_state = A_t @ x_t_pg_state + B_t @ u_t_pg_state + w_t

        # Current cost matrices
        M_t = M[p * t: p * (t + 1), p * t: p * (t + 1)]
        R_t = R[m * t: m * (t + 1), m * t: m * (t + 1)]

        # Compute current cost
        f_ours += (y_t_ours.T @ M_t @ y_t_ours + u_t_ours.T @ R_t @ u_t_ours)[0]
        f_optimal_lqr += (y_t_optimal_lqr.T @ M_t @ y_t_optimal_lqr + u_t_optimal_lqr.T @ R_t @ u_t_optimal_lqr)[0]
        f_furieri += (y_t_furieri.T @ M_t @ y_t_furieri + u_t_furieri.T @ R_t @ u_t_furieri)[0]
        if f_furieri > 1e10:
            f_furieri = 1e10
        f_fixed += (y_t_fixed.T @ M_t @ y_t_fixed + u_t_fixed.T @ R_t @ u_t_fixed)[0]
        f_pg_state += (y_t_pg_state.T @ M_t @ y_t_pg_state + u_t_pg_state.T @ R_t @ u_t_pg_state)[0]

        cumulative_reward_pg_state[t] = -(y_t_pg_state.T @ M_t @ y_t_pg_state + u_t_pg_state.T @ R_t @ u_t_pg_state)[0]
        cumulative_log_prob_pg_state[t] = log_prob_pg_state.mean()

    # Generate final measurement noise
    v_t = (2 * Pi_v * (torch.rand(p, device=device) - 0.5)).unsqueeze(dim=1)

    # Compute las output
    y_t_ours = C_t @ x_t_ours + v_t
    y_t_optimal_lqr = C_t @ x_t_optimal_lqr + v_t
    y_t_furieri = C_t @ x_t_furieri + v_t
    y_t_fixed = C_t @ x_t_fixed + v_t
    y_t_pg_state = C_t @ x_t_pg_state + v_t

    # Obtain running laplacian
    _, laplacian = neural_network_state(y_t_ours.clone(), x_t_ours.clone(), None)
    topology_t = topologies[n * N: n * (N + 1), n * N: n * (N + 1)]

    # Compute final cost
    f_ours += (y_t_ours.T @ M[p * N: p * (N + 1), p * N: p * (N + 1)] @ y_t_ours)[0]
    f_optimal_lqr += (y_t_optimal_lqr.T @ M[p * N: p * (N + 1), p * N: p * (N + 1)] @ y_t_optimal_lqr)[0]
    f_furieri += (y_t_furieri.T @ M[p * N: p * (N + 1), p * N: p * (N + 1)] @ y_t_furieri)[0]
    if f_furieri > 1e6:
        f_furieri = 1e6
    f_fixed += (y_t_fixed.T @ M[p * N: p * (N + 1), p * N: p * (N + 1)] @ y_t_fixed)[0]
    f_pg_state += (y_t_pg_state.T @ M[p * N: p * (N + 1), p * N: p * (N + 1)] @ y_t_pg_state)[0]

    for t in range(N):
        Gt = 0
        pw = 0
        for reward in cumulative_reward_pg_state[t:]:
            Gt = Gt + (gamma ** pw) * reward
            pw = pw + 1
        discounted_rewards_pg_state[t] = Gt

    for log_prob, Gt in zip(cumulative_log_prob_pg_state, discounted_rewards_pg_state):
        policy_gradient_pg_state.append(-log_prob * Gt)

    # Store cost at iteration i
    log_loss_ours[i] = f_ours.detach().cpu().numpy()
    log_loss_optimal_lqr[i] = f_optimal_lqr.detach().cpu().numpy()
    if f_furieri == 1e6:
        log_loss_furieri[i] = f_furieri
    else:
        log_loss_furieri[i] = f_furieri.detach().cpu().numpy()
    log_loss_fixed[i] = f_fixed.detach().cpu().numpy()
    log_loss_pg_state[i] = f_pg_state.detach().cpu().numpy()

    if i % 100 == 0:
        print('-------------------------------------')
        print('\n CENTRALIZED LQR: iteration {} with loss {}.'.format(i, log_loss_optimal_lqr[i]))
        print('\n FURIERI: iteration {} with loss {}.'.format(i, log_loss_furieri[i]))
        print('\n OURS FIXED: iteration {} with loss {}.'.format(i, log_loss_fixed[i]))
        print('\n OURS TIME-VARYING: iteration {} with loss {}.'.format(i, log_loss_ours[i]))
        print('\n PG OURS TIME-VARYING: iteration {} with loss {}.'.format(i, log_loss_pg_state[i]))
        print('\n-------------------------------------')

    # Backprop
    f_ours.backward()
    optimizer_state.step()

    f_fixed.backward()
    optimizer_fixed.step()

    if f_furieri < 1e6:
        pseudo_gradient = f_furieri * d / r / 2 * noise_furieri
        z -= lr * pseudo_gradient

    loss = torch.stack(policy_gradient_pg_state).sum()
    loss.backward()
    #torch.nn.utils.clip_grad_norm_(neural_network_pg_state.parameters(), 0.3)
    optimizer_pg_state.step()

np.save("loss_case_1_ours", log_loss_ours)
np.save("loss_case_1_optimal_lqr", log_loss_optimal_lqr)
np.save("loss_case_1_furieri", log_loss_furieri)
np.save("loss_case_1_fixed", log_loss_fixed)
np.save("loss_case_1_pg_state", log_loss_pg_state)

plt.rcParams.update({'font.size': 22})
plt.figure()
plt.plot(log_loss_optimal_lqr, '--', linewidth=3, label="centralized LQR")
plt.plot(log_loss_ours, linewidth=3, label="ours")
plt.plot(log_loss_furieri, linewidth=3, label="furieri")
plt.plot(log_loss_fixed, linewidth=3, label="ours fixed")
plt.plot(log_loss_pg_state, linewidth=3, label="ours pg state graph")
plt.xlabel('iterations')
plt.ylabel('cost function')
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.show()
