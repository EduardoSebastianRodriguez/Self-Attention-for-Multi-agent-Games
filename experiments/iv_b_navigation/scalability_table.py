#!/usr/bin/env python

import numpy as np
import torch
from tqdm import tqdm
import dpilqr
from dpilqr import split_agents
from functions_batch import SelfAttentionNonlinearPolicy

device = "cpu" #if not torch.cuda.is_available() else "cuda:0"

def train_one_seed(seed, n_agents=15, T=500):
    # Set seeds for reproducibility of this specific run
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    n_states = 4
    n_controls = 2
    x_dims = [n_states] * n_agents
    u_dims = [n_controls] * n_agents
    n_dims = [2] * n_agents

    x0, xf = dpilqr.random_setup(
        n_agents,
        n_states,
        is_rotation=False,
        rel_dist=2.0,
        var=n_agents / 2,
        n_d=2,
        random=True,
    )

    dt = 0.05
    N = 100
    radius = 0.5
    ids = [100 + i for i in range(n_agents)]

    lr = 1e-3
    gamma = 1.0
    
    neural_network_pg_state = SelfAttentionNonlinearPolicy(n_states, n_controls, n_agents, [64, 64], radius, device)
    neural_network_pg_state.train()
    optimizer_pg_state = torch.optim.Adam(neural_network_pg_state.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_pg_state, gamma=0.9999)

    Q = np.eye(n_states) 
    R = np.eye(n_controls)
    Qf = 100 * np.eye(n_states)

    Q_torch = [torch.from_numpy(Q) for _ in range(n_agents)]
    R_torch = [torch.from_numpy(R) for _ in range(n_agents)]
    Qf_torch = [torch.from_numpy(Qf) for _ in range(n_agents)]

    Q_sys = torch.block_diag(*Q_torch).to(device).to(torch.float32)
    R_sys = torch.block_diag(*R_torch).to(device).to(torch.float32)
    Qf_sys = torch.block_diag(*Qf_torch).to(device).to(torch.float32)

    B_k = [torch.Tensor([[0., 0.],
                         [0., 0.],
                         [1., 0.],
                         [0., 1.]
                         ]) for k in range(n_agents)]
    B_t = torch.block_diag(*B_k).to(device)

    final_loss = 0.0

    for i in range(T):
        optimizer_pg_state.zero_grad()
        f_pg_state = torch.Tensor([0.0]).to(device)
        cumulative_reward_pg_state = torch.zeros(N, device=device)
        cumulative_log_prob_pg_state = torch.zeros(N, device=device)
        discounted_rewards_pg_state = torch.zeros(N, device=device)
        policy_gradient_pg_state = []

        x_t_pg_state = torch.from_numpy(x0).to(device).to(torch.float32)
        x_desired = torch.from_numpy(xf).to(device).to(torch.float32)

        for t in range(N):
            action_distribution_pg_state = neural_network_pg_state(
                x_t_pg_state.clone().reshape(1, -1) - x_desired.clone().reshape(1, -1), 
                x_t_pg_state.clone().reshape(1, n_agents, n_states)[:, :, :2]
            )
            u_t_pg_state = action_distribution_pg_state.rsample()
            log_prob_pg_state = action_distribution_pg_state.log_prob(u_t_pg_state)
            u_t_pg_state = u_t_pg_state.reshape(-1, 1)
            log_prob_pg_state = log_prob_pg_state.reshape(-1, 1)

            # Compute next state
            A_t = torch.zeros(n_states * n_agents, n_states * n_agents, device=device)
            for k in range(n_agents):
                A_t[n_states * k, n_states * k + 2] = torch.cos(x_t_pg_state[4 * k + 3, 0])
                A_t[n_states * k, n_states * k + 3] = -x_t_pg_state[4 * k + 2, 0] * torch.sin(x_t_pg_state[4 * k + 3, 0])
                A_t[n_states * k + 1, n_states * k + 2] = torch.sin(x_t_pg_state[4 * k + 3, 0])
                A_t[n_states * k + 1, n_states * k + 3] = x_t_pg_state[4 * k + 2, 0] * torch.cos(x_t_pg_state[4 * k + 3, 0])

            x_t_pg_state = (A_t @ x_t_pg_state + B_t @ u_t_pg_state) * dt + x_t_pg_state

            # Compute current sparse cost matrices
            reference_cost = (x_t_pg_state - x_desired).T @ Q_sys @ (x_t_pg_state - x_desired) + u_t_pg_state.T @ R_sys @ u_t_pg_state

            Q1 = x_t_pg_state.reshape(n_agents, n_states)[:, :2].repeat(n_agents, 1)
            Q2 = torch.kron(x_t_pg_state.reshape(n_agents, n_states)[:, :2], torch.ones((n_agents, 1), device=device))
            q = (Q1 - Q2).norm(p=2, dim=1).reshape(n_agents, n_agents)
            distance_costs = (torch.min(torch.zeros_like(q), q - radius) ** 2).fill_diagonal_(0.)
            proximity_cost = distance_costs.sum()

            cost_t = (reference_cost.sum() + proximity_cost)
            cumulative_reward_pg_state[t] = -cost_t
            f_pg_state += cost_t

            if t == N - 1:
                terminal_cost = ((x_t_pg_state - x_desired).T @ Qf_sys @ (x_t_pg_state - x_desired)).sum()
                cumulative_reward_pg_state[t] -= terminal_cost
                f_pg_state += terminal_cost

            cumulative_log_prob_pg_state[t] = log_prob_pg_state.mean()

        for t in range(N):
            Gt = 0
            pw = 0
            for reward in cumulative_reward_pg_state[t:]:
                Gt = Gt + (gamma ** pw) * reward
                pw = pw + 1
            discounted_rewards_pg_state[t] = Gt

        for log_prob, Gt in zip(cumulative_log_prob_pg_state, discounted_rewards_pg_state):
            policy_gradient_pg_state.append(-log_prob * Gt)

        if i == T - 1:
            final_loss = f_pg_state.item()

        # Backprop
        loss = torch.stack(policy_gradient_pg_state).sum()
        loss.backward()
        optimizer_pg_state.step()
        scheduler.step()
    
    return final_loss / n_agents

def main():
    n_seeds = 20
    n_agents = 15
    T = 500
    
    losses = []
    print(f"Starting training for {n_seeds} seeds, {T} iterations each...")
    for seed in tqdm(range(n_seeds)):
        loss = train_one_seed(seed, n_agents=n_agents, T=T)
        losses.append(loss)
    
    losses = np.array(losses)
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    
    print(f"\nResults for {n_seeds} runs:")
    print(f"Mean loss per agent: {mean_loss:.4f}")
    print(f"Std loss per agent: {std_loss:.4f}")
    print(f"Individual losses: {losses}")

if __name__ == "__main__":
    main()
