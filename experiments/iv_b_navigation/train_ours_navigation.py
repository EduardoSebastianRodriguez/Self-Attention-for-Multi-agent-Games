#!/usr/bin/env python

import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from dpilqr import split_agents, plot_solve
import dpilqr
from functions_batch import SelfAttentionNonlinearPolicy

os.makedirs("videos", exist_ok=True)

np.random.seed(0)
torch.manual_seed(0)
device = "cpu" #if not torch.cuda.is_available() else "cuda:0"


def random_multiagent_simulation():
    n_states = 4
    n_controls = 2
    n_agents = 7
    x_dims = [n_states] * n_agents
    u_dims = [n_controls] * n_agents
    n_dims = [2] * n_agents

    n_d = n_dims[0]

    # Generate the initial configuration
    x0, xf = dpilqr.random_setup(
        n_agents,
        n_states,
        is_rotation=False,
        rel_dist=2.0,
        var=n_agents / 2,
        n_d=2,
        random=True,
    )
    
    # Store the initial configuration for the final iLQR baseline comparison
    x0_eval, xf_eval = x0.copy(), xf.copy()

    dt = 0.05
    N = 100
    radius = 0.5

    tol = 1e-6
    ids = [100 + i for i in range(n_agents)]

    T = 10_000
    lr = 1e-3

    gamma = 1.0
    neural_network_pg_state = SelfAttentionNonlinearPolicy(n_states, n_controls, n_agents, [64, 64], radius, device)
    neural_network_pg_state.train()
    optimizer_pg_state = torch.optim.Adam(neural_network_pg_state.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_pg_state, gamma=0.9999)

    log_loss_pg_state = np.zeros(T)

    model = dpilqr.UnicycleDynamics4D
    dynamics = dpilqr.MultiDynamicalModel([model(dt, id_) for id_ in ids])

    Q = np.eye(n_states) 
    R = np.eye(n_controls)
    Qf = 100 * np.eye(n_states)

    Q_torch = [torch.from_numpy(Q) for _ in range(n_agents)]
    R_torch = [torch.from_numpy(R) for _ in range(n_agents)]
    Qf_torch = [torch.from_numpy(Qf) for _ in range(n_agents)]

    Q_sys = torch.block_diag(*Q_torch).to(device).to(torch.float32)
    R_sys = torch.block_diag(*R_torch).to(device).to(torch.float32)
    Qf_sys = torch.block_diag(*Qf_torch).to(device).to(torch.float32)

    # Solve iLQR baseline ONLY for the initial configuration
    goal_costs = [
        dpilqr.ReferenceCost(xf_i, Q.copy(), R.copy(), Qf.copy(), id_)
        for xf_i, id_ in zip(split_agents(xf.T, x_dims), ids)
    ]
    prox_cost = dpilqr.ProximityCost(x_dims, radius, n_dims)
    game_cost = dpilqr.GameCost(goal_costs, prox_cost)

    problem = dpilqr.ilqrProblem(dynamics, game_cost)
    solver = dpilqr.ilqrSolver(problem, N)

    X, _, J, Js = solver.solve(x0, tol=tol, t_kill=None)

    B_k = [torch.Tensor([[0., 0.],
                         [0., 0.],
                         [1., 0.],
                         [0., 1.]
                         ]) for k in range(n_agents)]
    B_t = torch.block_diag(*B_k).to(device)

    for i in tqdm(range(T)):
        # --- NEW: Resample the configuration every 100 steps ---
        if i % 100 == 0 and i > 0:
            x0, xf = dpilqr.random_setup(
                n_agents,
                n_states,
                is_rotation=False,
                rel_dist=2.0,
                var=n_agents / 2,
                n_d=2,
                random=True,
            )

        optimizer_pg_state.zero_grad()
        f_pg_state = torch.Tensor([0.0]).to(device)
        cumulative_reward_pg_state = torch.zeros(N, device=device)
        cumulative_log_prob_pg_state = torch.zeros(N, device=device)
        discounted_rewards_pg_state = torch.zeros(N, device=device)
        policy_gradient_pg_state = []

        x_t_pg_state = torch.from_numpy(x0).to(device).to(torch.float32)
        x_desired = torch.from_numpy(xf).to(device).to(torch.float32)

        X_ours = torch.zeros(N + 1, n_states * n_agents, device=device)
        X_ours[0, :] = x_t_pg_state[:, 0]
        for t in range(N):
            action_distribution_pg_state = neural_network_pg_state(x_t_pg_state.clone().reshape(1, -1) - x_desired.clone().reshape(1, -1), x_t_pg_state.clone().reshape(1, n_agents, n_states)[:, :, :2])
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

            X_ours[t + 1, :] = x_t_pg_state[:, 0]

            # Compute current sparse cost matrices
            reference_cost = (x_t_pg_state - x_desired).T @ Q_sys @ (x_t_pg_state - x_desired) + u_t_pg_state.T @ R_sys @ u_t_pg_state

            Q1 = x_t_pg_state.reshape(n_agents, n_states)[:, :2].repeat(n_agents, 1)
            Q2 = torch.kron(x_t_pg_state.reshape(n_agents, n_states)[:, :2], torch.ones((n_agents, 1), device=device))
            q = (Q1 - Q2).norm(p=2, dim=1).reshape(n_agents, n_agents)
            distance_costs = (torch.min(torch.zeros_like(q), q - radius) ** 2).fill_diagonal_(0.)
            proximity_cost = distance_costs.sum()

            cumulative_reward_pg_state[t] = -(reference_cost.sum() + proximity_cost)
            f_pg_state += (reference_cost.sum() + proximity_cost)

            if t == N - 1:
                cumulative_reward_pg_state[t] -= ((x_t_pg_state - x_desired).T @ Qf_sys @ (x_t_pg_state - x_desired)).sum()
                f_pg_state += ((x_t_pg_state - x_desired).T @ Qf_sys @ (x_t_pg_state - x_desired)).sum()

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

        # Store cost at iteration i
        log_loss_pg_state[i] = f_pg_state.detach().cpu().numpy()[0]

        if i % 10 == 0:
            print('-------------------------------------')
            print('\n iteration {} with loss {}'.format(i, log_loss_pg_state[i]))
            print('\n-------------------------------------')
        
        # Save checkpoints and GIFs
        if i % 100 == 0:
            dpilqr.make_trajectory_gif(f"videos/{n_agents}-unicycles_ours_{i}.gif", X_ours.detach().cpu().numpy(), xf, x_dims, radius)
            torch.save(neural_network_pg_state.state_dict(), f"videos/{n_agents}-unicycles_policy_{i}.pth")

        # Backprop
        loss = torch.stack(policy_gradient_pg_state).sum()
        loss.backward()
        optimizer_pg_state.step()
        scheduler.step()

    # --- RESTORE ORIGINAL CONFIGURATION FOR EVALUATION ---
    x0, xf = x0_eval, xf_eval
    
    neural_network_pg_state.eval()
    
    X_ours = torch.zeros(N + 1, n_states * n_agents, device=device)
    x_t_pg_state = torch.from_numpy(x0).to(device).to(torch.float32)
    x_desired = torch.from_numpy(xf).to(device).to(torch.float32)
    X_ours[0, :] = x_t_pg_state[:, 0]
    
    for t in range(N):
        action_distribution_pg_state = neural_network_pg_state(x_t_pg_state.clone().reshape(1, -1) - x_desired.clone().reshape(1, -1), x_t_pg_state.clone().reshape(1, n_agents, n_states)[:, :, :2])
        u_t_pg_state = action_distribution_pg_state.mean
        u_t_pg_state = u_t_pg_state.reshape(-1, 1)

        # Compute next state
        for k in range(n_agents):
            A_t[n_states * k, n_states * k + 2] = torch.cos(x_t_pg_state[4 * k + 3, 0])
            A_t[n_states * k, n_states * k + 3] = -x_t_pg_state[4 * k + 2, 0] * torch.sin(x_t_pg_state[4 * k + 3, 0])
            A_t[n_states * k + 1, n_states * k + 2] = torch.sin(x_t_pg_state[4 * k + 3, 0])
            A_t[n_states * k + 1, n_states * k + 3] = x_t_pg_state[4 * k + 2, 0] * torch.cos(x_t_pg_state[4 * k + 3, 0])

        x_t_pg_state = A_t @ x_t_pg_state + B_t @ u_t_pg_state

        X_ours[t + 1, :] = x_t_pg_state[:, 0]

    X_ours = X_ours.detach().cpu().numpy()

    plt.clf()
    plot_solve(X_ours, 0.0, xf.T, x_dims, True, n_d)

    plt.figure()
    dpilqr.plot_pairwise_distances(X_ours, x_dims, n_dims, radius)

    plt.show()

    plt.clf()
    plot_solve(X, J, xf.T, x_dims, True, n_d)

    plt.figure()
    dpilqr.plot_pairwise_distances(X, x_dims, n_dims, radius)

    plt.show()

    dpilqr.make_trajectory_gif(f"{n_agents}-unicycles.gif", X, xf, x_dims, radius)
    dpilqr.make_trajectory_gif(f"{n_agents}-unicycles_ours.gif", X_ours, xf, x_dims, radius)

    np.save("log_loss_pg_state", log_loss_pg_state)
    np.save("Js", Js)


def main():
    random_multiagent_simulation()


if __name__ == "__main__":
    main()