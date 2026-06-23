#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import torch
import os

from dpilqr import split_agents, plot_solve
import dpilqr
from functions_batch import SelfAttentionNonlinearPolicy

def evaluate_checkpoint(checkpoint_path):
    # 1. DIFFERENT CONFIGURATION SETUP
    np.random.seed(42) # Different seed than training
    torch.manual_seed(42)
    device = "cpu"

    n_states = 4
    n_controls = 2
    n_agents = 7 
    x_dims = [n_states] * n_agents
    n_dims = [2] * n_agents
    n_d = n_dims[0]

    print("Setting up a challenging circle-swap configuration...")
    # Changed to a structured circle swap with a larger relative distance
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

    # 2. LOAD THE POLICY
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    neural_network_pg_state = SelfAttentionNonlinearPolicy(n_states, n_controls, n_agents, [64, 64], radius, device)
    
    # Load the weights and set to eval mode (crucial for disabling dropout/batchnorm if used)
    neural_network_pg_state.load_state_dict(torch.load(checkpoint_path, map_location=device))
    neural_network_pg_state.eval() 

    # 3. SETUP DYNAMICS MATRICES
    B_k = [torch.Tensor([[0., 0.],
                         [0., 0.],
                         [1., 0.],
                         [0., 1.]
                         ]) for k in range(n_agents)]
    B_t = torch.block_diag(*B_k).to(device)
    A_t = torch.zeros(n_states * n_agents, n_states * n_agents, device=device)

    X_ours = torch.zeros(N + 1, n_states * n_agents, device=device)
    x_t_pg_state = torch.from_numpy(x0).to(device).to(torch.float32)
    x_desired = torch.from_numpy(xf).to(device).to(torch.float32)
    
    X_ours[0, :] = x_t_pg_state[:, 0]

    # 4. SIMULATE TRAJECTORY
    print("Simulating trajectory with the loaded policy...")
    with torch.no_grad(): # Disable gradients for faster, memory-efficient evaluation
        for t in range(N):
            action_distribution_pg_state = neural_network_pg_state(
                x_t_pg_state.clone().reshape(1, -1) - x_desired.clone().reshape(1, -1), 
                x_t_pg_state.clone().reshape(1, n_agents, n_states)[:, :, :2]
            )
            
            # Use the mean of the distribution for deterministic evaluation acting
            u_t_pg_state = action_distribution_pg_state.mean
            u_t_pg_state = u_t_pg_state.reshape(-1, 1)

            # Compute next state based on unicycle dynamics
            for k in range(n_agents):
                A_t[n_states * k, n_states * k + 2] = torch.cos(x_t_pg_state[4 * k + 3, 0])
                A_t[n_states * k, n_states * k + 3] = -x_t_pg_state[4 * k + 2, 0] * torch.sin(x_t_pg_state[4 * k + 3, 0])
                A_t[n_states * k + 1, n_states * k + 2] = torch.sin(x_t_pg_state[4 * k + 3, 0])
                A_t[n_states * k + 1, n_states * k + 3] = x_t_pg_state[4 * k + 2, 0] * torch.cos(x_t_pg_state[4 * k + 3, 0])

            # Forward Euler step (matches your training setup)
            x_t_pg_state = (A_t @ x_t_pg_state + B_t @ u_t_pg_state) * dt + x_t_pg_state

            X_ours[t + 1, :] = x_t_pg_state[:, 0]

    X_ours_np = X_ours.detach().cpu().numpy()

    # 5. VISUALIZE RESULTS
    print("Generating plots and saving GIF...")
    
    # Plot Trajectories
    plt.clf()
    plot_solve(X_ours_np, 0.0, xf.T, x_dims, True, n_d)
    plt.title("Evaluation: Circle Swap Trajectories")
    plt.show()

    # Plot Distances (to check for collisions)
    plt.figure()
    dpilqr.plot_pairwise_distances(X_ours_np, x_dims, n_dims, radius)
    plt.title("Evaluation: Pairwise Distances")
    plt.show()

    # Save output video
    gif_name = "evaluation_circle_swap.gif"
    dpilqr.make_trajectory_gif(gif_name, X_ours_np, xf, x_dims, radius)
    print(f"Saved evaluation GIF to {gif_name}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a trained navigation policy on a fresh random configuration.")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="videos/7-unicycles_policy_300.pth",
        help="Path to a .pth checkpoint saved by train_ours_navigation.py "
             "(default: videos/7-unicycles_policy_300.pth)",
    )
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint)