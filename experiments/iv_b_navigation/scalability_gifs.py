#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from tqdm import tqdm

from dpilqr import split_agents, plot_solve, make_trajectory_gif
import dpilqr
from functions_batch import SelfAttentionNonlinearPolicy

def test_policy_generalization(model_path, n_agents_list=[3, 7, 10]):
    # Common parameters
    n_states = 4
    n_controls = 2
    radius = 0.5
    dt = 0.05
    N = 100
    device = "cpu"

    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train first.")
        return

    # Load weights from the trained model (expected to be trained on 7 agents)
    print(f"Loading weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    # The saved model might be a DDPG agent or just a state_dict
    if 'actor' in checkpoint:
        pretrained_dict = checkpoint['actor']
    else:
        pretrained_dict = checkpoint

    for n_agents in n_agents_list:
        print(f"\n--- Testing with {n_agents} agents ---")
        
        x_dims = [n_states] * n_agents
        u_dims = [n_controls] * n_agents
        n_dims = [2] * n_agents

        # Setup random scenario
        x0, xf = dpilqr.random_setup(
            n_agents,
            n_states,
            is_rotation=False,
            rel_dist=2.0,
            var=n_agents / 2,
            n_d=2,
            random=True,
        )

        # Initialize policy with the NEW number of agents
        # The key is that AttentionGain's weights don't depend on n_agents
        policy = SelfAttentionNonlinearPolicy(n_states, n_controls, n_agents, [64, 64], radius, device)
        
        # Load the pretrained weights
        # We might need to filter or adjust if there are exact shape mismatches 
        # but GNN layers usually match if state_dim_per_agent matches.
        try:
            policy.load_state_dict(pretrained_dict)
            print("Successfully loaded policy weights.")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Attempting to load with strict=False...")
            policy.load_state_dict(pretrained_dict, strict=False)

        policy.eval()

        # Run simulation
        X_ours = torch.zeros(N + 1, n_states * n_agents, device=device)
        x_t = torch.from_numpy(x0).to(device).to(torch.float32)
        X_ours[0, :] = x_t[:, 0]
        
        # Dynamics setup
        B_k = [torch.Tensor([[0., 0.], [0., 0.], [1., 0.], [0., 1.]]) for _ in range(n_agents)]
        B_t = torch.block_diag(*B_k).to(device)

        for t in range(N):
            # Format inputs for policy
            obs = x_t.clone().reshape(1, -1)
            pos = x_t.clone().reshape(1, n_agents, n_states)[:, :, :2]
            
            with torch.no_grad():
                action_dist = policy(obs, pos)
                u_t = action_dist.mean.reshape(-1, 1)

            # Simple Euler integration for unicycle
            A_t = torch.zeros(n_states * n_agents, n_states * n_agents, device=device)
            for k in range(n_agents):
                # Using the same linearized dynamics as in case_2_d_ilqr.py for simplicity
                A_t[n_states * k, n_states * k + 2] = torch.cos(x_t[4 * k + 3, 0])
                A_t[n_states * k, n_states * k + 3] = -x_t[4 * k + 2, 0] * torch.sin(x_t[4 * k + 3, 0])
                A_t[n_states * k + 1, n_states * k + 2] = torch.sin(x_t[4 * k + 3, 0])
                A_t[n_states * k + 1, n_states * k + 3] = x_t[4 * k + 2, 0] * torch.cos(x_t[4 * k + 3, 0])

            x_t = (A_t @ x_t + B_t @ u_t) * dt + x_t
            X_ours[t + 1, :] = x_t[:, 0]

        # Results
        X_ours_np = X_ours.detach().cpu().numpy()
        gif_name = f"generalization_{n_agents}_agents.gif"
        make_trajectory_gif(gif_name, X_ours_np, xf, x_dims, radius)
        print(f"Saved trajectory GIF to {gif_name}")

        # Plot final distances to goal
        final_pos = X_ours_np[-1].reshape(n_agents, n_states)[:, :2]
        goal_pos = xf.reshape(n_agents, n_states)[:, :2]
        dist = np.linalg.norm(final_pos - goal_pos, axis=1)
        print(f"Average distance to goal: {np.mean(dist):.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Roll out a trained 7-agent policy on {4, 8, 10, 15} agents "
                    "to reproduce the scalability GIFs of Fig. 5."
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="videos/7-unicycles_policy_9900.pth",
        help="Path to a .pth checkpoint saved by train_ours_navigation.py "
             "(default: videos/7-unicycles_policy_9900.pth — the last "
             "checkpoint of a full 10000-iter run).",
    )
    args = parser.parse_args()
    if not os.path.exists(args.checkpoint):
        print(f"No checkpoint at {args.checkpoint}. Train first with "
              f"train_ours_navigation.py or pass --checkpoint.")
    else:
        test_policy_generalization(args.checkpoint, n_agents_list=[4, 8, 10, 15])
