#Import Robotarium Utilities
import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.graph import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *
import matplotlib.patches as patches
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import torch
import numpy as np
from functions import SelfAttentionNonlinearPolicy

# Initial setup
device = "cpu" 
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Change parameters here for evaluation
num_agents_team1 = 3
num_agents_team2 = 3
num_agents_eval = num_agents_team1 + num_agents_team2
num_obstacles = 2

# Load the data from the .npy file
# allow_pickle=True is required because we saved a dictionary (an object)
loaded_weights = np.load('actor_weights.npy', allow_pickle=True).item()

# Now you can access the weights like a normal dictionary
team1_actor_weights = loaded_weights['team1']
team2_actor_weights = loaded_weights['team2']


# The observation for each agent is a vector of size 22.
observation_dim_per_agent_team1 = 22
observation_dim_per_agent_team2 = 20
action_dim_per_agent = 4

actor_net_weights_pytorch1 = {k: torch.from_numpy(v) for k, v in team1_actor_weights.items()}
actor_net1 = SelfAttentionNonlinearPolicy(observation_dim_per_agent_team1, 
                                          action_dim_per_agent, 
                                          num_agents_team1, 
                                          hidden_channels_dim=[64, 64], 
                                          r_communication=0.5, 
                                          device=device)
actor_net1.load_state_dict(actor_net_weights_pytorch1)
actor_net1.eval()

actor_net_weights_pytorch2 = {k: torch.from_numpy(v) for k, v in team2_actor_weights.items()}
actor_net2 = SelfAttentionNonlinearPolicy(observation_dim_per_agent_team2, 
                                          action_dim_per_agent, 
                                          num_agents_team2, 
                                          hidden_channels_dim=[64, 64], 
                                          r_communication=0.5, 
                                          device=device)
actor_net2.load_state_dict(actor_net_weights_pytorch2)
actor_net2.eval()

# Experiment constants
iterations = 900
magnitude_limit = 0.15
dt = 0.033

# Robotarium setup
r = robotarium.Robotarium(number_of_robots=num_agents_eval, show_figure=True, sim_in_real_time=False)
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary(safety_radius=0.13)
si_to_uni_dyn = create_si_to_uni_dynamics()

# Logging
num_catches = 0
catch_flags = np.zeros(num_agents_team2)
threshold = 0.2 # Distance threshold to consider a "catch"
travel_distances = 1000 * np.ones([num_agents_eval * num_agents_eval, iterations])
min_distances = 0
max_distances = 0
traces = []

# Obstacles
pos_obstacles = (np.random.rand(2, num_obstacles) - np.array([[0.5], [0.5]])) * np.array([[2.2], [1.0]])
radius_obstacles = 0.2 * np.ones(num_obstacles)

goal_points = generate_initial_conditions(num_agents_eval)
goal_points[0:2, 0] = np.array([-1.0, -0.5])
goal_points[0:2, 1] = np.array([-0.8, -0.6])
goal_points[0:2, 2] = np.array([-1.2, -0.3])

goal_points[0:2, 3] = np.array([1.0, 0.5])
goal_points[0:2, 4] = np.array([0.8, 0.6])
goal_points[0:2, 5] = np.array([1.2, 0.3]) # Set initial position of agent 1

# Create single integrator position controller
single_integrator_position_controller = create_si_position_controller()

_, uni_to_si_states = create_si_to_uni_mapping()


x = r.get_poses()
x_si = uni_to_si_states(x)
r.step()

while (np.size(at_position(x, goal_points[:2, :], position_error=0.1)) != num_agents_eval):

    # Get poses of agents
    x = r.get_poses()
    x_si = uni_to_si_states(x)

    # Create single-integrator control inputs
    dxi = single_integrator_position_controller(x_si, goal_points[:2][:])

    # Create safe control inputs (i.e., no collisions)
    dxi = si_barrier_cert(dxi, x_si)

    # Transform single integrator velocity commands to unicycle
    dxu = si_to_uni_dyn(dxi, x)

    # Set the velocities by mapping the single-integrator inputs to unciycle inputs
    r.set_velocities(np.arange(num_agents_eval), dxu)
    # Iterate the simulation
    r.step()


with torch.no_grad():
    lines = []

    X = np.zeros((2, num_agents_eval))
    V = np.zeros((2, num_agents_eval))

    for k in range(iterations):

        # Get the poses of the robots
        x = r.get_poses()

        # Update the observation matrix
        X = x[:2, :]
        traces.append(np.copy(x[:2, :]))
        
        # Create observation vector for each agent based on the specified rules
        obs1 = np.zeros((observation_dim_per_agent_team1, num_agents_team1))
        obs2 = np.zeros((observation_dim_per_agent_team2, num_agents_team2))
        for i in range(num_agents_eval):
            # 1. Self velocity (2 dims)
            self_vel = V[:, i]
            
            # 2. Self position (2 dims)
            self_pos = X[:, i]
            
            # 3. Relative position to obstacles (2 * num_obstacles dims)
            # Reshape self_pos to allow broadcasting
            rel_obstacle_pos = (pos_obstacles - self_pos.reshape(2, 1)).flatten()
            
            # 4. Relative positions of other agents ((num_agents_eval - 1) * 2 dims)
            other_agent_indices = np.delete(np.arange(num_agents_eval), i)
            other_agent_pos = X[:, other_agent_indices]
            rel_other_pos = (other_agent_pos - self_pos.reshape(2, 1)).flatten()

            # 5. Velocities of other agents on the same team
            if i < num_agents_team1: # Agent i is in Team 1
                # Teammate indices are the other agents in Team 1
                teammate_indices = np.delete(np.arange(num_agents_team1), i)
            else: # Agent i is in Team 2
                # Team 2 indices are from num_agents_team1 to num_agents_eval - 1
                team2_indices_all = np.arange(num_agents_team1, num_agents_eval)
                # Find the index of i within the team2_indices_all array to delete it
                i_in_team2 = i - num_agents_team1
                teammate_indices = np.delete(team2_indices_all, i_in_team2)
            
            # Flatten the array of teammate velocities
            teammate_vel = V[:, teammate_indices].flatten()
            
            if i < num_agents_team1:
                obs_i = np.concatenate([
                    self_vel,
                    self_pos,
                    rel_obstacle_pos,
                    rel_other_pos,
                    # -1 * np.ones(4), # Padding to make the observation size consistent
                    teammate_vel,
                    # -1 * np.ones(2)
                ])
                obs1[:, i] = obs_i
            else:
                obs_i = np.concatenate([
                    # self_vel,
                    self_pos,
                    rel_obstacle_pos,
                    rel_other_pos,
                    # -1 * np.ones(4), # Padding to make the observation size consistent
                    teammate_vel,
                    # -1 * np.ones(2)
                ])
                obs2[:, i - num_agents_team1] = obs_i
            

        # Update graphics
        while len(lines) > 0:
            lines.pop(0).remove()

        # Plot desired positions
        trace = np.array(traces)
        # for i in range(num_obstacles):
        #     # Use a Circle patch to accurately represent obstacle radius
        #     obstacle_patch = patches.Circle((pos_obstacles[0, i], pos_obstacles[1, i]), radius=radius_obstacles[i], color='k', zorder=0)
        #     r.axes.add_patch(obstacle_patch)
        #     lines.append(obstacle_patch)
        for i in range(num_agents_eval):
            if i < num_agents_team1:
                lines.append(plt.plot(trace[:, 0, i], trace[:, 1, i], linewidth=3, color='r', zorder=0)[0])
                adv_patch = patches.Circle((trace[k, 0, i], trace[k, 1, i]), radius=0.15, color='r', zorder=0)
                r.axes.add_patch(adv_patch)
                lines.append(adv_patch)
            else:
                lines.append(plt.plot(trace[:, 0, i], trace[:, 1, i], linewidth=3, color='g', zorder=0)[0])
                age_patch = patches.Circle((trace[k, 0, i], trace[k, 1, i]), radius=0.15, color='g', zorder=0)
                r.axes.add_patch(age_patch)
                lines.append(age_patch)
        
        plt.gcf().canvas.flush_events()

        # Get actions from the policy
        actions1 = actor_net1(torch.from_numpy(obs1).T.unsqueeze(0).float())
        actions2 = actor_net2(torch.from_numpy(obs2).T.unsqueeze(0).float())
        
        eval_actions = torch.cat((actions1, actions2), dim=1).squeeze(0).numpy().T

        # Update velocity vector using the calculated 2D vector
        V += eval_actions * dt

        #Keep single integrator control vectors under specified magnitude
        norms = np.linalg.norm(V, 2, 0)
        idxs_to_normalize = (norms > magnitude_limit)
        V[:, idxs_to_normalize] *= magnitude_limit/norms[idxs_to_normalize]

        # Make sure that the robots don't collide
        V = si_barrier_cert(V, x[:2, :])

        # Logg info
        for i in range(num_agents_eval):
            for j in range(num_agents_eval):
                if i != j:
                    travel_distances[i * num_agents_eval + j, k] = np.linalg.norm(X[:, i] - X[:, j])
                    if np.linalg.norm(X[:, i] - X[:, j]) < threshold and k > 10 and i < num_agents_team1 and j >= num_agents_team1 and catch_flags[j - num_agents_team1] == 0:
                        num_catches += 1
                        catch_flags[j - num_agents_team1] = 1
                    if k == iterations - 1:
                        if min_distances == 0 or np.min(travel_distances[i * num_agents_eval + j, :]) < min_distances:
                            min_distances = np.min(travel_distances[i * num_agents_eval + j, :])
                        if np.max(travel_distances[i * num_agents_eval + j, :]) > max_distances:
                            max_distances = np.max(travel_distances[i * num_agents_eval + j, :])

        # Transform the single-integrator dynamcis to unicycle dynamics
        dxu = si_to_uni_dyn(V, x)

        # Set the velocities of the robots
        r.set_velocities(np.arange(num_agents_eval), dxu)

        # Iterate the simulation
        r.step()

    r.call_at_scripts_end()

# Report numbers
print("Minimum distance between agents: ", min_distances)
print("Maximum distance between agents: ", max_distances)
print("Number of catches: ", num_catches)
