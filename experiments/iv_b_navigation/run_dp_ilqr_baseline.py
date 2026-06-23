import numpy as np
import matplotlib.pyplot as plt

from dpilqr import split_agents, plot_solve
import dpilqr

np.random.seed(0)

N_STATES = 4
N_CONTROLS = 2
N_AGENTS = 7
MIN_DISTANCE_0 = 2.0
DIM_SPACE = 2
DT = 0.05
MAX_STEPS = 60
COMMUNICATION_RADIUS = 0.5
TOLERANCE_ILQR = 1e-6
IDS = [100 + i for i in range(N_AGENTS)]


x0, xf = dpilqr.random_setup(
    N_AGENTS,
    N_STATES,
    is_rotation=False,
    rel_dist=MIN_DISTANCE_0,
    var=N_AGENTS / 2,
    n_d=DIM_SPACE,
    random=True,
)

model = dpilqr.UnicycleDynamics4D
dynamics = dpilqr.MultiDynamicalModel([model(DT, id_) for id_ in IDS])

Q_cost = np.eye(N_STATES)
R_cost = np.eye(N_CONTROLS)
Qf_cost = 1e3 * np.eye(N_STATES)

goal_costs = [
    dpilqr.ReferenceCost(xf_i, Q_cost.copy(), R_cost.copy(), Qf_cost.copy(), id_)
    for xf_i, id_ in zip(split_agents(xf.T, [N_STATES] * N_AGENTS), IDS)
]
prox_cost = dpilqr.ProximityCost([N_STATES] * N_AGENTS, COMMUNICATION_RADIUS, [DIM_SPACE] * N_AGENTS)
game_cost = dpilqr.GameCost(goal_costs, prox_cost)

problem = dpilqr.ilqrProblem(dynamics, game_cost)
solver = dpilqr.ilqrSolver(problem, MAX_STEPS)

X, _, J = solver.solve(x0, tol=TOLERANCE_ILQR, t_kill=None)


plt.clf()
plot_solve(X, J, xf.T, [N_STATES] * N_AGENTS, True, [DIM_SPACE] * N_AGENTS)

plt.figure()
dpilqr.plot_pairwise_distances(X, [N_STATES] * N_AGENTS, [DIM_SPACE] * N_AGENTS, COMMUNICATION_RADIUS)

plt.show()

dpilqr.make_trajectory_gif(f"{N_AGENTS}-unicycles.gif", X, xf, [N_STATES] * N_AGENTS, COMMUNICATION_RADIUS)
