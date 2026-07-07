#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING

import torch
from vmas import render_interactively
from vmas.simulator.core import Agent


@dataclass
class TaskConfig:
    max_steps: int = MISSING
    num_good_agents: int = MISSING
    num_adversaries: int = MISSING
    num_landmarks: int = MISSING
    shape_agent_rew: bool = MISSING
    shape_adversary_rew: bool = MISSING
    agents_share_rew: bool = MISSING
    adversaries_share_rew: bool = MISSING
    observe_same_team: bool = MISSING
    observe_pos: bool = MISSING
    observe_vel: bool = MISSING
    bound: float = MISSING
    respawn_at_catch: bool = MISSING

from vmas.scenarios.mpe.simple_tag import Scenario


def info(self, agent:Agent):
    '''
        Defines an info function overwriting the one in the VMAS BaseScenario class (vmas.simulator.scenario)
        "catch": indicates if the agent is colliding with at least one member of opposing team
        
        Returns: Dict[str : torch.Tensor]
    '''

    # initialize as logical False, since an agent is always colliding with itself
    agent_catch = ~self.is_collision(agent, agent)
    
    for other in self.world.agents:
        # ignore collisions if on same team
        if agent.adversary == other.adversary:
            continue
        
        agent_catch = agent_catch | self.is_collision(agent, other)
    
    return {"catch":agent_catch}


def observation(self, agent: Agent):
    # get positions of all entities in this agent's reference frame
    entity_pos = []
    for entity in self.world.landmarks:
        entity_pos.append(entity.state.pos - agent.state.pos)

    other_pos = []
    other_vel = []
    for other in self.world.agents:
        if other is agent:
            continue
        if agent.adversary and not other.adversary:
            other_pos.append(other.state.pos - agent.state.pos)
            other_vel.append(other.state.vel)
        elif not agent.adversary and not other.adversary and self.observe_same_team:
            other_pos.append(other.state.pos - agent.state.pos)
            other_vel.append(other.state.vel)
        elif not agent.adversary and other.adversary:
            other_pos.append(other.state.pos - agent.state.pos)
        elif agent.adversary and other.adversary and self.observe_same_team:
            other_pos.append(other.state.pos - agent.state.pos)

    return {
        "obs": torch.cat(
            [
                *([agent.state.vel] if self.observe_vel else []),
                *([agent.state.pos] if self.observe_pos else []),
                *entity_pos,
                *other_pos,
                *other_vel,
            ],
            dim=-1,
        ),
        "pos": agent.state.pos,
        "vel": agent.state.vel
    }

Scenario.info = info
Scenario.observation = observation
SimpleTagScenario = Scenario

if __name__ == "__main__":
    render_interactively(
        SimpleTagScenario(),
        control_two_agents=True,
    )
