#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING

# additional imports
from vmas.simulator.core import Agent
from vmas.scenarios.mpe.simple_tag import Scenario
from vmas import render_interactively
import torch


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

# modifying Scenario
class SimpleTagTimeScenario(Scenario):
    def info(self, agent:Agent):
        '''
            Defines an info function overwriting the one in the VMAS BaseScenario class (vmas.simulator.scenario) 
            "catch": indicates if the agent is colliding with at least one member of opposing team
            
            Returns: Dict[str, torch.Tensor]
        '''

        # agent_catch = torch.tensor([False], device=self.world.device)
        agent_catch = ~self.is_collision(agent, agent)
        
        for other in self.world.agents:
            # ignore collisions if on same team
            if agent.adversary == other.adversary:
                continue
            
            agent_catch = agent_catch | self.is_collision(agent, other)
        
        return {"catch":agent_catch}


if __name__ == "__main__":
    render_interactively(SimpleTagTimeScenario(), control_two_agents=True)