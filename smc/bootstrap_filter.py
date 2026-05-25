import asyncio
from dataclasses import dataclass, field
import logging
from logging import Logger
import random
from typing import Any, Sequence, List

from smc.agents.agent import Agent
import math


@dataclass
class SMCBootstrapFilter:
    """
    Applies a bootstrap filter for sequential monte carlo sampling.
    """

    agents: Sequence[Agent]  # collection of current particles

    goal_state: Any  # if goal state is reached, we can terminate the filter early.

    time_step: int = 0  # current time step
    resample_steps: List[int] = field(default_factory=list) # resampling schedule
    max_step: int = 10  # maximum number of time steps to run
    weights: list[float] = field(init=False)  # current particle weight
    ess_threshold: float = 0.5
    weight_temperature: float = 1.0  # temperature parameter for weight calculation

    logger: Logger = logging.getLogger()
    _adaptive_mode: bool = field(init=False)

    def __post_init__(self):        
        self.weights = [1 / len(self.agents) for _ in self.agents]
        if not self.resample_steps:
            self._adaptive_mode = True
            self.logger.info(f"Adaptive ESS resampling enabled (Threshold: N * {self.ess_threshold}).")
        else:
            self._adaptive_mode = False
            self.logger.info(f"Manual resampling schedule active: {self.resample_steps}. Adaptive mode is disabled.")

        self.logger.info("Setting initial V_0 (value_at_last_resample) = 0.0 for all particles.")
        for agent in self.agents:
            if hasattr(agent, 'value_at_last_resample'):
                agent.value_at_last_resample = 0.0
            else:
                self.logger.error(f"FATAL: Agent {agent} is missing 'value_at_last_resample' attribute.")

    async def take_steps(self) -> None:
        self.logger.info("Polling agents for new states.")
        steps = [agent.step(agent.state) for agent in self.agents]
        await asyncio.gather(*steps)
        # todo handle exceptions

    async def get_raw_weights(self) -> list[float]:
        self.logger.info("Computing raw weights for new states.")
        weights = [agent.evaluate_state(agent.state) for agent in self.agents]
        weights = await asyncio.gather(*weights)
        # todo handle exceptiosn
        return weights    

    def compute_next_step(self) -> bool:
        """
        Computes the next step in the bootstrap filter synchronously.
        Returns true if we should compute the next step, otherwise return false
        """

        asyncio.run(self.take_steps())
        self.time_step += 1
        self.logger.info(f"Advanced to step {self.time_step}.")

        if any([self.goal_state == a.state for a in self.agents]):
            self.logger.info("Goal state reached. Terminating early")
            return False

        is_manual_resample_step = self.time_step in self.resample_steps
        should_check_weights = self._adaptive_mode or is_manual_resample_step

        resample_triggered = False

        if should_check_weights and self.time_step < self.max_step:
            self.logger.info(f"Step {self.time_step}: Re-weighting check required.")

            v_current_list = asyncio.run(self.get_raw_weights())
            self.logger.info(f"Current values (V_t) received: {v_current_list}")

            ## This weight calculation assumes the sparse reward happening only at the end of the episode.

            raw_weights = []

            for v_curr, agent in zip(v_current_list, self.agents):    
                ### debug:
                self.logger.info(f"Computing weight for agent: {agent}, cumulative_reward: {agent.cumulative_reward}, reward_at_last_resample: {agent.reward_at_last_resample}, value_at_last_resample: {agent.value_at_last_resample}, value_current: {v_curr}, terminal: {agent.terminal}, terminal_updated: {agent.terminal_updated})")
                
                reward_from_last_resample = agent.cumulative_reward - agent.reward_at_last_resample        
                if agent.terminal == False:
                    agent_raw_weight = (reward_from_last_resample + v_curr - agent.value_at_last_resample) / self.weight_temperature
                else:
                    if agent.terminal_updated == True:
                        self.logger.info(f"No more weight update for terminated agent: {agent}")
                        agent_raw_weight = 0.0 
                    else:
                        self.logger.info(f"Final weight update for terminated agent: {agent}")
                        agent_raw_weight = (reward_from_last_resample - agent.value_at_last_resample) / self.weight_temperature        
                        agent.terminal_updated = True
                raw_weights.append(math.exp(agent_raw_weight))


            self.logger.info(f"Raw weights (exponentiated, non-normalized): {raw_weights}")
            total_weight = sum(raw_weights)
            if total_weight > 0:
                current_normalized_weights = [w / total_weight for w in raw_weights]
            else:
                current_normalized_weights = [1 / len(self.agents) for _ in self.agents]
            
            self.weights = current_normalized_weights 

            if self._adaptive_mode:
                sum_of_squares = sum(w**2 for w in current_normalized_weights)
                current_ess = 1.0 / sum_of_squares if sum_of_squares > 0 else 0.0
                self.logger.info(f"Current ESS: {current_ess:.2f}")
                
                ess_limit = len(self.agents) * self.ess_threshold
                if current_ess < ess_limit:
                    resample_triggered = True
                    self.logger.info(f"Resampling triggered: ESS ({current_ess:.2f}) < Threshold ({ess_limit:.2f})")
                
            elif is_manual_resample_step:
                resample_triggered = True
                self.logger.info(f"Resampling triggered by manual schedule at step {self.time_step}.")

            self.logger.info("Updating base value memory (V_resample = V_t) for next step's incremental weight.")
            for agent, new_value in zip(self.agents, v_current_list):
                    agent.value_at_last_resample = new_value
                    agent.reward_at_last_resample = agent.cumulative_reward

            if resample_triggered:

                self.logger.info("Performing resampling...")
                self.logger.info(f"Normalized weights used: {[f'{w:.4f}' for w in self.weights]}")

                total_agents = len(self.agents)
                selected = random.choices(self.agents, self.weights, k=total_agents)
                self.agents = [agent.clone() for agent in selected]
                self.logger.info("Agents resampled.")
            
            elif self._adaptive_mode:
                self.logger.info(f"Step {self.time_step}: No resampling (Adaptive Mode).")
        else:
            self.logger.info(f"Step {self.time_step}: Manual mode, skipping value function call to save cost.")


        return True
    