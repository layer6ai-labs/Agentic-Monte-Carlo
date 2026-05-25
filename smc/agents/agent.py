import abc
import asyncio
import copy
import logging
from typing import Any, List

from smc.clients import AgentGymClient
from smc.utils.action_utils import NO_ACTION, ActionParserError, parse_action

# Assume 'State' is a defined type or class
# from your_state_module import State


class Agent(abc.ABC):
    def __init__(self) -> None:
        """
        Abstract class for agent
        """
        super()

    @abc.abstractmethod
    async def step(self, state: Any) -> None:
        """
        Asynchronously executes a step based on the policy function.
        Updates its own internal state
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def evaluate_state(self, state: Any) -> float:
        """
        Evaluates the state, akin to value function in RL.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def state(self) -> Any:
        """Returns state"""
        raise NotImplementedError

    @abc.abstractmethod
    def clone(self):
        """
        Clones the current Agent, but to a new agent
        """
        raise NotImplementedError

class AgentGymAgent(Agent):
    def __init__(self, client: AgentGymClient, task_idx, system_prompt, model, value_function) -> None:
        super().__init__()
        self.client = client
        self.task_idx = task_idx
        self.system_prompt = system_prompt
        self.model = model
        
        logger_name = f"AgentGymAgent.{id(self)}"
        self.logger = logging.getLogger(logger_name)
        
        # track actions for value function and replay during cloning
        self.action_history: List[str] = []  
        self.value_function = value_function        
        self.cumulative_reward: float = 0.0
        self.terminal: bool = False
        self.reward_at_last_resample: float = 0.0
        self.value_at_last_resample: float = 0.0
        self.terminal_updated = False   

        self.logger.info(f"Connected to environment index {client.env_id}")
        self.client.reset(task_idx)
        self._initialize_state()
    
    def _initialize_state(self):
        """
        Set up the initial state from its environment client.
        """
        initial_instruction = self.client.observe()
        if isinstance(initial_instruction, dict):
            initial_instruction = initial_instruction["observation"]

        self.conversation = [
            {"from": "human", "value": self.system_prompt, "loss": None},
            {"from": "gpt", "value": "Ok.", "loss": False},
            {"from": "human", "value": initial_instruction, "loss": None},
        ]
        self.terminal = False

    @property
    def state(self) -> List["ConversationMessage"]:
        return self.conversation

    async def step(self, state: Any) -> None:
        if self.terminal:
            return
        
        raw_output, _ = self.model.generate(self.conversation)
        self.logger.info(f"{self.conversation[-1]['from']}: {self.conversation[-1]['value']}")
        self.logger.info(f"Model output: {raw_output}")
        try:
            action = parse_action(raw_output)
        except ActionParserError as e:
            self.logger.warning(f"Error parsing action {e}")
            action = NO_ACTION
        
        self.action_history.append(action)
        self.conversation.append(
            {"from": "gpt", "value": raw_output, "loss": True}
        )
        
        try:
            # The environment returns a raw observation string
            step_output = self.client.step(action)
            obs, reward, done = step_output['observation'], step_output['reward'], step_output['done']
        except Exception as e:
            self.logger.warning(
                f"Error stepping environment with action '{action}': {e}"
            )
            obs, reward, done = (
                "Invalid action! Please check the available actions and try again.",
                0.0,
                False,
            )

        self.cumulative_reward += reward
        self.terminal = done
        self.conversation.append({"from": "human", "value": obs, "loss": None})

        action_summary = (
            action.splitlines()[0] if action else NO_ACTION
        )
        self.logger.info(
            f"Reward: {reward:.2f} | Total: {self.cumulative_reward:.2f} | Action: {action_summary} | Done: {done}"
        )
    
    async def evaluate_state(self, state: Any) -> float:
        """evaluates state based on cumulative reward. Higher is better."""

        predicted_value = await self.value_function.predict_value(
            self.conversation, self.action_history
        )
        self.logger.info(f"Value Function Predicted: {predicted_value:.4f}")

        return predicted_value
    
    def clone(self):
        """
        clone by creating a new environment client and replaying
        the action history to sync its state.
        """
        new_client = self.client
        new_client.reset(self.task_idx)

        # replay the action history to bring it to the current state
        for action in self.action_history:
            new_client.step(action)

        # create the new agent with the now-synced client
        new_agent = self.__class__(
            client=new_client,
            model=self.model,
            task_idx=self.task_idx,
            value_function=self.value_function,
            system_prompt = self.system_prompt
        )

        # copy the rest of the agent's state
        new_agent.conversation = copy.deepcopy(self.conversation)
        new_agent.cumulative_reward = self.cumulative_reward
        new_agent.terminal = self.terminal
        new_agent.action_history = copy.deepcopy(self.action_history)
        new_agent.value_at_last_resample = self.value_at_last_resample
        new_agent.terminal_updated = self.terminal_updated
        new_agent.reward_at_last_resample = self.reward_at_last_resample

        return new_agent

class OracleAgent(AgentGymAgent):
    """
    An agent that overrides the value function with a
    Monte Carlo rollout to get the "true" value of its current state.
    """
    
    def __init__(self, client: AgentGymClient, task_idx, system_prompt, model, value_function, max_steps, num_rollouts):  
        self.max_steps = max_steps
        self.oracle_rollouts = num_rollouts
        super().__init__(client, task_idx, system_prompt, model, value_function)

    async def _run_rollout(self) -> float:
        """
        clones the agent's current state and runs a simulation to the end
        using the policy model.
        """

        cloned_agent = self.clone()
                
        current_step = len(cloned_agent.action_history)
                
        while current_step < self.max_steps and not cloned_agent.terminal:
            await cloned_agent.step(cloned_agent.state)
            current_step += 1                 

        return cloned_agent.cumulative_reward

    async def evaluate_state(self, state: Any) -> float:
        
        self.logger.info(f"Task {self.task_idx}: OracleAgent (particle) starting {self.oracle_rollouts} rollouts...")
                
        rollout_tasks = [self._run_rollout() for _ in range(self.oracle_rollouts)]                
        all_rewards = await asyncio.gather(*rollout_tasks)                
        avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        
        self.logger.info(f"Task {self.task_idx}: Oracle rollouts complete. Avg reward: {avg_reward:.4f}")
                
        return avg_reward + 1e-9

    def clone(self) -> "OracleAgent":
        """
        clone by creating a new environment client and replaying
        the action history to sync its state.
        """
        new_client = self.client
        new_client.reset(data_idx=self.task_idx)
        
        for action in self.action_history:
            new_client.step(action)

        new_agent = self.__class__(
            client=new_client,
            task_idx = self.task_idx,
            system_prompt = self.system_prompt,
            model = self.model,
            value_function = self.value_function,
            max_steps = self.max_steps,
            num_rollouts = self.oracle_rollouts
        )

        # copy the rest of the agent's state
        new_agent.conversation = copy.deepcopy(self.conversation)
        new_agent.cumulative_reward = self.cumulative_reward
        new_agent.terminal = self.terminal
        new_agent.action_history = copy.deepcopy(self.action_history)
        new_agent.value_at_last_resample = self.value_at_last_resample
        new_agent.reward_at_last_resample = self.reward_at_last_resample
        new_agent.terminal_updated = self.terminal_updated

        return new_agent
