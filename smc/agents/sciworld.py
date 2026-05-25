import asyncio
import copy
import logging
from typing import List, Any
from agentenv.controller.types import ActionFormat, ConversationMessage
from agentenv.controller.utils import BaseAdapter
from agentenv.envs.sciworld import SciworldEnvClient
from smc.agents.agent import Agent
from smc.policy import PolicyModel
from smc.value_function.value_function import ValueFunction


class SciworldAgent(Agent):
    """
    Agent that interacts with the Sciworld environment using a language model.
    """

    def __init__(
        self,
        task_idx: int,
        client: SciworldEnvClient,
        model: PolicyModel,
        client_args: dict,
        value_function: ValueFunction,
        policy_prompt: str,
        action_format: str = "react",
        **kwargs,
    ):
        super().__init__()
        self.logger = logging.getLogger()
        self.client = client
        self.model = model
        self.task_idx = task_idx
        self.policy_prompt = policy_prompt
        self.action_format = action_format
        self.client_args = client_args  # store args to create new clients
        
        self.conversation: List[ConversationMessage] = []
        self.action_history: List[str] = []  # track actions for value function and replay during cloning
        self.cumulative_reward: float = 0.0
        self.terminal: bool = False
        self.terminal_updated: bool = False
        self.value_function = value_function        
        self.value_at_last_resample: float = 0.0      
        self.reward_at_last_resample: float = 0.0
                
        # initialize state from the provided client
        self._initialize_state()

    def _initialize_state(self):
        """
        Set up the initial state from its environment client.
        """
        initial_instruction = self.client.observe()

        self.conversation = [
            {"from": "human", "value": self.policy_prompt, "loss": None},
            {"from": "gpt", "value": "OK. I'll follow your instructions and try my best to solve the task.", "loss": False},
            {"from": "human", "value": initial_instruction, "loss": None},
        ]
        self.terminal = False

    @property
    def state(self) -> List[ConversationMessage]:
        return self.conversation

    def _step_sync(self):
        """synchronously performs one turn"""
        if self.terminal:
            return

        try:
            raw_model_output, _ = self.model.generate(self.conversation)
        except Exception as e:
            self.logger.error(f"Error during agent generation: {e}", exc_info=True)
            raw_model_output = ""

        if not raw_model_output:
            self.logger.error("Model returned empty output. Terminating agent.")
            self.terminal = True
            return

        if self.action_format == "react":
            parsed_action = BaseAdapter.action_parser(raw_model_output, ActionFormat.REACT) 
        else:
            parsed_action = BaseAdapter.action_parser(raw_model_output, ActionFormat.REFLACT)

        self.action_history.append(parsed_action)
        self.conversation.append({"from": "gpt", "value": raw_model_output, "loss": True})

        try:
            step_output = self.client.step(raw_model_output) 
            obs, reward, done = step_output.state, step_output.reward, step_output.done
        except Exception as e:
            self.logger.warning(f"Error stepping environment with action '{parsed_action}': {e}")
            obs, reward, done = (
                "Invalid action! Please check the available actions and try again.",
                0.0,
                False,
            )

        reward = reward / 100 # sciworld gives score between -100, 0-100
        self.cumulative_reward = reward ## sciworld has intermediate reward which is stacked.
        self.terminal = done
        self.conversation.append({"from": "human", "value": obs, "loss": None, "reward": reward})
                    
        action_summary = (parsed_action.splitlines()[0] if parsed_action else "[empty action]")
        self.logger.info(f"Reward: {reward:.2f} | Total: {self.cumulative_reward:.2f} | Action: {action_summary}")

    async def step(self, state: Any) -> None:
        """asynchronously executes a step by running the sync method in a thread."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._step_sync)

    async def evaluate_state(self, state: Any) -> float:
        """evaluates state based on cumulative reward. Higher is better."""

        predicted_value = await self.value_function.predict_value(self.conversation, self.action_history)
        self.logger.info(f"Value Function Predicted: {predicted_value:.4f}")

        return predicted_value

    def _get_synced_client(self):
        """Creates a new client and replays history to sync state."""
        new_client = SciworldEnvClient(**self.client_args)
        new_client.reset(data_idx=self.task_idx)
        for action in self.action_history:
            if self.policy_prompt == "react":
                new_client.step("Thought:\nskip\n\nAction:"+action)
            else:
                new_client.step("Reflection:\nskip\n\nAction:"+action)
        return new_client

    def _copy_state_to(self, new_agent: "SciworldAgent"):
        """Copies mutable state variables to the new agent."""
        new_agent.conversation = copy.deepcopy(self.conversation)
        new_agent.cumulative_reward = self.cumulative_reward
        new_agent.terminal = self.terminal
        new_agent.terminal_updated = self.terminal_updated
        new_agent.action_history = copy.deepcopy(self.action_history)
        new_agent.value_at_last_resample = self.value_at_last_resample
        new_agent.reward_at_last_resample = self.reward_at_last_resample
        
        return new_agent

    def clone(self) -> "SciworldAgent":
        new_client = self._get_synced_client()

        # create the new agent with the now-synced client
        new_agent = self.__class__(
            client=new_client,
            model=self.model.clone(),
            task_idx=self.task_idx,
            client_args=self.client_args,
            value_function=self.value_function,
            policy_prompt=self.policy_prompt,
            action_format=self.action_format,
        )
        
        return self._copy_state_to(new_agent)

    def force_action_sync(self, action: str):
        """Forces the agent to take a specific action, bypassing the policy model."""
        if self.terminal:
            return

        # store the forced action for history/cloning consistency
        self.action_history.append(action)
        try:
            if self.policy_prompt == "react":
                step_output = self.client.step("Thought:\nskip\n\nAction:"+action)
            else:
                step_output = self.client.step("Reflection:\nskip\n\nAction:"+action)

            obs, reward, done = step_output.state, step_output.reward, step_output.done

        except Exception as e:
            logging.warning(f"Error during forced action '{action}': {e}")
            obs, reward, done = "Error during forced action.", 0.0, True
        
        reward = reward / 100 # sciworld gives score between -100, 0-100
        self.cumulative_reward = reward
        self.terminal = done
        # append the final observation to the history for a complete record
        self.conversation.append({"from": "human", "value": obs, "loss": None, "reward": reward})

        logging.info(
            f"FORCED ACTION: {action} | Final Reward: {reward:.2f} | Final Total: {self.cumulative_reward:.2f}"
        )
