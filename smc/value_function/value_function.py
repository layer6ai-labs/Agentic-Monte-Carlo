import torch
from typing import List, Dict
import logging
import re
from transformers import GenerationConfig
from smc.prompts import (
    VALUE_FUNCTION_PROMPT_TEMPLATE_SCIWORLD,
    VALUE_FUNCTION_PROMPT_TEMPLATE_NO_REGRESSION_SCIWORLD,
    VALUE_FUNCTION_PROMPT_TEMPLATE_WEBSHOP,
    VALUE_FUNCTION_PROMPT_TEMPLATE_NO_REGRESSION,
    VALUE_FUNCTION_PROMPT_TEMPLATE_MOVIE,
    VALUE_FUNCTION_PROMPT_TEMPLATE_NO_REGRESSION_MOVIE,
    VALUE_FUNCTION_PROMPT_TEMPLATE_WEATHER,
    VALUE_FUNCTION_PROMPT_TEMPLATE_NO_REGRESSION_WEATHER,
    VALUE_FUNCTION_PROMPT_TEMPLATE_TEXTCRAFT,
    VALUE_FUNCTION_PROMPT_TEMPLATE_NO_REGRESSION_TEXTCRAFT,
)

logger = logging.getLogger()
    
class ValueFunction:
    def __init__(self, model, tokenizer, use_regression_head: bool = True, MAX_LEN: int = 8192, template_version: str = "V2", loss_function: str = "mse"):
        self.model = model
        self.tokenizer = tokenizer
        self.use_regression_head = use_regression_head        
        self.MAX_LEN = MAX_LEN
        self.device = next(self.model.parameters()).device
        self.template_version = template_version
        self.loss_function = loss_function

    def _tokens(self, s: str):
        return [t.strip() for t in s.split("[SEP]") if t.strip()]

    def _extract_instruction(self, state_str: str) -> str:
        """return the instruction text (token after 'Instruction:')."""
        ts = self._tokens(state_str)
        for i, t in enumerate(ts):
            if t.lower().rstrip(":") == "instruction":
                return ts[i + 1].strip() if i + 1 < len(ts) else ""
        return ""

    def _strip_to_after_instruction(self, state_str: str) -> str:
        """keep everything after the instruction text, prefixed with '[SEP]'."""
        ts = self._tokens(state_str)
        for i, t in enumerate(ts):
            if t.lower().rstrip(":") == "instruction":
                start = i + 2  # after instruction text
                rest = ts[start:] if start < len(ts) else []
                return "[SEP] " + " [SEP] ".join(rest) if rest else ""
        # fallback (no explicit 'Instruction:' token)
        return "[SEP] " + " [SEP] ".join(ts)

    def _extract_crafting_goal(self, input_text: str):
        """Separates crafting commands from the goal. Returns (commands, goal)."""
        parts = input_text.strip().rsplit("Goal:", 1)
        if len(parts) < 2:
            return input_text.strip(), None
        commands = parts[0].strip()
        goal = parts[1].strip().rstrip('.')
        return commands, goal

    def _format_prompt(
        self, conversation: List[Dict], action_history: List[str]
    ) -> str:
        """
        formats the agent's conversation and action history into the specific prompt
        that the value function was trained on.
        """
        if self.template_version == "TEXTCRAFT":
            states_raw = [msg["value"] for msg in conversation if msg["from"] == "human"][1:]
            T = len(states_raw)
            _, instruction = self._extract_crafting_goal(states_raw[0])

            history_parts = []
            for i in range(1, T - 1):
                state_i, _ = self._extract_crafting_goal(states_raw[i])
                action_i = action_history[i] if i < len(action_history) else "(unknown)"
                history_parts.append(f"Round {i + 1} — STATE:\n{state_i}")
                history_parts.append(f"Round {i + 1} — ACTION:\n{action_i}")
            history_str = "\n".join(history_parts) if history_parts else "(none)"

            current_state_raw, _ = self._extract_crafting_goal(states_raw[T - 1])
            now_state = re.sub(r"\[SEP\] Reward.*", "", current_state_raw, flags=re.DOTALL).strip()
        else:
            initial_obs = conversation[2]["value"]

            if self.template_version == "SCIWORLD":
                instruction = initial_obs[:initial_obs.find("\n")]
            elif self.template_version in ("WEATHER", "MOVIE"):
                start_inst = "You should perform actions to accomplish the goal: "
                end_inst = "Give me one action."
                instruction = initial_obs[initial_obs.find(start_inst) + len(start_inst):initial_obs.find(end_inst)]
            else:
                instruction = self._extract_instruction(initial_obs) or "(instruction not found)"

            if not action_history:
                history_str = "(none)"
            else:
                history_parts = []
                if self.template_version == "SCIWORLD":
                    state_turns = [initial_obs[initial_obs.find("\n") + 1:]] + conversation[4:-1:2]
                    for i, action in enumerate(action_history):
                        raw_state_i = state_turns[i] if i == 0 else state_turns[i]["value"]
                        action_i = action if action else "(unknown)"
                        history_parts.append(f"Round {i + 1} — STATE:\n{raw_state_i}")
                        history_parts.append(f"Round {i + 1} — ACTION:\n{action_i}")
                elif self.template_version in ("WEATHER", "MOVIE"):
                    state_turns = [end_inst] + conversation[4:-1:2]
                    for i, action in enumerate(action_history):
                        raw_state_i = state_turns[i] if i == 0 else state_turns[i]["value"]
                        action_i = action if action else "(unknown)"
                        history_parts.append(f"Round {i + 1} — STATE:\n{raw_state_i}")
                        history_parts.append(f"Round {i + 1} — ACTION:\n{action_i}")
                else:
                    state_turns = conversation[2:-1:2]
                    for i, action in enumerate(action_history):
                        raw_state_i = state_turns[i]["value"]
                        stripped_state_i = self._strip_to_after_instruction(raw_state_i)
                        action_i = action if action else "(unknown)"
                        history_parts.append(f"Round {i + 1} — STATE:\n{stripped_state_i}")
                        history_parts.append(f"Round {i + 1} — ACTION:\n{action_i}")

                history_str = "\n".join(history_parts)

            if self.template_version in ("SCIWORLD", "WEATHER", "MOVIE"):
                now_state = conversation[-1]["value"]
            else:
                now_state = self._strip_to_after_instruction(conversation[-1]["value"])

        if self.use_regression_head:
            if self.template_version == "WEBSHOP":
                template = VALUE_FUNCTION_PROMPT_TEMPLATE_WEBSHOP
            elif self.template_version == "SCIWORLD":
                template = VALUE_FUNCTION_PROMPT_TEMPLATE_SCIWORLD
            elif self.template_version == "MOVIE":
                template = VALUE_FUNCTION_PROMPT_TEMPLATE_MOVIE
            elif self.template_version == "WEATHER":
                template = VALUE_FUNCTION_PROMPT_TEMPLATE_WEATHER
            elif self.template_version == "TEXTCRAFT":
                template = VALUE_FUNCTION_PROMPT_TEMPLATE_TEXTCRAFT
            else:
                logger.error("Not supported template version")
                raise ValueError("Not supported template version")
        else:
            logger.info("Using no-regression value function template.")
            if self.template_version == "WEBSHOP":
                template = VALUE_FUNCTION_PROMPT_TEMPLATE_NO_REGRESSION
            elif self.template_version == "SCIWORLD":
                template = VALUE_FUNCTION_PROMPT_TEMPLATE_NO_REGRESSION_SCIWORLD
            elif self.template_version == "MOVIE":
                template = VALUE_FUNCTION_PROMPT_TEMPLATE_NO_REGRESSION_MOVIE
            elif self.template_version == "WEATHER":
                template = VALUE_FUNCTION_PROMPT_TEMPLATE_NO_REGRESSION_WEATHER
            elif self.template_version == "TEXTCRAFT":
                template = VALUE_FUNCTION_PROMPT_TEMPLATE_NO_REGRESSION_TEXTCRAFT
            else:
                logger.error("Not supported template version")
                raise ValueError("Not supported template version")

        return template.format(
            instruction=instruction, history=history_str, now=now_state
        )
        
    async def predict_value(
        self, conversation: List[Dict], action_history: List[str]
    ) -> float:
        """
        generates the prompt, runs the value function model, and returns the
        predicted value as a float.
        """
        
        # # test: when uniform distribution / no-resampling

        # return 1.0

        prompt = self._format_prompt(conversation, action_history)

        # # debug print
        # logger.info(f"\n\n--- GENERATED VALUE FUNCTION PROMPT ---\n{prompt}\n---------------------------------------\n")

        if self.use_regression_head:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=self.MAX_LEN
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            try:                
                value_tensor = outputs['logits'] if isinstance(outputs, dict) else outputs
                                
                if self.loss_function == "bce":
                    predicted_logit = value_tensor.flatten()[0]
                    predicted_value_tensor = torch.sigmoid(predicted_logit)
                    predicted_value = predicted_value_tensor.item()
                else:
                    predicted_value = value_tensor.flatten()[0].item()
                
                return predicted_value

            except Exception as e:
                logger.error(
                    f"Could not extract value from regression model output. Error: {e}. "
                    f"Defaulting value to 0.0."
                )
                return 0.0 
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            gen_config = GenerationConfig(max_new_tokens=10, do_sample=False) 
            
            with torch.no_grad():
                outputs = self.model.generate(inputs.input_ids, generation_config=gen_config)
            
            response_text = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
            logger.debug(f"Value function generated text: '{response_text}'")

            match = re.search(r"(\d\.\d+)", response_text)
            
            if match:
                try:
                    return float(match.group(1))
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse float from value generator output: '{response_text}'. Defaulting to 0.0.")
                    return 0.0
            else:
                logger.warning(f"No float found in value generator output: '{response_text}'. Defaulting to 0.0.")
                return 0.0


class DummyValueFunction:
    """
    A placeholder value function that always returns 0.0.
    """
    async def predict_value(self, conversation: List[Dict], action_history: List[str]) -> float:
        
        return 0.0            
