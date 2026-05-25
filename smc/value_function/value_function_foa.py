import torch
from typing import List, Dict
import logging
import re
from transformers import GenerationConfig

from smc.prompts import VALUE_FUNCTION_FOA_TEMPLATE

logger = logging.getLogger()

    
class ValueFunction_FOA:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(self.model.parameters()).device

    def _format_prompt(
        self, conversation: List[Dict], action_history: List[str]
    ) -> str:
        """
        formats the agent's conversation and action history into the specific prompt
        that the value function was trained on.
        """
        # The instruction is always at index 2 in our conversation structure.
        template = VALUE_FUNCTION_FOA_TEMPLATE

        state_turns = conversation[2:-1:2]
        action_turns = conversation[3:-1:2]

        init_flag = True

        for state, action in zip(state_turns, action_turns):
            if init_flag:
                template += f"\n\n{state['value']}\n\n{action['value']}"
                init_flag = False
            else:
                try: 
                    tmp_state = state['value'][state['value'].find("Instruction:"):]
                    no_inst_loc = [m.start() for m in re.finditer(r"\[SEP\]", tmp_state)][1]                
                    no_inst_state = tmp_state[no_inst_loc+len(" [SEP]"):]
                    template += f"\n\nObservation:\n{no_inst_state}\n\n{action['value']}"
                except:
                    template += f"\n\n{state['value']}\n\n{action['value']}"
        try: 
            tmp_state = conversation[-1]['value'][conversation[-1]['value'].find("Instruction:"):]
            no_inst_loc = [m.start() for m in re.finditer(r"\[SEP\]", tmp_state)][1]                
            no_inst_state = tmp_state[no_inst_loc+len(" [SEP]"):]
            template += f"\n\nObservation:\n{no_inst_state}\n\nReflection:"
        except:
            template += f"\n\nObservation:\n{conversation[-1]['value']}\n\nReflection:"

        return template
        
    async def predict_value(
        self, conversation: List[Dict], action_history: List[str]
    ) -> float:
        """
        generates the prompt, runs the value function model, and returns the
        predicted value as a float.
        """

        prompt = self._format_prompt(conversation, action_history)

        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(self.device)
        gen_config = GenerationConfig(max_new_tokens=128, do_sample=False) 
        
        with torch.no_grad():
            outputs = self.model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], generation_config=gen_config)
        
        response_text = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        logger.info(f"Value function generated text: '{response_text}'")
        
        pattern = r"the correctness score is (\d+).*"
        match = re.search(pattern, response_text)
                
        if match:
            try:
                score_int_str = match.group(1) 
                score_int = int(score_int_str)
                    
                if 1 <= score_int <= 10:            
                    return score_int / 10.0 
                else:
                    logger.warning(f"Extracted score {score_int} is out of the valid range [1, 10]. Defaulting to 0.0.")
                    return 0.0

            except (ValueError, IndexError):
                logger.warning(f"Could not parse valid integer from match object. Defaulting to 0.0.")
                return 0.0            
    
        else:
            logger.warning(f"No float found in value generator output: '{response_text}'. Defaulting to 0.0.")
            return 0.0


