import abc
import logging
import time
import torch
from typing import Any, List, Optional, Tuple
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from agentenv.controller.agent import BaseChatTemplate


class Model(abc.ABC):

    model: Any
    tokenizer: Optional[Any]

    @abc.abstractmethod
    def generate(self, inputs) -> Tuple[str, Optional[str]]:
        """ Generate model output + optional output"""
        raise NotImplementedError
    
    def __call__(self, inputs) -> Tuple[str, Optional[str]]:
        return self.generate(inputs)


class APIModel(Model):
    """
    Model using external API
    """
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model        
        self.role = {"system": "system", "human": "user", "gpt": "assistant"}

    def _convert_conversation_to_api_format(self, conversation) -> List[dict]:
        """
        converts the WebshopAgent's 'from/value' history to the API's 'role/content' format.
        """
        api_messages = []
                
        system_prompt = f"{conversation[0]['value']}"
        api_messages.append({"role": "system", "content": system_prompt})


        for turn in conversation[1:]:
            role = 'user' if turn.get('from') == 'human' else 'assistant'
            content = turn.get('value', '')
            api_messages.append({"role": role, "content": content})
            
        return api_messages   

    def generate(self, conversation, generation_config: dict) -> Tuple[str, Optional[str]]:
        
        max_tokens = generation_config.get("max_tokens", 4096)
        max_completion_tokens = generation_config.get("max_completion_tokens", 4096)
        temperature = generation_config.get("temperature", 1)
        top_p = generation_config.get("top_p", 1)

        conversation = self._convert_conversation_to_api_format(conversation)

        while True:
            try:
                if "gpt-5" in self.model:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        # messages=[{"role": self.role[c["from"]], "content": c["value"]} for c in conversation],
                        # messages=conversation,
                        messages=[{"role": c["role"], "content": c["content"]} for c in conversation],
                        max_completion_tokens=max_completion_tokens,
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        # messages=[{"role": self.role[c["from"]], "content": c["value"]} for c in conversation],
                        # messages=conversation,
                        messages=[{"role": c["role"], "content": c["content"]} for c in conversation],
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p
                    )
            
                return response.choices[0].message.content, response.choices[0].message.reasoning_content if hasattr(response.choices[0].message, "reasoning_content") else None
            
            except Exception as e:
                print(e)
                time.sleep(1)


class LocalModel(Model):
    def __init__(
        self,
        model_path: str,
        policy_template: BaseChatTemplate, 
        device_map: str = "sequential",
        torch_dtype = torch.bfloat16,
        trust_remote_code = True
    ):
        logger = logging.getLogger()
        logger.info(f"Loading local policy model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map=device_map, torch_dtype=torch_dtype
        ).eval()
        self.policy_template = policy_template
        logger.info("Local policy model loaded successfully.")
    
    def generate(self, inputs, generation_config: GenerationConfig) -> Tuple[str | list[str], Optional[str]]:
        generation_config.update(
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        )

        tokenized = self.policy_template.tokenize_conversation(
            inputs, self.tokenizer, add_generation_prompt=True
        )

        input_ids = tokenized["input_ids"]

        inputs_tensor = torch.tensor([input_ids], device=self.model.device)
        outputs = self.model.generate(inputs_tensor, generation_config=generation_config)

        if generation_config.num_return_sequences == 1:

            new_tokens = outputs[0][len(input_ids) :]
            raw_model_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            if raw_model_output.endswith("</s>"):
                raw_model_output = raw_model_output[:-5]

        else:
            raw_model_output = []
            for output in outputs:
                new_tokens = output[len(input_ids) :]                                
                
                raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                if raw_output.endswith("</s>"):
                    raw_output = raw_output[:-5]
                
                raw_model_output.append(raw_output)

        return raw_model_output, None


class PolicyModel:
    """ Adds a clone method to a model """

    def __init__(
            self,
            model_class: type[Model],
            model_kwargs: dict,
            generation_config: GenerationConfig | dict,
            recreate_model_on_clone: bool = False,
            model: Optional[Model] = None
        ):
        if model is None:
            self.model = model_class(**model_kwargs)
        else:
            self.model = model
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.generation_config = generation_config
        self.recreate_model_on_clone = recreate_model_on_clone

    def clone(self):
        if self.recreate_model_on_clone:
            return self.__class__(self.model_class, self.model_kwargs, self.generation_config, True, None)
        else:
            return self.__class__(self.model_class, self.model_kwargs, self.generation_config, False, self.model)

    def generate(self, input):
        return self.model.generate(input, self.generation_config)
