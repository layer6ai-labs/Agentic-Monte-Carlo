import argparse
import json
import os
import sys
import torch
import time
from typing import Optional, List
from peft import PeftModel
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


sys.path.append("./envs/AgentGym/agentenv")
from agentenv.controller.agent import Llama2Template, Llama3Template, ChatMLTemplate
from agentenv.envs import WebshopTask
from agentenv.envs.sciworld import SciworldTask
from agentenv.envs.movie import MovieTask
from agentenv.envs.weather import WeatherTask

from smc.agents.sciworld import SciworldAgent
from smc.agents.webshop import WebshopAgent
from smc.agents.movie import MovieAgent
from smc.agents.weather import WeatherAgent
from smc.clients import CustomTextCraftTask
from smc.agents.agent import AgentGymAgent
from smc.smc import run_smc_for_task
from smc.policy import APIModel, LocalModel, PolicyModel
from smc.utils.logging_utils import setup_logging
from smc.value_function.value_function import DummyValueFunction, ValueFunction
from smc.value_function.value_function_foa import ValueFunction_FOA
from smc.train_lora_regression import LlamaWithLoraRegHead
from smc.prompts import WEBSHOP_REACT_PROMPT, SCIWORLD_REACT_PROMPT, SCIWORLD_REFLACT_PROMPT, TEXTCRAFT_REACT_PROMPT, MOVIE_REACT_PROMPT, WEATHER_REACT_PROMPT


class Args(argparse.Namespace):
    env: str
    n_particles: int
    max_steps: int
    resample_steps: Optional[List[int]]
    model_path: str
    api_key: str
    api_base_url: str
    vf_ckpt_path: Optional[str]
    vf_base_model: Optional[str]
    template_version: str
    loss_function: str
    max_len: int
    port: int
    vf_as_generator: bool
    foa: bool
    force_buy_on_extra_step: bool  # set True for webshop, False for all others
    caching: bool                  # set True for textcraft, False for all others
    ess_threshold: float
    weight_temperature: float
    policy_prompt: str
    mode: str
    tasks: Optional[list[int]]

def parse_arguments() -> Args:    
    parser = argparse.ArgumentParser(
        description="Run a Sequential Monte Carlo experiment on the specified environment."
    )
    parser.add_argument(
        "--env",
        type=str,
        default="webshop",
        choices=["webshop", "sciworld", "textcraft", "movie", "weather"],
        help="Which environment to run the experiment in."
    )
    parser.add_argument(
        "--n-particles",
        type=int,
        default=15,
        help="Number of agent particles in the filter.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum number of steps each agent can take per task.",
    )
    parser.add_argument(
        '--resample-steps',
        type=int,
        nargs='+', # This tells argparse to accept one or more numbers
        default=40,
        help="A list of steps to resample at (e.g., --resample-steps 10 12 14)."
    )    
    parser.add_argument(
        "--model-path",
        type=str,
        default="meta-llama/Llama-3.2-11B-Vision-Instruct",
        help="Policy model name.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="API Key for the external model. If provided (non-empty), the API agent is used.",
    )    
    parser.add_argument(
        "--api-base-url",
        type=str,
        default="",
        help="Base URL for the API endpoint.",
    )    
    parser.add_argument(
        "--vf-ckpt-path", 
        type=str,
        required=False,
        help="Path to the value function."
    )
    parser.add_argument(
        "--vf-base-model",
        type=str,
        required=False,
        help="Base model for the value function.",
    )
    parser.add_argument(
        '--template_version',
        type=str,
        default="WEBSHOP",
        choices=["WEBSHOP", "SCIWORLD", "TEXTCRAFT", "MOVIE", "WEATHER"],
        help="Template version for trained value function.",
    )
    parser.add_argument(
        "--loss-function",
        type=str,
        default="mse",
        choices=["mse", "bce"],
        help="Loss function used in trained value function ('mse' or 'bce').",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=1024, 
        help="Max length to generate per step for the policy model.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=36001,
        help="Port for the AgentGym environment server.",
    )
    parser.add_argument(
        '--vf-as-generator',
        action='store_true',
        help="Use a standard generator model for the value function, prompted to output a score as text."
    )
    parser.add_argument(
        '--foa',
        action='store_true',
        help="FoA-style baseline for the value function, prompted to output a score as text."
    )    
    parser.add_argument(
        '--ess-threshold',
        type=float,
        default=0.5,
        help="ESS threshold (fraction of N) to trigger resampling in adaptive mode."
    )    
    parser.add_argument(
        '--weight-temperature',
        type=float,
        default=1.0,
        help="Weight temperature in value function update. Small value (e.g.,0.1) requires for ESS."
    )   
    parser.add_argument(
        '--policy-prompt',
        type=str,
        default="react",
        choices=["react", "reflact"],
        help="Policy prompt for the agent."
    )
    parser.add_argument(
        '--mode',
        type=str,
        default="test",
        choices=["test", "valid", "train"],
        help="Train or Test mode."
    )
    parser.add_argument(
        '--tasks',
        type=int,
        nargs='+',
        default=None,
        help="A list of tasks to run smc on"
    )
    return parser.parse_args(namespace=Args())

def main():
    args = parse_arguments()
    args.force_buy_on_extra_step = False  # overridden per-env below
    args.caching = False                  # overridden per-env below
    logger, timestamp = setup_logging()

    ### Prepare the policy model ###
    if args.api_key:
        policy_model_class = APIModel
        policy_model_kwargs = {
            "api_key": args.api_key, "model": args.model_path, "base_url": args.api_base_url
        }
        generation_config = {
            "max_tokens": args.max_len,
            "max_completion_tokens": args.max_len,
            "temperature": 1.0,
            "top_p": 0.95
        }
        recreate_model_on_clone = True
        
    else:
        policy_model_class = LocalModel
        policy_template = Llama2Template()
        if "Llama-3" in args.model_path:
            policy_template = Llama3Template()
        if "Qwen" in args.model_path:
            policy_template = ChatMLTemplate()
        policy_model_kwargs = {
            "model_path": args.model_path, 
            "policy_template": policy_template
        }
        generation_config = GenerationConfig(
            max_new_tokens=args.max_len, do_sample=True, temperature=1.0, top_p=0.95,            
        )
        recreate_model_on_clone = False

    try:    
        logger.info(f"Initializing Policy Model: {args.model_path}")
        policy_model = PolicyModel(
            model_class = policy_model_class,
            model_kwargs = policy_model_kwargs,
            generation_config = generation_config,
            recreate_model_on_clone = recreate_model_on_clone
        )
    except Exception as e:
        logger.error(f"Failed to initialize Policy Model: {e}", exc_info=True)
        return

    policy_model_name = args.model_path.split("/")[-1]
    
    ### Prepare value function ###
    # TODO: This should mostly live in a separate file in the value_function subfolder
    # It would help to break up the args parser as well
    value_function = None
    try:
        if not args.vf_as_generator and not args.vf_ckpt_path:
            logger.info("--- Mode: Value Function with Dummy ---")                        
            value_function = DummyValueFunction()

        elif args.vf_as_generator:
            logger.info("--- Mode: Value Function as a Generator ---")
            
            if not args.vf_base_model:
                logger.info(f"Same model for policy and value function")                    
                vf_base_model = policy_model.model.model              
                vf_tokenizer = policy_model.model.tokenizer
                if vf_tokenizer is None:
                    # TODO: Can we update the value function class to work with api models or do we need white box models?
                    logger.error("Can't use the same model for policy and value function if using API model.")
                    return
                
            else:
                logger.info(f"Model for value function: {args.vf_base_model}")
                vf_tokenizer = AutoTokenizer.from_pretrained(args.vf_base_model, use_fast=True)
                vf_base_model = AutoModelForCausalLM.from_pretrained(
                    args.vf_base_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                )
                vf_base_model.to("cuda" if torch.cuda.is_available() else "cpu")

            if args.foa:
                logger.info("Using FoA-style prompting for value function")
                value_function = ValueFunction_FOA(
                    model=vf_base_model, 
                    tokenizer=vf_tokenizer,
                )
            else:
                value_function = ValueFunction(
                    model=vf_base_model, 
                    tokenizer=vf_tokenizer, 
                    use_regression_head=False,
                    template_version=args.template_version,
                    loss_function=args.loss_function
                )
        else:
            logger.info("--- Mode: Value Function with Regression Head ---")

            if not args.vf_ckpt_path or not args.vf_base_model:            
                logger.error("Vf-ckpt-path and vf-base-model are required for this mode.")
                raise ValueError("Missing VF paths for regression mode.")                

            vf_tokenizer = AutoTokenizer.from_pretrained(args.vf_ckpt_path, use_fast=False)
            vf_base_model = AutoModelForCausalLM.from_pretrained(
                args.vf_base_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
            )
        
            lora_model = PeftModel.from_pretrained(vf_base_model, args.vf_ckpt_path)
            value_model = LlamaWithLoraRegHead(lora_model, num_outputs=1)
            reg_head_state_dict = torch.load(
                os.path.join(args.vf_ckpt_path, "reg_head_state_dict.pt"), map_location="cpu"
            )
            value_model.head.load_state_dict(reg_head_state_dict, strict=False)
            value_model.to("cuda" if torch.cuda.is_available() else "cpu")
            value_model.eval()
            
            value_function = ValueFunction(
                model=value_model,
                tokenizer=vf_tokenizer,
                use_regression_head=True,
                template_version=args.template_version,
                loss_function=args.loss_function
            )
    
        logger.info("Value function model loaded successfully.")

        # TODO: ValueFunction and all the other classes should have a name attribute
        if args.vf_as_generator: 
            value_model_name = "vf_generator"
        elif args.vf_ckpt_path:
            value_model_name = args.vf_ckpt_path.split("/")[-1]        
        else:
            value_model_name = "dummy"
    
    except Exception as e:
        logger.error(f"Failed to load value function: {e}", exc_info=True)
        return

    ### Prepare the correct environment ###

    # Per-environment config for the four "standard" envs that share the same structure.
    # TextCraft is handled separately below (different client/agent interface).
    STANDARD_ENV_CONFIGS = {
        "webshop":  dict(task_class=WebshopTask,  agent_class=WebshopAgent,
                         policy_prompt=WEBSHOP_REACT_PROMPT,
                         test_idx="webshop_idx/test_idx.json",
                         train_idx="webshop_idx/train_idx.json",
                         force_buy=True),
        "sciworld": dict(task_class=SciworldTask, agent_class=SciworldAgent,
                         policy_prompt=SCIWORLD_REACT_PROMPT if args.policy_prompt == "react" else SCIWORLD_REFLACT_PROMPT,
                         test_idx="sciworld_idx/sciworld_test_idx.json",
                         train_idx="sciworld_idx/sciworld_train_idx.json",
                         force_buy=False),
        "movie":    dict(task_class=MovieTask,    agent_class=MovieAgent,
                         policy_prompt=MOVIE_REACT_PROMPT,
                         test_idx="movie_idx/test_idx.json",
                         train_idx="movie_idx/train_idx.json",
                         force_buy=False),
        "weather":  dict(task_class=WeatherTask,  agent_class=WeatherAgent,
                         policy_prompt=WEATHER_REACT_PROMPT,
                         test_idx="weather_idx/test_idx.json",
                         train_idx="weather_idx/train_idx.json",
                         force_buy=False),
    }

    if args.env == "textcraft":
        task_class = CustomTextCraftTask
        client_args = {"base_url": f"http://127.0.0.1:{args.port}", "timeout": 500}
        agent_class = AgentGymAgent
        agent_kwargs = {
            "model": policy_model,
            "value_function": value_function,
            "system_prompt": TEXTCRAFT_REACT_PROMPT,
        }
        args.force_buy_on_extra_step = False
        args.caching = True
        if args.tasks:
            task_ids = args.tasks
        else:
            idx_file = "textcraft_idx/textcraft_test_ids.csv" if args.mode == "test" else "textcraft_idx/textcraft_train_ids.csv"
            with open(idx_file) as f:
                task_ids = f.read().splitlines()
            logger.info(f"{'Test' if args.mode == 'test' else 'Train'} mode. Sample size: {len(task_ids)}")

    elif args.env in STANDARD_ENV_CONFIGS:
        cfg = STANDARD_ENV_CONFIGS[args.env]
        task_class = cfg["task_class"]
        client_args = {
            "env_server_base": f"http://127.0.0.1:{args.port}",
            "data_len": 200,
            "timeout": 500,
            "action_format": args.policy_prompt,
        }
        agent_class = cfg["agent_class"]
        agent_kwargs = {
            "model": policy_model,
            "client_args": client_args,
            "value_function": value_function,
            "policy_prompt": cfg["policy_prompt"],
            "action_format": args.policy_prompt,
        }
        args.force_buy_on_extra_step = cfg["force_buy"]
        if args.tasks:
            task_ids = args.tasks
        else:
            idx_file = cfg["test_idx"] if args.mode == "test" else cfg["train_idx"]
            with open(idx_file) as f:
                task_ids = json.load(f)
            logger.info(f"{'Test' if args.mode == 'test' else 'Train'} mode. Sample size: {len(task_ids)}")

    else:
        logger.error(f"Environment {args.env} not supported.")
        return

    ### Create necessary folders ###
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    results_filename = os.path.join(
        results_dir, 
        f"{args.env}_{policy_model_name}-{value_model_name}_{timestamp}.json"
    )

    ### Starting experiment ###
    logger.info("--- Starting SMC Experiment ---")
    logger.info(f"Policy Model: {policy_model_name}")
    logger.info(f"Value Model: {value_model_name}")
    logger.info(f"Saving results to: {results_filename}")
    logger.info(f"Using '{args.policy_prompt}' policy prompt.")
    logger.info(f"Temperature for SMC weighting: {args.weight_temperature}")
    logger.info(f"Cache is {'enabled' if args.caching else 'disabled'}")
    logger.info("Loading value function model")

    # loop through tasks
    results = []

    with open(results_filename, 'w') as f:
        json.dump(results, f)

    for task_id in tqdm.tqdm(task_ids, desc="Running tasks", position=0):
        task_start_time = time.time()
        
        all_final_trajectories = run_smc_for_task(
            task_idx=task_id,
            task_class=task_class,
            client_args=client_args,
            agent_class=agent_class,
            agent_kwargs=agent_kwargs,
            n_particles=args.n_particles,
            max_steps=args.max_steps,
            resample_steps=args.resample_steps,
            force_buy_on_extra_step=args.force_buy_on_extra_step, # TODO: This really needs to be removed and made an argument for the WebshopAgent
            ess_threshold=args.ess_threshold,
            weight_temperature=args.weight_temperature,
            cache=args.caching
        )
        
        task_end_time = time.time()
        task_duration = task_end_time - task_start_time
        
        logger.info(f"Task {task_id} finished in {task_duration:.2f} seconds")

        results.append({
            "task_id": task_id, 
            "trajectories": all_final_trajectories,
            "duration_seconds": task_duration  
        })
        
        with open(results_filename, 'w') as f:
            json.dump(results, f, indent=4)

    # display final summary
    logger.info(f"\n\n{'#' * 20} FINAL SUMMARY {'#' * 20}")
    if not results:
        logger.warning("No results to display.")
        return        

    total_best_reward = 0
    total_time = 0
    
    for result in results:
        best_reward_for_task = result['trajectories'][0]['reward'] if result['trajectories'] else 0
        task_time = result.get('duration_seconds', 0)
        total_time += task_time
        
        logger.info(f"Task {result['task_id']}: Best Reward = {best_reward_for_task:.2f} | Time = {task_time:.2f}s")
        total_best_reward += best_reward_for_task

    avg_reward = total_best_reward / len(results) if results else 0
    logger.info(f"\nAverage Best Reward Across All Tasks: {avg_reward:.4f}")
    logger.info(f"Total Experiment Time: {total_time:.2f}s")


if __name__ == "__main__":
    main()
