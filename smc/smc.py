import logging
import copy
from typing import List

import tqdm

from agentenv.controller.task import BaseTask

from smc.agents.agent import Agent
from smc.bootstrap_filter import SMCBootstrapFilter


def run_smc_for_task(
    task_idx: int,
    task_class: type[BaseTask],
    client_args: dict,
    n_particles: int,
    agent_class: type[Agent],
    agent_kwargs: dict,
    max_steps: int,
    resample_steps: List[int],
    force_buy_on_extra_step: bool,
    ess_threshold: float,
    weight_temperature: float,
    cache: bool = False,
):
    """    
    runs the complete SMC experiment for a single task ID and returns ALL trajectories (sorted by reward).
    """
    logger = logging.getLogger()
    logger.info(f"\n{'#' * 20} RUNNING EXPERIMENT FOR TASK ID: {task_idx} {'#' * 20}")

    all_created_clients = []
    processed_ids = set()
    all_trajectories = []

    best_overall_reward = -float('inf')
    best_overall_trajectory = None

    try:
        # create agent pool for task
        task = task_class(client_args=client_args, n_clients=n_particles)
        all_created_clients.extend(task.clients)
        for client in task.clients:
            client.reset(task_idx)
        
        logger.info(f"Task Goal: {task.clients[0].observe()}")

        agents = [
            agent_class(task_idx=task_idx, client=client, **agent_kwargs)
            for client in task.clients
        ]
        logger.info(f"{len(agents)} agents created successfully for task {task_idx}.")

        # instantiate and run SMC Filter ---
        smc_filter = SMCBootstrapFilter(
            agents=agents,
            goal_state=None,
            resample_steps=resample_steps,
            max_step=max_steps,
            logger=logger,
            ess_threshold=ess_threshold,
            weight_temperature=weight_temperature,
        )

        for step_num in tqdm.tqdm(range(max_steps), desc=f"Steps in task {task_idx}", position=1):
            logger.info(f"\n--- Task {task_idx} | Step {step_num + 1}/{max_steps} ---")
            if not smc_filter.compute_next_step() or all(a.terminal for a in smc_filter.agents):
                logger.info(f"Task {task_idx}: All agents terminated or max steps reached.")
                break

            for agent in smc_filter.agents:
                if agent.client not in all_created_clients:
                    all_created_clients.append(agent.client)

            if cache:
                for agent in smc_filter.agents:
                    if agent.cumulative_reward > best_overall_reward: 
                        logger.info(f"NEW PEAK DISCOVERED: Agent {agent} improved reward from {best_overall_reward:.2f} to {agent.cumulative_reward:.2f}")                                                            
                        best_overall_reward = agent.cumulative_reward
                        best_overall_trajectory = {
                            "rank": 0, 
                            "reward": agent.cumulative_reward,
                            "history": copy.deepcopy(agent.conversation),
                            "is_peak_tracker": True # for debugging
                        }

                    if agent.terminal and id(agent) not in processed_ids:
                        all_trajectories.append({
                            "rank": 0, # Placeholder
                            "reward": agent.cumulative_reward,
                            "history": copy.deepcopy(agent.conversation),
                        })
                        processed_ids.add(id(agent))

            current_pool_best = max(a.cumulative_reward for a in smc_filter.agents) if smc_filter.agents else 0.0
            if cache:
                archived_best = max(t['reward'] for t in all_trajectories) if all_trajectories else 0.0            
                current_max_reward = max(current_pool_best, archived_best, best_overall_reward)
                # current_max_reward = max(current_pool_best, archived_best)
            else:
                current_max_reward = current_pool_best

            logger.info(f"Task {task_idx}: Best reward in current step: {current_max_reward:.2f}")

        if force_buy_on_extra_step: # TODO: this logic needs to be moved to webshop agent / client, or be abstracted away somehow
            logger.info(f"\n--- Task {task_idx} | Extra Step: Forcing 'click[Buy Now]' ---")
            if smc_filter.agents:
                for agent in smc_filter.agents:
                    # Only force the action if the agent hasn't already terminated
                    if not agent.terminal:
                        agent.force_action_sync("click[Buy Now]")
            else:
                logger.warning("No agents available to force 'Buy Now' action.")

        if smc_filter.agents:
            for agent in smc_filter.agents:
                if id(agent) not in processed_ids:
                    all_trajectories.append({
                        "rank": 0,  # Placeholder
                        "reward": agent.cumulative_reward,
                        "history": copy.deepcopy(agent.conversation),
                    })
                    processed_ids.add(id(agent))

        if cache and best_overall_trajectory:
            if not any(x['reward'] == best_overall_reward for x in all_trajectories):
                logger.info(f"Adding global peak trajectory (Reward: {best_overall_reward:.2f}) to final results.")
                all_trajectories.append(best_overall_trajectory)

        final_trajectories = sorted(all_trajectories, key=lambda x: x['reward'], reverse=True)

        for i, traj in enumerate(final_trajectories):
            traj['rank'] = i + 1

        logger.info(f"Task {task_idx}: Finished. Recorded {len(final_trajectories)} trajectories.")

        if len(final_trajectories) > 0:
            logger.info(f"  - Best Reward Found: {final_trajectories[0]['reward']:.2f}")


    finally:
        for client in all_created_clients:
            try:
                client.close()
            except Exception:
                pass

    return final_trajectories