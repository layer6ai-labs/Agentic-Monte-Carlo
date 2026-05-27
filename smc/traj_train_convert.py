import argparse
import random
import re

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict


PREAMBLES = {
    "webshop": (
        "You are a value estimator for a web shopping task.\n"
        "Your job is to estimate the expected future reward (return) from the current STATE,\n"
        "given the INSTRUCTION and the previous STATES, ACTIONS provided as context.\n\n"
        "Format of the input you will receive:\n"
        "- INSTRUCTION: The original shopping request to evaluate progress against.\n"
        "- HISTORY: Consecutive pairs of (STATE, ACTION) taken so far.\n"
        "- NOW: Current STATE.\n\n"
        "Use the INSTRUCTION with HISTORY and NOW to infer progress and likelihood of success.\n"
        "Focus on attributes, type, options, and price of shown products relative to the INSTRUCTION.\n"
        "Based on your analysis, provide a score between 0.0 and 1.0."
    ),
    "sciworld": (
        "You are a value estimator for a science world task.\n"
        "Your job is to estimate the expected future reward (return) from the current STATE,\n"
        "given the INSTRUCTION and the previous STATES, ACTIONS provided as context.\n\n"
        "Format of the input you will receive:\n"
        "- INSTRUCTION: The original scientific request to evaluate progress against.\n"
        "- HISTORY: Consecutive pairs of (STATE, ACTION) taken so far.\n"
        "- NOW: Current STATE.\n\n"
        "Use the INSTRUCTION with HISTORY and NOW to infer progress and likelihood of success.\n"
        "Focus on milestones, such as locating the correct room, acquiring required tools,\n"
        "and following the specific steps (e.g., 'focus', 'interaction') outlined in the INSTRUCTION.\n"
        "Heavily penalize states where the observation is 'No known action matches that input',\n"
        "as this indicates the agent is stuck in an invalid command loop or syntax error.\n"
        "Based on your analysis, provide a score between -1.0 and 1.0."
    ),
    "textcraft": (
        ## @Raunaq please update this
        "You are a value estimator for a Minecraft crafting task.\n"
        "Your job is to estimate the expected future reward (return) from the current STATE,\n"
        "given the INSTRUCTION and the previous STATES, ACTIONS provided as context.\n\n"
        "Format of the input you will receive:\n"
        "- INSTRUCTION: The original crafting request to evaluate progress against.\n"
        "- HISTORY: Consecutive pairs of (STATE, ACTION) taken so far.\n"
        "- NOW: Current STATE.\n\n"
        "Use the INSTRUCTION with HISTORY and NOW to infer progress and likelihood of success.\n"
        "Focus on the crafting material, intermediate crafting results and quantities relative to the INSTRUCTION.\n"
        "Based on your analysis, provide a score between 0.0 and 1.0."
    ),
    "movie": (
        "You are a value estimator for an autonomous tool-use and API-calling agent.\n"
        "Your job is to estimate the expected future reward (return) from the current STATE,\n"
        "given the INSTRUCTION and the previous STATES and ACTIONS provided as context.\n\n"
        "Format of the input you will receive:\n"
        "- INSTRUCTION: The original user request or problem to solve.\n"
        "- HISTORY: Consecutive pairs of (STATE, ACTION) taken so far.\n"
        "- NOW: Current STATE (usually the output of the most recent tool call or an error message).\n\n"
        "Use the INSTRUCTION with HISTORY and NOW to infer progress and likelihood of success.\n"
        "Focus on logical milestones, such as:\n"
        "- Gathering necessary prerequisite context (e.g., properly using `get_search_movie` or "
        "`get_search_person` to map string names to their respective numeric IDs).\n"
        "- Successfully navigating relational data structures (e.g., isolating a specific `person_id` "
        "from a `get_movie_crew` payload to subsequently look up their biography).\n"
        "- Making valid tool calls that return successful, populated data payloads rather than empty lists or errors.\n\n"
        "Heavily penalize states where:\n"
        "- The agent hallucinates or guesses a `movie_id` or `person_id` instead of retrieving it via search.\n"
        "- The agent passes a string name into an endpoint that strictly requires a numeric ID.\n"
        "- The STATE returns an API error, missing parameter warning, or 'invalid action'.\n"
        "- The agent is stuck in an endless loop of repeating the exact same failed action.\n"
        "- The agent calls the finish action with a clearly incorrect or ungrounded answer.\n\n"
        "Based on your analysis, provide a score between 0.0 and 1.0."
    ),
    "weather": (
        "You are a value estimator for an autonomous tool-use and API-calling agent.\n"
        "Your job is to estimate the expected future reward (return) from the current STATE,\n"
        "given the INSTRUCTION and the previous STATES and ACTIONS provided as context.\n\n"
        "Format of the input you will receive:\n"
        "- INSTRUCTION: The original user request or problem to solve.\n"
        "- HISTORY: Consecutive pairs of (STATE, ACTION) taken so far.\n"
        "- NOW: Current STATE (usually the output of the most recent tool call or an error message).\n\n"
        "Use the INSTRUCTION with HISTORY and NOW to infer progress and likelihood of success.\n"
        "Focus on logical milestones, such as:\n"
        "- Gathering necessary prerequisite context (e.g., fetching the current date or user location).\n"
        "- Successfully mapping named entities to required API parameters "
        "(e.g., converting a city name to latitude and longitude).\n"
        "- Making valid tool calls that return successful data payloads rather than errors.\n\n"
        "Heavily penalize states where:\n"
        "- The STATE returns an API error, missing parameter warning, or 'invalid action'.\n"
        "- The agent hallucinated a tool name that does not exist in the system prompt.\n"
        "- The agent is stuck in an endless loop of repeating the exact same failed action.\n"
        "- The agent calls the finish action with a clearly incorrect or ungrounded answer.\n\n"
        "Based on your analysis, provide a score between 0.0 and 1.0."
    ),
}


# ---------------------------------------------------------------------------
# State / instruction helpers
# ---------------------------------------------------------------------------

def _tokens(s: str):
    return [t.strip() for t in s.split("[SEP]") if t.strip()]


def _extract_instruction_webshop(state_str: str) -> str:
    """Return the instruction text (token after 'Instruction:') for Webshop."""
    ts = _tokens(state_str)
    for i, t in enumerate(ts):
        if t.lower().rstrip(":") == "instruction":
            return ts[i + 1].strip() if i + 1 < len(ts) else ""
    return ""


def _strip_to_after_instruction(state_str: str) -> str:
    """Keep the state content after the instruction text (Webshop format)."""
    ts = _tokens(state_str)
    for i, t in enumerate(ts):
        if t.lower().rstrip(":") == "instruction":
            start = i + 2
            rest = ts[start:] if start < len(ts) else []
            return "[SEP] " + " [SEP] ".join(rest) if rest else ""
    return "[SEP] " + " [SEP] ".join(ts)


def _extract_crafting_goal(input_text: str):
    """Split a TextCraft state into (commands, goal). Returns (full_text, None) if no Goal: found."""
    ########## @Raunaq This should be updated ################
    parts = input_text.strip().rsplit("Goal:", 1)
    if len(parts) < 2:
        return input_text.strip(), None
    return parts[0].strip(), parts[1].strip().rstrip(".")


# ---------------------------------------------------------------------------
# Return-to-go
# ---------------------------------------------------------------------------

def returns_to_g(rewards: list, gamma: float = 1.0) -> list:
    """
    Compute discounted return-to-go for each step.
    Handles intermediate rewards by converting to incremental deltas first.
    """
    diff_rewards = rewards.copy()
    for i in range(1, len(rewards)):
        if rewards[i] < 0:
            diff_rewards[i] = rewards[i]
        else:
            diff_rewards[i] = rewards[i] - rewards[i - 1]

    rtg = [0.0] * len(rewards)
    for g_t in range(len(rtg)):
        if g_t == len(rtg) - 1:
            rtg[g_t] = diff_rewards[-1]
        else:
            for r_t in range(g_t + 1, len(diff_rewards)):
                rtg[g_t] += round(pow(gamma, r_t - g_t) * diff_rewards[r_t], 4)
    return rtg


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_value_prompts(
    states_raw: list,
    actions_taken: list,
    dataset: str,
    preamble: str,
    K: int = 5,
) -> list:
    """
    Build one value-function prompt per step.

    Args:
        states_raw:    list of raw environment observation strings (length T).
        actions_taken: list of parsed action strings (length T).
        dataset:       one of webshop | sciworld | textcraft | movie | weather.
        preamble:      the introductory preamble string (from PREAMBLES).
        K:             maximum number of (state, action) history pairs to include.

    Returns:
        list of {"t": int, "input_text": str} dicts, one per step.
    """
    T = len(states_raw)

    # --- extract instruction and pre-process states per dataset ---
    if dataset == "webshop":
        instruction = _extract_instruction_webshop(states_raw[0]) or "(none found)"
        stripped = [_strip_to_after_instruction(s) for s in states_raw]

    elif dataset == "sciworld":
        newline_pos = states_raw[0].find("\n")
        instruction = states_raw[0][:newline_pos]
        stripped = [states_raw[0][newline_pos + 1:]] + states_raw[1:]

    elif dataset == "textcraft":
        ## @Raunaq please update this
        _, instruction = _extract_crafting_goal(states_raw[0])
        instruction = instruction or "(none found)"
        stripped = states_raw  # commands extracted per-step below

    elif dataset in ("movie", "weather"):
        start_marker = "You should perform actions to accomplish the goal: "
        end_marker = "Give me one action."
        instruction = states_raw[0][
            states_raw[0].find(start_marker) + len(start_marker):
            states_raw[0].find(end_marker)
        ]
        stripped = [end_marker] + states_raw[1:]

    else:
        raise ValueError(f"Unsupported dataset: {dataset!r}")

    # --- build one prompt per time step ---
    rows = []
    for t in range(1, T + 1):
        curr_idx = t - 1
        hist_start = max(0, curr_idx - K)
        hist_indices = range(hist_start, curr_idx)

        parts = [preamble, "\n=== INSTRUCTION ===", instruction, "\n=== HISTORY ==="]

        if not hist_indices:
            parts.append("(none)")
        else:
            for i in hist_indices:
                if dataset == "textcraft":
                    ## @Raunaq please update this
                    state_i, _ = _extract_crafting_goal(stripped[i])
                else:
                    state_i = stripped[i]
                action_i = actions_taken[i] if i < len(actions_taken) and actions_taken[i] else "(unknown)"
                parts.append(f"Round {i + 1} — STATE:\n{state_i}")
                parts.append(f"Round {i + 1} — ACTION:\n{action_i}")

        parts.append("\n=== NOW ===")
        if dataset == "textcraft":
            ## @Raunaq please update this
            current_state, _ = _extract_crafting_goal(stripped[curr_idx])
            current_state = re.sub(r"\[SEP\] Reward.*", "", current_state, flags=re.DOTALL).strip()
        else:
            current_state = stripped[curr_idx]
            current_state = re.sub(r"\[SEP\] Reward.*", "", current_state, flags=re.DOTALL).strip()
        parts.append(f"[STATE] {current_state}")

        rows.append({"t": t, "input_text": "\n".join(parts)})

    return rows


# ---------------------------------------------------------------------------
# Trajectory parsing
# ---------------------------------------------------------------------------

def _extract_action(gpt_message: str) -> str:
    """Extract the action text after 'Action:' in a model output string."""
    idx = gpt_message.find("Action:")
    if idx == -1:
        return ""
    return gpt_message[idx + len("Action:"):].lstrip().splitlines()[0].strip()


def parse_trajectory(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """
    Expand the raw trajectory dataframe into per-step state/action/reward columns.
    """
    # state_split: human observations, skipping index 0 (system prompt)
    df["state_split"] = df["state"].apply(
        lambda conv: [msg["value"] for msg in conv if msg["from"] == "human"][1:]
    )
    # action_split: gpt outputs, skipping index 0 ("Ok." acknowledgement)
    df["action_split"] = df["state"].apply(
        lambda conv: [_extract_action(msg["value"]) for msg in conv if msg["from"] == "gpt"][1:]
    )

    # For webshop: force_buy_on_extra_step appends a final human observation with no
    # corresponding gpt action. Drop any trailing states beyond len(action_split) + 1
    # so that every state in state_split has an associated action (or is the initial state).
    if dataset == "webshop":
        df["state_split"] = df.apply(
            lambda row: row["state_split"][:len(row["action_split"]) + 1], axis=1
        )

    # reward_split: per-step rewards where available, else final reward only
    if dataset == "webshop":
        # no per-step rewards; sparse final reward at the last step
        df["reward_split"] = df.apply(
            lambda row: [0.0] * (len(row["action_split"]) - 1) + [row["reward"]], axis=1
        )
    elif dataset == "textcraft":
        df["reward_split"] = df.apply(
            lambda row: [0.0] * (len(row["action_split"]) - 1) + [row["reward"]], axis=1
        )
    else:
        # sciworld, movie, weather store reward in each human message
        df["reward_split"] = df["state"].apply(
            lambda conv: [
                msg["reward"] for msg in conv
                if msg.get("from") == "human" and "reward" in msg
            ]
        )

    # prepend a 0.0 for the initial state (before any action)
    df["reward_split"] = df["reward_split"].apply(lambda x: [0.0] + x)

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert SMC trajectory JSON into a HuggingFace dataset for VF training."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the trajectory JSON file produced by main.py --mode train."
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=["webshop", "sciworld", "textcraft", "movie", "weather"],
        help="Environment the trajectories were collected in."
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory for the HuggingFace dataset. "
             "Defaults to <input_path_without_.json>-train-value."
    )
    parser.add_argument(
        "--history-length", type=int, default=5,
        help="Maximum number of (state, action) pairs to include in the prompt history (default: 5)."
    )
    parser.add_argument(
        "--gamma", type=float, default=1.0,
        help="Discount factor for return-to-go computation (default: 1.0)."
    )
    parser.add_argument(
        "--valid-ratio", type=float, default=0.2,
        help="Fraction of trajectories to hold out as validation set (default: 0.2)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/validation split (default: 42)."
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = args.output or args.input.replace(".json", "") + "-train-value"
    preamble = PREAMBLES[args.dataset]

    # --- load trajectories ---
    print(f"Loading trajectories from: {args.input}")
    traj = pd.read_json(args.input)
    traj.rename(columns={"task_id": "idx"}, inplace=True)

    # take the best trajectory per task (rank 1)
    traj["state"] = traj["trajectories"].apply(lambda x: x[0]["history"])
    traj["reward"] = traj["trajectories"].apply(lambda x: x[0]["reward"])
    print(f"Loaded {len(traj)} trajectories. Average reward: {traj['reward'].mean():.4f}")

    # --- parse into per-step columns ---
    traj = parse_trajectory(traj, args.dataset)

    # --- compute return-to-go ---
    traj["g_reward_split"] = traj["reward_split"].apply(
        lambda x: returns_to_g(x, gamma=args.gamma)
    )

    # --- train / validation split ---
    valid_indices = set(
        random.sample(range(len(traj)), int(len(traj) * args.valid_ratio))
    )

    train_rows, valid_rows = [], []

    for i, sample in traj.iterrows():
        states = sample["state_split"]
        actions = sample["action_split"]
        rewards = sample["g_reward_split"]
        task_idx = int(sample["idx"])

        prompts = build_value_prompts(
            states, actions,
            dataset=args.dataset,
            preamble=preamble,
            K=args.history_length,
        )

        target = train_rows if i not in valid_indices else valid_rows
        for j, entry in enumerate(prompts):
            target.append({
                "idx":    task_idx,
                "t":      entry["t"],
                "text":   entry["input_text"],
                "reward": rewards[j],
            })

    print(f"Train examples: {len(train_rows)} | Validation examples: {len(valid_rows)}")

    # --- save ---
    ds = DatasetDict({
        "train":      Dataset.from_list(train_rows),
        "validation": Dataset.from_list(valid_rows),
    })
    ds.save_to_disk(output_dir)
    print(f"Dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()