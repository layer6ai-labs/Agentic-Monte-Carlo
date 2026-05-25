# Agentic Monte Carlo: Simulating Reinforcement Learning for Black-Box Agents

### Initializing
1. Install uv https://docs.astral.sh/uv/getting-started/installation/. See the `uv` primer below for details on how to use it.

2. Run the setup script to install the main Python environment and set up the environment server:
```
./scripts/setup.sh
```

> **Note:** Each environment server (SciWorld, TextCraft, Movie, Weather) requires its own separate conda environment and setup steps, detailed in the sections below. The Webshop environment is the only one fully configured by `setup.sh`. The main `uv` environment (used to run `main.py`) is shared across all environments.


---

## Running Experiments (`main.py`)

All trajectory collection and evaluation runs go through `main.py`. Key arguments:

| Argument | Description |
|---|---|
| `--env` | Environment: `webshop`, `sciworld`, `textcraft`, `movie`, `weather` |
| `--mode` | `train` (collect trajectories) or `test` (evaluate) |
| `--n-particles` | Number of parallel agent particles |
| `--max-steps` | Maximum steps per task |
| `--resample-steps` | Fixed steps to resample at (omit to use adaptive ESS) |
| `--model-path` | Policy model (HuggingFace path or API model name) |
| `--api-key` / `--api-base-url` | Use an OpenAI-compatible API model instead of a local model |
| `--template_version` | Value function (VF) prompt template matching the environment |
| `--ess-threshold` | ESS fraction threshold for adaptive resampling (default: 0.5) |

**Value function modes** (controls how particles are weighted during resampling):

- **No value function** (ReAct / Best-of-N): omit all `--vf-*` flags
- **SMC Zero-shot**: `--vf-as-generator --vf-base-model <model> --template_version <env>`
- **AMC** (trained value function): `--vf-ckpt-path <ckpt> --vf-base-model <model> --template_version <env>`


---

## Converting Trajectories to Training Data (`smc/traj_train_convert.py`)

After collecting trajectories with `--mode train`, convert them into a HuggingFace dataset before training the value function:

```
uv run python smc/traj_train_convert.py \
  --input   results/[trajectory_file].json \
  --dataset [webshop|sciworld|textcraft|movie|weather] \
  --output  data/[output_dataset_dir]
```

Key options:
- `--history-length` — past (state, action) pairs to include in the prompt. Values used in the paper: webshop=10, sciworld=20, textcraft=20, movie=12, weather=10
- `--gamma` — discount factor for return-to-go (default: 1.0)
- `--valid-ratio` — fraction of trajectories held out for validation (default: 0.2)


---

## Training the Value Function (`smc/train_lora_regression.py`)

Train a LoRA + regression head on the converted dataset. Run from the repo root with:

```
torchrun --nproc_per_node=<N_GPUS> smc/train_lora_regression.py \
  --model_name      <base_model> \
  --input_dir       <converted_dataset_dir> \
  --output_dir      <checkpoint_output_dir> \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --gradient_checkpointing \
  --max_grad_norm 1 \
  --loss_function mse
```

Additional options:
```
  --early_stopping_patience 1 \
  --metric_for_best_model pearsonr_y \
  --loss_weight [Ratio of samples with reward > 0 to samples with reward <= 0 in the training data]
```

Run `mlflow server --port 8080` to track training progress where the `mlruns` folder is located.


---

## Webshop Environment

Setup is handled by `./scripts/setup.sh`.

1. Run the webshop environment:
```
conda activate agentenv-webshop
cd envs/AgentGym/agentenv-webshop/
webshop --host 0.0.0.0 --port 36001
```

2. Generate training trajectories:
```
uv run python main.py --env webshop --mode train --n-particles 3 --max-steps 10 --model-path meta-llama/Llama-3.2-11B-Vision-Instruct --port 36001
```
3. Convert trajectories and train the value function:
```
uv run python smc/traj_train_convert.py --input results/[trajectory_file].json --dataset webshop --history-length 10 --output data/webshop-train-value
```
```
torchrun --nproc_per_node=<N_GPUS> smc/train_lora_regression.py \
  --model_name meta-llama/Llama-3.2-11B-Vision-Instruct \
  --input_dir  data/webshop-train-value \
  --output_dir checkpoints/webshop-vf \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --gradient_checkpointing \
  --max_grad_norm 1 \
  --loss_function mse
```

4. Run Evaluation:

ReAct:
```
uv run python main.py --env webshop --mode test --n-particles 1 --max-steps 10 --model-path meta-llama/Llama-3.2-11B-Vision-Instruct
```

Best-of-15 (ReAct):
```
uv run python main.py --env webshop --mode test --n-particles 15 --max-steps 10 --model-path meta-llama/Llama-3.2-11B-Vision-Instruct
```

SMC (FoA-ReAct):
```
uv run python main.py --env webshop --mode test --n-particles 15 --max-steps 10 --resample-steps 6 --model-path meta-llama/Llama-3.2-11B-Vision-Instruct --vf-base-model meta-llama/Llama-3.2-11B-Vision-Instruct --vf-as-generator --foa --template_version WEBSHOP
```

SMC (Zero-shot):
```
uv run python main.py --env webshop --mode test --n-particles 15 --max-steps 10 --resample-steps 6 --model-path meta-llama/Llama-3.2-11B-Vision-Instruct --vf-base-model meta-llama/Llama-3.2-11B-Vision-Instruct --vf-as-generator --template_version WEBSHOP
```

AMC (ReAct):
```
uv run python main.py --env webshop --mode test --n-particles 15 --max-steps 10 --resample-steps 6 --model-path meta-llama/Llama-3.2-11B-Vision-Instruct --vf-ckpt-path [Trained Value Function Checkpoint] --vf-base-model meta-llama/Llama-3.2-11B-Vision-Instruct --template_version WEBSHOP
```

> For API models, replace `--model-path meta-llama/Llama-3.2-11B-Vision-Instruct` with `--model-path gpt-4.1-mini-2025-04-14 --api-key [OpenAI key] --api-base-url https://api.openai.com/v1`


## SciWorld Environment

1. Set up the SciWorld environment (requires Java 1.8+ and conda):
```
conda create --name agentenv-sciworld python=3.8
conda activate agentenv-sciworld
cd envs/AgentGym/agentenv-sciworld/
pip install -e .
```

2. Run the sciworld environment:
```
conda activate agentenv-sciworld
cd envs/AgentGym/agentenv-sciworld/
sciworld --host 0.0.0.0 --port 36001
```

3. Generate training trajectories:
```
uv run python main.py --env sciworld --mode train --n-particles 3 --max-steps 20 --model-path meta-llama/Llama-3.1-8B-Instruct --port 36001
```

4. Convert trajectories and train the value function:
```
uv run python smc/traj_train_convert.py --input results/[trajectory_file].json --dataset sciworld --history-length 20 --output data/sciworld-train-value
```
```
torchrun --nproc_per_node=<N_GPUS> smc/train_lora_regression.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --input_dir  data/sciworld-train-value \
  --output_dir checkpoints/sciworld-vf \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --gradient_checkpointing \
  --max_grad_norm 1 \
  --loss_function mse
```

5. Run Evaluation:

ReAct:
```
uv run python main.py --env sciworld --mode test --n-particles 1 --max-steps 20 --model-path meta-llama/Llama-3.1-8B-Instruct
```

Best-of-15 (ReAct):
```
uv run python main.py --env sciworld --mode test --n-particles 15 --max-steps 20 --model-path meta-llama/Llama-3.1-8B-Instruct
```

Best-of-15 (ReflAct):
```
uv run python main.py --env sciworld --mode test --n-particles 15 --max-steps 20 --model-path meta-llama/Llama-3.1-8B-Instruct --policy-prompt reflact
```

SMC (Zero-shot):
```
uv run python main.py --env sciworld --mode test --n-particles 15 --max-steps 20 --resample-steps 4 12 --model-path meta-llama/Llama-3.1-8B-Instruct --vf-base-model meta-llama/Llama-3.1-8B-Instruct --vf-as-generator --template_version SCIWORLD
```

AMC (ReAct):
```
uv run python main.py --env sciworld --mode test --n-particles 15 --max-steps 20 --resample-steps 4 12 --model-path meta-llama/Llama-3.1-8B-Instruct --vf-ckpt-path [Trained Value Function CKPT] --vf-base-model meta-llama/Llama-3.1-8B-Instruct --template_version SCIWORLD
```

> For API models, replace `--model-path meta-llama/Llama-3.1-8B-Instruct` with `--model-path gpt-4.1-mini-2025-04-14 --api-key [OpenAI key] --api-base-url https://api.openai.com/v1`


## TextCraft Environment

1. Set up the TextCraft environment (requires conda):
```
conda create --name agentenv-textcraft python=3.9
conda activate agentenv-textcraft
cd envs/AgentGym/agentenv-textcraft/
pip install -e .
```

2. Run the textcraft environment:
```
conda activate agentenv-textcraft
cd envs/AgentGym/agentenv-textcraft/
textcraft --host 0.0.0.0 --port 36001
```

3. Generate training trajectories:
```
uv run python main.py --env textcraft --mode train --n-particles 8 --max-steps 20 --model-path meta-llama/Llama-3.2-11B-Vision-Instruct --port 36001
```

4. Convert trajectories and train the value function:
```
uv run python smc/traj_train_convert.py --input results/[trajectory_file].json --dataset textcraft --history-length 20 --output data/textcraft-train-value
```
```
torchrun --nproc_per_node=<N_GPUS> smc/train_lora_regression.py \
  --model_name meta-llama/Llama-3.2-11B-Vision-Instruct \
  --input_dir  data/textcraft-train-value \
  --output_dir checkpoints/textcraft-vf \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --gradient_checkpointing \
  --max_grad_norm 1 \
  --loss_function mse
```

5. Run Evaluation:

ReAct:
```
uv run python main.py --env textcraft --mode test --n-particles 1 --max-steps 20 --model-path meta-llama/Llama-3.2-11B-Vision-Instruct
```

Best-of-15 (ReAct):
```
uv run python main.py --env textcraft --mode test --n-particles 15 --max-steps 20 --model-path meta-llama/Llama-3.2-11B-Vision-Instruct
```

SMC (Zero-shot):
```
uv run python main.py --env textcraft --mode test --n-particles 15 --max-steps 20 --resample-steps 4 --model-path meta-llama/Llama-3.2-11B-Vision-Instruct --vf-base-model meta-llama/Llama-3.2-11B-Vision-Instruct --vf-as-generator --template_version TEXTCRAFT
```

AMC (ReAct):
```
uv run python main.py --env textcraft --mode test --n-particles 15 --max-steps 20 --resample-steps 4 --model-path meta-llama/Llama-3.2-11B-Vision-Instruct --vf-ckpt-path [Trained Value Function CKPT] --vf-base-model meta-llama/Llama-3.2-11B-Vision-Instruct  --template_version TEXTCRAFT
```


## Movie Environment

Movie and Weather environments are both served from the `agentenv-tool` package. Set it up once and use it for both.

1. Obtain a TMDB API key at https://developer.themoviedb.org/docs/getting-started, then set it in `envs/AgentGym/agentenv-tool/setup.sh`:
```
export MOVIE_KEY="your_tmdb_api_key"
```

2. Set up the tool environment (requires conda):
```
conda create --name agentenv-tool python=3.8
conda activate agentenv-tool
cd envs/AgentGym/agentenv-tool/
source ./setup.sh
```

3. Run the movie environment:
```
conda activate agentenv-tool
cd envs/AgentGym/agentenv-tool/
movie --host 0.0.0.0 --port 36001
```

4. Generate training trajectories:
```
uv run python main.py --env movie --mode train --n-particles 3 --max-steps 12 --model-path meta-llama/Llama-3.2-11B-Vision-Instruct --port 36001
```

5. Convert trajectories and train the value function:
```
uv run python smc/traj_train_convert.py --input results/[trajectory_file].json --dataset movie --history-length 12 --output data/movie-train-value
```
```
torchrun --nproc_per_node=<N_GPUS> smc/train_lora_regression.py \
  --model_name meta-llama/Llama-3.2-11B-Vision-Instruct \
  --input_dir  data/movie-train-value \
  --output_dir checkpoints/movie-vf \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --gradient_checkpointing \
  --max_grad_norm 1 \
  --loss_function mse
```

6. Run Evaluation:

ReAct:
```
uv run python main.py --env movie --mode test --n-particles 1 --max-steps 12 --model-path meta-llama/Llama-3.2-11B-Vision-Instruct --port 36001
```

Best-of-15 (ReAct):
```
uv run python main.py --env movie --mode test --n-particles 15 --max-steps 12 --model-path meta-llama/Llama-3.2-11B-Vision-Instruct --port 36001
```

SMC (Zero-shot):
```
uv run python main.py --env movie --mode test --n-particles 15 --max-steps 12 --resample-steps 6 --model-path meta-llama/Llama-3.2-11B-Vision-Instruct --vf-base-model meta-llama/Llama-3.2-11B-Vision-Instruct --vf-as-generator --template_version MOVIE --port 36001
```

AMC (ReAct):
```
uv run python main.py --env movie --mode test --n-particles 15 --max-steps 12 --resample-steps 6 --model-path meta-llama/Llama-3.2-11B-Vision-Instruct --vf-ckpt-path [Trained Value Function CKPT] --vf-base-model meta-llama/Llama-3.2-11B-Vision-Instruct --template_version MOVIE --port 36001
```


## Weather Environment

Weather uses the same `agentenv-tool` conda environment as Movie (no API key required — data comes from open-meteo).

1. If not already done, complete steps 1–2 from the Movie Environment section above to set up `agentenv-tool`.

2. Run the weather environment:
```
conda activate agentenv-tool
cd envs/AgentGym/agentenv-tool/
weather --host 0.0.0.0 --port 36001
```

3. Generate training trajectories:
```
uv run python main.py --env weather --mode train --n-particles 3 --max-steps 10 --model-path meta-llama/Llama-3.2-11B-Vision-Instruct --port 36001
```

4. Convert trajectories and train the value function:
```
uv run python smc/traj_train_convert.py --input results/[trajectory_file].json --dataset weather --history-length 10 --output data/weather-train-value
```
```
torchrun --nproc_per_node=<N_GPUS> smc/train_lora_regression.py \
  --model_name meta-llama/Llama-3.2-11B-Vision-Instruct \
  --input_dir  data/weather-train-value \
  --output_dir checkpoints/weather-vf \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --gradient_checkpointing \
  --max_grad_norm 1 \
  --loss_function mse
```

5. Run Evaluation:

ReAct:
```
uv run python main.py --env weather --mode test --n-particles 1 --max-steps 10 --model-path meta-llama/Llama-3.2-11B-Vision-Instruct --port 36001
```

Best-of-15 (ReAct):
```
uv run python main.py --env weather --mode test --n-particles 15 --max-steps 10 --model-path meta-llama/Llama-3.2-11B-Vision-Instruct --port 36001
```

SMC (Zero-shot):
```
uv run python main.py --env weather --mode test --n-particles 15 --max-steps 10 --resample-steps 4 --model-path meta-llama/Llama-3.2-11B-Vision-Instruct --vf-base-model meta-llama/Llama-3.2-11B-Vision-Instruct --vf-as-generator --template_version WEATHER --port 36001
```

AMC (ReAct):
```
uv run python main.py --env weather --mode test --n-particles 15 --max-steps 10 --resample-steps 4 --model-path meta-llama/Llama-3.2-11B-Vision-Instruct --vf-ckpt-path [Trained Value Function CKPT] --vf-base-model meta-llama/Llama-3.2-11B-Vision-Instruct --template_version WEATHER --port 36001
```


---

### uv primer
1. `uv` keeps an env at `.venv` and package info in `pyproject.toml` and `uv.lock`. To sync your local environment to the directory's requirements, run the following (already done by scripts/setup.sh.)
```
uv sync
```

2. To install a new package, run the following, which will automatically update `.venv`, `pyproject.toml`, and `uv.lock`.
```
uv add <package>
```

3. When running a python script or other command, use the local `.venv` by running
```
uv run <command>
```
