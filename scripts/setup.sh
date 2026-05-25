#!/bin/bash

ARROW="\033[1;34m==>\033[0m"
BOLD="\033[1m"
RESET="\033[0m"

print_step() {
    # Helper function for outputs to user
    echo -e "\n${ARROW} ${BOLD}$1${RESET}"
}

cd "$(dirname "$0")/.." # Change to script's parent directory

print_step "Syncing uv environment with repo"
uv sync

print_step "Installing system dependencies (sudo required)"
sudo apt update
sudo apt install -y g++ default-jdk

print_step "Setting up Webshop environment"
cd envs/AgentGym/agentenv-webshop/
uv sync
uv run python -m spacy download en_core_web_lg
cd webshop/search_engine/
mkdir -p resources resources_100 resources_1k resources_100k
uv run python convert_product_file_format.py
uv run bash ./run_indexing.sh
