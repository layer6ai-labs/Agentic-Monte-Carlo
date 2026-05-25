# Agent Environments - Webshop

## Setup

``` sh
sudo apt update
sudo apt install -y g++ default-jdk

uv sync
uv run python -m spacy download en_core_web_lg
```

## Indexing
```sh
cd webshop/search_engine/
mkdir -p resources resources_100 resources_1k resources_100k
uv run python convert_product_file_format.py
uv run bash ./run_indexing.sh
```

## Launch
``` sh
uv run webshop --host 0.0.0.0 --port 36001
```
