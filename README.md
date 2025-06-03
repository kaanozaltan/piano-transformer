# Piano Transformer

## Environment Setup
Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create environment:
```bash
uv venv --python 3.11
source .venv/bin/activate
```

Install dependencies:
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
uv pip install -e .
```

Set environment variables:
```bash
export CXX=g++
export CC=gcc
export FORCE_CUDA=1
export PYTHONPATH=$PWD/src
```
