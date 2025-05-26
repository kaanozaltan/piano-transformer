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
uv pip install -r requirements.txt

Set environment variables:
```bash
export TORCH_CUDA_ARCH_LIST="6.1 8.6 8.9"
module load CUDA/12.3
export CXX=g++
export CC=gcc
export FORCE_CUDA=1
export PYTHONPATH=$PWD
```
