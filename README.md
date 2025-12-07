# Piano Transformer

## Check out the music samples used in our Turing-like test here:
https://drive.google.com/drive/folders/1F_HzY3lxXqOzP_PXCYyZM_5SipklndYb

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
