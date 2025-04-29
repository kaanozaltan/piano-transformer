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
uv pip install \
    torch==2.2.2+cu121 \
    torchvision==0.17.2+cu121 \
    torchaudio==2.2.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
```

Set environment variables:
```bash
export TORCH_CUDA_ARCH_LIST="6.1 8.6 8.9"
module load CUDA/12.3
export CXX=g++
export CC=gcc
export FORCE_CUDA=1
export PYTHONPATH=$PWD
```