# ja-NCM training and evaluation image.
#
# Matches the environment the ja-NCM runs used: Ubuntu 24.04, Python 3.12, torch 2.14.1 from PyPI
# (its default wheel bundles CUDA 13.0 and cuDNN, so the image needs no CUDA toolkit), on H100s.
# The host needs an NVIDIA driver that supports CUDA 13.0 (R580 or newer) and the NVIDIA
# Container Toolkit.
#
#   docker build -t ja-ncm .
#   docker run --gpus all --ipc=host --rm -it -v $PWD/data:/work/data -v $PWD/runs:/work/runs ja-ncm
#   # inside: python -m torch.distributed.run --standalone --nproc_per_node 8 scripts/train_ncm.py --config ...
#
# --ipc=host (or a large --shm-size) is needed for multi-GPU NCCL and data loading.
FROM nvidia/cuda:13.0.0-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=8 \
    # NVLink SHARP multicast failed to bind for a second torchrun job on H100 nodes; plain NVLink
    # rings are enough for models of this size.
    NCCL_NVLS_ENABLE=0 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev build-essential git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

WORKDIR /work

# Dependencies first so code edits do not reinstall torch.
COPY requirements.txt requirements-ja.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt -r requirements-ja.txt

COPY . .
RUN pip install --no-deps -e . \
    && python -c "import jancm, models.next_concept; print('ja-ncm import ok')"

# Data, checkpoints and results are expected on mounted volumes (see the docker run line above).
CMD ["bash"]
