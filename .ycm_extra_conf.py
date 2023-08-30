import os
import subprocess
import sys

from pathlib import Path

CONDA_PREFIX = os.environ['CONDA_PREFIX']
VER = sys.version_info
PYTHON_INCLUDE_NAME = f'python{VER.major}.{VER.minor}'
CONDA_INCLUDE_DIR = Path(CONDA_PREFIX, 'include')
PYTHON_INCLUDE_DIR = Path(CONDA_INCLUDE_DIR, PYTHON_INCLUDE_NAME)
CUDA_INCLUDE_DIR = '/usr/local/cuda/include'

cp = subprocess.run(['llvm-config', '--includedir'], capture_output=True)
LLVM_INCLUDE_DIR = cp.stdout.decode().strip()

flags = [
    '--cuda-gpu-arch=sm_50',
    f'-I{CONDA_INCLUDE_DIR}',
    f'-I{CUDA_INCLUDE_DIR}',
    f'-I{PYTHON_INCLUDE_DIR}',
    f'-I{LLVM_INCLUDE_DIR}',
]


def Settings(**kwargs):
    return {'flags': flags}


if __name__ == '__main__':
    print(Settings())
