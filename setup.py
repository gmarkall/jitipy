# Copyright (c) 2023, NVIDIA CORPORATION

import os
import shutil
from distutils.sysconfig import get_config_var, get_python_inc
from setuptools import setup, Extension

include_dirs = [os.path.dirname(get_python_inc())]
library_dirs = [get_config_var("LIBDIR")]

# Find and add CUDA include paths
CUDA_HOME = os.environ.get("CUDA_HOME", False)
if not CUDA_HOME:
    path_to_cuda_gdb = shutil.which("cuda-gdb")
    if path_to_cuda_gdb is None:
        raise OSError(
            "Could not locate CUDA. "
            "Please set the environment variable "
            "CUDA_HOME to the path to the CUDA installation "
            "and try again."
        )
    CUDA_HOME = os.path.dirname(os.path.dirname(path_to_cuda_gdb))

if not os.path.isdir(CUDA_HOME):
    raise OSError(f"Invalid CUDA_HOME: directory does not exist: {CUDA_HOME}")

include_dirs.append(os.path.join(CUDA_HOME, "include"))
library_dirs.append(os.path.join(CUDA_HOME, "lib64"))

lib = Extension(
    'jitipy._lib',
    sources=['jitipy/_lib.cpp'],
    libraries=['cuda', 'cudart', 'nvrtc'],
    extra_compile_args=['-Wall', '-Werror', '-pthread'],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
)


setup(
    name='jitipy',
    version='0.0.1',
    description='Jitify for Python',
    ext_modules=[lib],
    packages=['jitipy'],
)
