# Copyright (c) 2023, NVIDIA Corporation

from skbuild import setup

setup(
    name='jitipy',
    version='0.0.1',
    description='JIT compilation of CUDA C/C++ using jitify',
    author='Graham Markall (NVIDIA)',
    license='BSD 3-clause',
    packages=['jitipy'],
    python_requires=">=3.8",
)
