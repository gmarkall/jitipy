# Copyright (c) 2023, NVIDIA CORPORATION

from setuptools import setup, Extension
import subprocess

cp = subprocess.run(['llvm-config', '--cflags'], capture_output=True)
llvm_compile_args = cp.stdout.split()

cp = subprocess.run(['llvm-config', '--ldflags'], capture_output=True)
llvm_link_args = cp.stdout.split()

cp = subprocess.run(['llvm-config', '--libs'], capture_output=True)
llvm_libs = cp.stdout.split()

clang_link_args = ['-lclangAST', '-lclangBasic', '-lclangFrontend',
                   '-lclangInterpreter']


lib = Extension(
    'jitipy._lib',
    sources=['jitipy/_lib.cpp'],
    extra_compile_args=['-Wall', '-std=c++17'] + llvm_compile_args,
    extra_link_args=llvm_link_args + llvm_libs + clang_link_args,
)


setup(
    name='jitipy',
    version='0.0.1',
    description='Jitify for Python',
    ext_modules=[lib],
    packages=['jitipy'],
)
