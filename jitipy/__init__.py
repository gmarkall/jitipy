# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from jitipy import clanginterpreter


def create_interpreter():
    return clanginterpreter.create_interpreter()


def delete_interpreter(interpreter):
    clanginterpreter.delete_interpreter(interpreter)


def llvm_shutdown():
    clanginterpreter.llvm_shutdown()


__all__ = (
    'create_interpreter',
    'delete_interpreter',
    'llvm_shutdown',
)
