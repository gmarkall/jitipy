# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from jitipy import clanginterpreter
import functools


def create_interpreter():
    return clanginterpreter.create_interpreter()


def delete_interpreter(interpreter):
    clanginterpreter.delete_interpreter(interpreter)


def llvm_shutdown():
    clanginterpreter.llvm_shutdown()


def parse_and_execute(interpreter, code):
    return clanginterpreter.parse_and_execute(interpreter, code)


def load_dynamic_library(interpreter, name):
    clanginterpreter.load_dynamic_library(interpreter, name)


@functools.cache
def get_interpreter():
    """
    Call this to create the interpreter eagerly if desired. Creating
    the interpreter carries a small startup cost and also initializes CUDA.

    :return: The interpreter singleton object.
    """
    return Interpreter()


class Interpreter:
    def __init__(self):
        self._interpreter = clanginterpreter.create_interpreter()

        # FIXME: This is probably needed because it works around loading the
        # wrong library or some sort of clash. Without it, jitify encounters a
        # CUDA initialization error.
        self.parse_and_execute(["#include <cuda.h>", "cudaSetDevice(0);"])

        # Load jitify2
        self.parse_and_execute('#include "jitipy/jitify2.hpp"')

    def __del__(self):
        delete_interpreter(self._interpreter)

    def parse_and_execute(self, code):
        return parse_and_execute(self._interpreter, code)


program_code = """jitify2::Program("{name}", R"({source})")"""


class Program:
    def __init__(self, name, source):
        code = program_code.format(name=name, source=source)
        print(f"Program code:\n\n{code}")
        self._program = get_interpreter().parse_and_execute(code)


__all__ = (
    'create_interpreter',
    'delete_interpreter',
    'llvm_shutdown',
    'load_dynamic_library'
    'parse_and_execute',
)
