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


def construct_cpp_variable(ty, ptr):
    variable = f"(*(reinterpret_cast<{ty}*>(0x{ptr:x}ull)))"
    print(f"Variable: {variable}")
    return variable


program_code = """jitify2::Program("{name}", R"({source})")"""

preprocess_program_code = """\
auto x = {program}->preprocess(); x"""

test_kernel_code = """\
float *data = reinterpret_cast<float*>(0x{data:x}ull);
{program}->preprocess()->get_kernel("my_kernel")->configure(1, 1)->launch(data);"""


class Program:
    def __init__(self, name, source):
        code = program_code.format(name=name, source=source)
        print(f"Program code:\n\n{code}")
        self._value = get_interpreter().parse_and_execute(code)

    @property
    def ptr(self):
        return self._value[0].Data.m_ULongLong

    @functools.cached_property
    def variable(self):
        return construct_cpp_variable("jitify2::Program", self.ptr)

    def preprocess(self):
        code = preprocess_program_code.format(program=self.variable)
        print(f"Preprocess program code:\n\n{code}")
        value = get_interpreter().parse_and_execute(code)
        return PreprocessedProgram(value)

    def test_kernel(self, data):
        code = test_kernel_code.format(program=self.variable, data=data)
        print(f"Preprocess program code:\n\n{code}")
        get_interpreter().parse_and_execute(code)


class PreprocessedProgram:
    def __init__(self, value):
        self._value = value

    @property
    def ptr(self):
        return self._value[0].Data.m_ULongLong


__all__ = (
    'create_interpreter',
    'delete_interpreter',
    'llvm_shutdown',
    'load_dynamic_library'
    'parse_and_execute',
)
