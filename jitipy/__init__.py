# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from jitipy import clanginterpreter
import functools
import itertools

_variable_indices = itertools.count()


def _new_variable():
    return f'__jitipy_var_{next(_variable_indices)}'


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
    return variable


program_code = """jitify2::Program("{name}", R"({source})")"""

preprocess_program_code = """\
auto {var} = {jitify_object}->preprocess(); {var}"""

get_kernel_code = """\
auto {var} = {jitify_object}->get_kernel("{name}"); {var}"""

configure_code = """\
auto {var} = {jitify_object}->configure({grid_dim}, {block_dim}); {var}"""

launch_code = """\
{jitify_object}->launch({args});"""

jitipy_method_code = """\
auto {var} = {jitify_object}->{method}({args}); {var}"""


def stringify_args(args):
    new_args = []

    for arg in args:
        if isinstance(arg, str):
            arg = f'"{arg}"'
        new_args.append(arg)

    return new_args


class JitifyObject:
    @property
    def ptr(self):
        return self._value[0].Data.m_ULongLong

    @property
    def ty(self):
        return self._ty

    @functools.cached_property
    def variable(self):
        return construct_cpp_variable(self.ty, self.ptr)

    def __getattr__(self, fname):
        if fname == "_functions":
            return {}

        if fname not in self._functions:
            raise AttributeError

        jitify_object = self.variable
        return_type = self._functions[fname]

        def method(*args):
            args = ", ".join(stringi::fy_args(args))
            code = jitipy_method_code.format(jitify_object=jitify_object,
                                             var=_new_variable(),
                                             args=args,
                                             method=fname)
            value = get_interpreter().parse_and_execute(code)
            return return_type(value)

        return method


class Program(JitifyObject):
    def __init__(self, name, source):
        code = program_code.format(name=name, source=source)
        self._value = get_interpreter().parse_and_execute(code)
        self._ty = "jitify2::Program"

    def preprocess(self):
        code = preprocess_program_code.format(jitify_object=self.variable,
                                              var=_new_variable())
        value = get_interpreter().parse_and_execute(code)
        return PreprocessedProgram(value)


class PreprocessedProgram(JitifyObject):
    def __init__(self, value):
        self._value = value
        self._ty = "jitify2::PreprocessedProgram"
        self._functions = {'get_kernel': Kernel}

#    def get_kernel(self, name):
#        code = get_kernel_code.format(jitify_object=self.variable, name=name,
#                                      var=_new_variable())
#        value = get_interpreter().parse_and_execute(code)
#        return Kernel(value)


class Kernel(JitifyObject):
    def __init__(self, value):
        self._value = value
        self._ty = "jitify2::Kernel"

    def configure(self, grid_dim, block_dim):
        code = configure_code.format(jitify_object=self.variable,
                                     grid_dim=grid_dim,
                                     block_dim=block_dim,
                                     var=_new_variable())
        value = get_interpreter().parse_and_execute(code)
        return ConfiguredKernel(value)


class ConfiguredKernel(JitifyObject):

    def __init__(self, value):
        self._value = value
        self._ty = "jitify2::ConfiguredKernel"

    def launch(self, *args):
        args = ", ".join(str(arg) for arg in args)
        code = launch_code.format(jitify_object=self.variable,
                                  args=args)
        get_interpreter().parse_and_execute(code)
