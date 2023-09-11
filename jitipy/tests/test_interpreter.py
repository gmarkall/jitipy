# Copyright (c) 2023, NVIDIA Corporation

import ctypes
import pytest
from jitipy import clanginterpreter


@pytest.fixture
def interpreter(llvm_shutdown_at_end):
    interpreter = clanginterpreter.create_interpreter()
    yield interpreter
    clanginterpreter.delete_interpreter(interpreter)


def test_create_delete_interpreter(interpreter):
    assert interpreter != 0


def test_cuda_include(interpreter):
    clanginterpreter.parse_and_execute(interpreter, "#include <cuda.h>")


def test_cuda_initialize(interpreter):
    clanginterpreter.parse_and_execute(interpreter, "#include <cuda.h>")
    clanginterpreter.parse_and_execute(interpreter, "cudaSetDevice(0);")


def test_include_jitify(interpreter):
    clanginterpreter.parse_and_execute(interpreter,
                                       '#include "jitipy/jitify2.hpp"')


def test_code_execution(interpreter):
    result = clanginterpreter.parse_and_execute(interpreter, "2 + 3")
    assert result[0].Data.m_Int == 5


def test_code_execution_multiple_lines(interpreter):
    code = [
        "int add(int x, int y) { return x + y; }",
        "add(3, 4)",
    ]
    result = clanginterpreter.parse_and_execute(interpreter, code)
    assert result[0].Data.m_Int == 7


def test_double_value(interpreter):
    result = clanginterpreter.parse_and_execute(interpreter, "3.14")
    assert result[0].Data.m_Double == 3.14


def test_pointer_value(interpreter):
    result = clanginterpreter.parse_and_execute(interpreter, "new int(42)")
    ptr = result[0].Data.m_Ptr
    assert ptr != 0
    assert ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int))[0] == 42
