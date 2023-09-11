# Copyright (c) 2023, NVIDIA Corporation

import pytest
from jitipy import clanginterpreter


@pytest.fixture(scope='session')
def llvm_shutdown_at_end():
    yield
    clanginterpreter.llvm_shutdown()


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
