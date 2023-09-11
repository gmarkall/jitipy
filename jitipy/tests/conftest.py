# Copyright (c) 2023, NVIDIA Corporation

import pytest

from jitipy import clanginterpreter


@pytest.fixture(scope='session')
def llvm_shutdown_at_end():
    yield
    print("LLVM shutdown")
    clanginterpreter.llvm_shutdown()
