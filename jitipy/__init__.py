# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from jitipy import clanginterpreter

clanginterpreter.initialize(['clang'])


class Interpreter:
    def __init__(self):
        self.c_obj = clanginterpreter.create_interpreter()

    def __del__(self):
        clanginterpreter.delete_interpreter(self.c_obj)

    def execute(self, code):
        if isinstance(code, str):
            code = (code,)

        for line in code:
            error = clanginterpreter.parse_and_execute(self.c_obj, line)
            if error:
                raise RuntimeError('Clang error')


def llvm_shutdown():
    clanginterpreter.llvm_shutdown()


__all__ = (
    'Interpreter',
    'llvm_shutdown',
)
