# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import ctypes
import pathlib

DEBUG_EXECUTION = False

sopath = pathlib.Path(__file__).absolute().parent / 'lib_clanginterpreter.so'

lib = ctypes.CDLL(sopath, mode=ctypes.RTLD_GLOBAL)
lib.create_interpreter.restype = ctypes.c_void_p

lib.delete_interpreter.argtypes = [ctypes.c_void_p]

lib.parse_and_execute.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                  ctypes.c_void_p]
lib.parse_and_execute.restype = ctypes.c_bool


def create_interpreter():
    interp = lib.create_interpreter()
    if not interp:
        raise RuntimeError("Error creating interpreter")
    return interp


def delete_interpreter(interpreter):
    lib.delete_interpreter(interpreter)


def llvm_shutdown():
    lib.llvm_shutdown()


def parse_and_execute(interpreter, code):
    result = ctypes.c_int(0)

    if isinstance(code, str):
        code = (code,)

    continuation = ''

    for line in code:
        if line.endswith('\\'):
            continuation = line
            continue

        line = continuation + line
        if DEBUG_EXECUTION:
            print(f"--- Executing: {line}")

        if lib.parse_and_execute(interpreter, line.encode(),
                                 ctypes.byref(result)):
            print("Error returned")
            return

        continuation = ''

    return result.value


def load_dynamic_library(interpreter, name):
    if lib.load_dynamic_library(name):
        print("Error returned")
