# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import ctypes
import pathlib

sopath = pathlib.Path(__file__).absolute().parent / 'lib_clanginterpreter.so'

lib = ctypes.CDLL(sopath)

lib.initialize.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]

lib.create_interpreter.restype = ctypes.c_void_p

lib.delete_interpreter.argtypes = [ctypes.c_void_p]

lib.parse_and_execute.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.parse_and_execute.restype = ctypes.c_bool


def initialize(args):
    argc = len(args)
    if argc != 1:
        raise RuntimeError("Multiple arguments unsupported")
    lib.initialize(argc, ctypes.byref(ctypes.c_char_p(args[0].encode())))


def create_interpreter():
    return lib.create_interpreter()


def delete_interpreter(interpreter):
    lib.delete_interpreter(interpreter)


def parse_and_execute(interpreter, line):
    return lib.parse_and_execute(interpreter, line.encode())


def llvm_shutdown():
    lib.llvm_shutdown()
