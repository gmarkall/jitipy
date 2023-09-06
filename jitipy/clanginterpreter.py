# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import ctypes
import pathlib


class Storage(ctypes.Union):
    _fields_ = [("m_Bool", ctypes.c_bool),
                ("m_Char_S", ctypes.c_char),
                ("m_SChar", ctypes.c_char),
                ("m_UChar", ctypes.c_uint8),
                ("m_Short", ctypes.c_short),
                ("m_UShort", ctypes.c_ushort),
                ("m_Int", ctypes.c_int),
                ("m_UInt", ctypes.c_uint),
                ("m_Long", ctypes.c_long),
                ("m_ULong", ctypes.c_ulong),
                ("m_LongLong", ctypes.c_longlong),
                ("m_ULongLong", ctypes.c_ulonglong),
                ("m_Float", ctypes.c_float),
                ("m_Double", ctypes.c_double),
                ("m_LongDouble", ctypes.c_longdouble),
                ("m_Ptr", ctypes.c_void_p)]


class Value(ctypes.Structure):
    _fields_ = [("Interp", ctypes.c_void_p),
                ("OpaqueType", ctypes.c_void_p),
                ("Data", Storage),
                ("ValueKind", ctypes.c_int),
                ("IsManuallyAlloc", ctypes.c_bool)]


ValuePointer = ctypes.POINTER(Value)

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
    result = ctypes.c_uint64(0)

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

    return ctypes.cast(result.value, ValuePointer)


def load_dynamic_library(interpreter, name):
    if lib.load_dynamic_library(name):
        print("Error returned")
