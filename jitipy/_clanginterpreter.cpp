/*
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject*
create_interpreter(PyObject *self, PyObject *args)
{
  Py_RETURN_NONE;
}

static PyMethodDef ext_methods[] = {
    {"create_interpreter", (PyCFunction)create_interpreter, METH_VARARGS,
      "Create a Clang interpreter"},
    {nullptr}
};

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT, "jitipy",
  "Provides access to the Clang interpreter", -1, ext_methods};

PyMODINIT_FUNC PyInit__clanginterpreter(void) {
  PyObject *m = PyModule_Create(&moduledef);
  return m;
}
