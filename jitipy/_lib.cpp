/*
 * Copytight (c) 2023, NVIDIA CORPORATION. All rights reserved.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "jitify.hpp"

static PyObject *test_simple(PyObject *self, PyObject *args) {
  Py_RETURN_NONE;
}

static PyMethodDef ext_methods[] = {
    {"test_simple", (PyCFunction)test_simple, METH_VARARGS,
      "Runs a simple test"},
};

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT, "jitipy",
  "Provides access to jitify", -1, ext_methods};

PyMODINIT_FUNC PyInit__lib(void) {
  PyObject *m = PyModule_Create(&moduledef);
  return m;
}
