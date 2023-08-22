/*
 * Copytight (c) 2023, NVIDIA CORPORATION. All rights reserved.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define JITIFY_PRINT_INSTANTIATION 1
#define JITIFY_PRINT_SOURCE 1
#define JITIFY_PRINT_LOG 1
#define JITIFY_PRINT_PTX 1
#define JITIFY_PRINT_LINKER_LOG 1
#define JITIFY_PRINT_LAUNCH 1

#include "jitify.hpp"

#define CHECK_CUDA(call)                                                  \
  do {                                                                    \
    if (call != CUDA_SUCCESS) {                                           \
      const char* str;                                                    \
      cuGetErrorName(call, &str);                                         \
      std::cout << "(CUDA) returned " << str;                             \
      std::cout << " (" << __FILE__ << ":" << __LINE__ << ":" << __func__ \
                << "())" << std::endl;                                    \
      Py_RETURN_NONE;                                                     \
    }                                                                     \
  } while (0)

template <typename T>
bool are_close(T in, T out) {
  return fabs(in - out) <= 1e-5f * fabs(in);
}

static PyObject *test_simple(PyObject *self, PyObject *args) {
  using T = float;
  const char* program_source =
      "my_program\n"
      "template<int N, typename T>\n"
      "__global__\n"
      "void my_kernel(T* data) {\n"
      "    T data0 = data[0];\n"
      "    for( int i=0; i<N-1; ++i ) {\n"
      "        data[0] *= data0;\n"
      "    }\n"
      "}\n";
  static jitify::JitCache kernel_cache;
  jitify::Program program = kernel_cache.program(program_source, 0);
  T h_data = 5;
  T* d_data;
  cudaMalloc((void**)&d_data, sizeof(T));
  cudaMemcpy(d_data, &h_data, sizeof(T), cudaMemcpyHostToDevice);
  dim3 grid(1);
  dim3 block(1);
  using jitify::reflection::type_of;
  CHECK_CUDA(program.kernel("my_kernel")
                 .instantiate(3, type_of(*d_data))
                 .configure(grid, block)
                 .launch(d_data));
  cudaMemcpy(&h_data, d_data, sizeof(T), cudaMemcpyDeviceToHost);
  cudaFree(d_data);
  std::cout << h_data << std::endl;
  assert(are_close(h_data, 125.f));

  Py_RETURN_NONE;
}

static PyObject* jitify_error;

static PyObject *create_jit_cache(PyObject *self, PyObject *args)
{
  jitify::JitCache *cache = nullptr;
  try {
    cache = new jitify::JitCache{};
  } catch (const std::bad_alloc &) {
    PyErr_NoMemory();
    return nullptr;
  }

  PyObject *ret = PyLong_FromUnsignedLongLong((unsigned long long)cache);
  if (ret == nullptr) {
    delete cache;
  }

  return ret;
}

static PyObject *delete_jit_cache(PyObject *self, PyObject *args)
{
  jitify::JitCache *cache = nullptr;
  if (!PyArg_ParseTuple(args, "K", &cache))
    return nullptr;

  delete cache;

  Py_RETURN_NONE;
}

static PyObject *jit_cache_program(PyObject *self, PyObject *args)
{
  jitify::JitCache *cache = nullptr;
  char *program_source;
  if (!PyArg_ParseTuple(args, "Ks", &cache, &program_source))
    return nullptr;

  jitify::Program *program = nullptr;
  try {
    program = new jitify::Program(*cache, program_source);
  } catch (const std::bad_alloc &) {
    PyErr_NoMemory();
    return nullptr;
  } catch (const std::runtime_error &re) {
    PyErr_SetString(jitify_error, re.what());
    return nullptr;
  }

  PyObject *ret = PyLong_FromUnsignedLongLong((unsigned long long)program);
  if (ret == nullptr) {
    delete program;
  }

  return ret;
}

static PyObject *delete_program(PyObject *self, PyObject *args)
{
  jitify::Program *program = nullptr;
  if (!PyArg_ParseTuple(args, "K", &program))
    return nullptr;

  delete program;

  Py_RETURN_NONE;
}

static PyMethodDef ext_methods[] = {
    {"test_simple", (PyCFunction)test_simple, METH_VARARGS,
      "Runs a simple test"},
    {"create_jit_cache", (PyCFunction)create_jit_cache, METH_VARARGS,
      "Create a jitify::JitCache instance"},
    {"delete_jit_cache", (PyCFunction)delete_jit_cache, METH_VARARGS,
      "Delete a jitify::JitCache instance"},
    {"jit_cache_program", (PyCFunction)jit_cache_program, METH_VARARGS,
      "Create a cached jitify::Program instance"},
    {"delete_program", (PyCFunction)delete_program, METH_VARARGS,
      "Delete a jitify::Program instance"},
    {nullptr}
};

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT, "jitipy",
  "Provides access to jitify", -1, ext_methods};

PyMODINIT_FUNC PyInit__lib(void) {
  PyObject *m = PyModule_Create(&moduledef);
  jitify_error = PyErr_NewException("_lib.JitifyError", nullptr, nullptr);
  if (jitify_error == nullptr)
    return nullptr;

  if (PyModule_AddObjectRef(m, "JitifyError", jitify_error) < 0)
    return nullptr;

  return m;
}
