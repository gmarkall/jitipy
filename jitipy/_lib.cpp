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

static PyMethodDef ext_methods[] = {
    {"test_simple", (PyCFunction)test_simple, METH_VARARGS,
      "Runs a simple test"},
    {nullptr}
};

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT, "jitipy",
  "Provides access to jitify", -1, ext_methods};

PyMODINIT_FUNC PyInit__lib(void) {
  PyObject *m = PyModule_Create(&moduledef);
  return m;
}
