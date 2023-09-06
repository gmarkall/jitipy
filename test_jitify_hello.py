import jitipy

code = """
#include <cmath>
#include <cassert>
#include <iostream>

#define CHECK_CUDA(call)                                                  \
  do {                                                                    \
    if (call != CUDA_SUCCESS) {                                           \
      const char* str;                                                    \
      cuGetErrorName(call, &str);                                         \
      std::cout << "(CUDA) returned " << str;                             \
      std::cout << " (" << __FILE__ << ":" << __LINE__ << ":" << __func__ \
                << "())" << std::endl;                                    \
      return false;                                                       \
    }                                                                     \
  } while (0)

template <typename T> bool are_close(T in, T out) { return fabs(in - out) <= 1e-5f * fabs(in); }

const char* program_source =               \
    "template<int N, typename T>\\n"       \
    "__global__\\n"                        \
    "void my_kernel(T* data) {\\n"         \
    "    T data0 = data[0];\\n"            \
    "    for( int i=0; i<N-1; ++i ) {\\n"  \
    "        data[0] *= data0;\\n"         \
    "    }\\n"                             \
    "}\\n";


using T = float;
T h_data = 5;
T* d_data;

using jitify2::reflection::type_of;

auto program = jitify2::Program("my_program", program_source);
auto preprocessed_program = program->preprocess();
auto kernel_inst = jitify2::reflection::Template("my_kernel").instantiate(3, type_of(*d_data));
auto compiled = preprocessed_program->compile(kernel_inst);
auto linked = compiled->link();
auto kernel = linked->load()->get_kernel(kernel_inst);
cudaMalloc((void**)&d_data, sizeof(T));
cudaMemcpy(d_data, &h_data, sizeof(T), cudaMemcpyHostToDevice);

dim3 grid(1);
dim3 block(1);

CHECK_CUDA(kernel->configure(grid, block)->launch(d_data));
cudaMemcpy(&h_data, d_data, sizeof(T), cudaMemcpyDeviceToHost);
cudaFree(d_data);
std::cout << h_data << std::endl;
assert(are_close(h_data, 125.f));
""".splitlines()  # noqa: E501


print("Create interpreter")
interpreter = jitipy.create_interpreter()


print("Include jitify")
jitipy.parse_and_execute(interpreter, '#include "jitipy/jitify2.hpp"')

print("Execute code")
jitipy.parse_and_execute(interpreter, code)

print("Delete interpreter")
jitipy.delete_interpreter(interpreter)

print("Shutdown")
jitipy.llvm_shutdown()
