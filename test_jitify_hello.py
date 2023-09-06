import jitipy

code = """
#include <cmath>
#include <iostream>

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

kernel->configure(grid, block)->launch(d_data);
cudaMemcpy(&h_data, d_data, sizeof(T), cudaMemcpyDeviceToHost);
cudaFree(d_data);
std::cout << "Expected: " << 125.f << ", Actual: " << h_data << std::endl;
""".splitlines()  # noqa: E501


print("Create interpreter")
interpreter = jitipy.create_interpreter()

# FIXME: This is probably needed because it works around loading the wrong
# library or some sort of clash
print("Initialize cuda")
jitipy.parse_and_execute(interpreter, "#include <cuda.h>")
jitipy.parse_and_execute(interpreter, "cudaSetDevice(0);")

print("Include jitify")
jitipy.parse_and_execute(interpreter, '#include "jitipy/jitify2.hpp"')

print("Execute code")
jitipy.parse_and_execute(interpreter, code)

print("Delete interpreter")
jitipy.delete_interpreter(interpreter)

print("Shutdown")
jitipy.llvm_shutdown()
