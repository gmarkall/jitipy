# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

# This file demonstrates the use of Jitipy, a Python API to Jitify backed by
# the Clang interpreter, to instantiate a simple kernel and launch it.


# Main library
import jitipy

# Use to create and manage test data
import numpy as np
from numba import cuda

# A simple templated kernel
source = """
#include <stdint.h>

template<int N, typename T>
__global__
void my_kernel(T* data) {
    T data0 = data[0];
    for( int i=0; i<N-1; ++i ) {
        data[0] *= data0;
    }
}
"""

# Program creation
program = jitipy.Program("my_program", source)


# Instantiation for float32 data
data = cuda.to_device(np.asarray([5], dtype=np.float32))
data_ptr = data.__cuda_array_interface__['data'][0]
kernel = program.preprocess().get_kernel("my_kernel<3, float>")
kernel.configure(1, 1).launch(data_ptr)

print(data[0])  # prints 125.0


# Instantiation for int64 data
data = cuda.to_device(np.asarray([5], dtype=np.int64))
data_ptr = data.__cuda_array_interface__['data'][0]
kernel = program.preprocess().get_kernel("my_kernel<4, int64_t>")
kernel.configure(1, 1).launch(data_ptr)

print(data[0])  # prints 625
