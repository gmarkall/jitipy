import jitipy
import numpy as np
from numba import cuda


source = """
#include <stdint.h>

template<typename T, int N>
__global__
void my_kernel(T* data) {
    float data0 = data[0];
    for( int i=0; i<N-1; ++i ) {
        data[0] *= data0;
    }
}
"""

program = jitipy.Program("my_program", source)

data = cuda.device_array(1, dtype=np.float32)
data[0] = 5
data_ptr = data.__cuda_array_interface__['data'][0]
float_3 = program.preprocess().get_kernel("my_kernel<float, 3>")
float_3.configure(1, 1).launch(data_ptr)
print(data[0])

data = cuda.device_array(1, dtype=np.int64)
data[0] = 5
data_ptr = data.__cuda_array_interface__['data'][0]
uint64_4 = program.preprocess().get_kernel("my_kernel<uint64_t, 4>")
uint64_4.configure(1, 1).launch(data_ptr)
print(data[0])
