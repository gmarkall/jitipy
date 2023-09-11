import jitipy
import numpy as np
from numba import cuda


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

program = jitipy.Program("my_program", source)
data = cuda.to_device(np.asarray([5], dtype=np.float32))
data_ptr = data.__cuda_array_interface__['data'][0]
kernel = program.preprocess().get_kernel("my_kernel<3, float>")
kernel.configure(1, 1).launch(data_ptr)

print(data[0])
