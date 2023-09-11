import jitipy
import numpy as np
from numba import cuda


source = """
__global__
void my_kernel(float* data) {
    float data0 = data[0];
    int N = 3;
    for( int i=0; i<N-1; ++i ) {
        data[0] *= data0;
    }
}
"""


data = cuda.device_array(1, dtype=np.float32)
data[0] = 5
data_ptr = data.__cuda_array_interface__['data'][0]

program = jitipy.Program("my_program", source)
program.preprocess().get_kernel("my_kernel").configure(1, 1).launch(data_ptr)

print(data[0])
