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


program = jitipy.Program("my_program", source)
print(f'0x{program.ptr:x}')
#input()

#sys.exit(0)

data = cuda.device_array(1, dtype=np.float32)
data[0] = 5
data_ptr = data.__cuda_array_interface__['data'][0]
print(f"Data: 0x{data_ptr:x}")
program.test_kernel(data_ptr)

print(data[0])
