# Copyright (c) 2023, NVIDIA Corporation

import jitipy
import pytest
import numpy as np
from numba import cuda


templated_kernel_source = """
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


@pytest.fixture
def templated_program(llvm_shutdown_at_end):
    return jitipy.Program("my_program", templated_kernel_source)


def test_create_program(templated_program):
    assert templated_program.ptr != 0


def test_preprocess_program(templated_program):
    preprocessed_program = templated_program.preprocess()
    assert preprocessed_program.ptr != 0


def test_launch_float_3(templated_program):
    data = cuda.to_device(np.asarray([5], dtype=np.float32))
    data_ptr = data.__cuda_array_interface__['data'][0]
    float_3 = templated_program.preprocess().get_kernel("my_kernel<3, float>")
    float_3.configure(1, 1).launch(data_ptr)
    assert data[0] == 125.0


def test_launch_int64_4(templated_program):
    data = cuda.to_device(np.asarray([5], dtype=np.int64))
    data_ptr = data.__cuda_array_interface__['data'][0]
    preprocessed = templated_program.preprocess()
    int64_4 = preprocessed.get_kernel("my_kernel<4, int64_t>")
    int64_4.configure(1, 1).launch(data_ptr)
    assert data[0] == 625
