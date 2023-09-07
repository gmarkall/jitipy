import jitipy

source = """
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
print(f'0x{program.ptr:x}')

preprocessed_program = program.preprocess()
print(f'0x{preprocessed_program.ptr:x}')
