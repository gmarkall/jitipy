import jitipy

program_source = """\
my_program
template<int N, typename T>
__global__
void my_kernel(T* data) {
    T data0 = data[0];
    for( int i=0; i<N-1; ++i ) {
        data[0] *= data0;
    }
}
"""

cache = jitipy.create_jit_cache()
program = jitipy.jit_cache_program(cache, program_source)
jitipy.delete_program(program)
jitipy.delete_jit_cache(cache)
