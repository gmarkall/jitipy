import jitipy


code = "my_program\n__global__ void my_kernel(int *data) { *data = 1; }"
cache = jitipy.create_jit_cache()
program = jitipy.jit_cache_program(cache, code)
jitipy.delete_program(program)
jitipy.delete_jit_cache(cache)
