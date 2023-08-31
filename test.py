import jitipy

print("Create interpreter")
interpreter = jitipy.Interpreter()

print("Execute code")
interpreter.execute([
    '#include <iostream>',
    'std::cout << "Hello World!" << std::endl;',
])

print("Delete interpreter")
del interpreter

print("Shutdown")
jitipy.llvm_shutdown()
