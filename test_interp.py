print("Import")
from jitipy import create_interpreter, delete_interpreter, llvm_shutdown

print("Create interpreter")
interpreter = create_interpreter()
print("Delete interpreter")
delete_interpreter(interpreter)
print("Shutdown")
llvm_shutdown()
