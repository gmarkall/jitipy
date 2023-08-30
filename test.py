import jitipy

print("Create interpreter")
interpreter = jitipy.create_interpreter()
print("Delete interpreter")
jitipy.delete_interpreter(interpreter)
print("Shutdown")
jitipy.llvm_shutdown()
