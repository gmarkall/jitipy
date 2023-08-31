import jitipy

print("Create interpreter")
interpreter = jitipy.create_interpreter()

print("Include iostream")
jitipy.parse_and_execute(interpreter, "#include <iostream>")

print("Delete interpreter")
jitipy.delete_interpreter(interpreter)

print("Shutdown")
jitipy.llvm_shutdown()
