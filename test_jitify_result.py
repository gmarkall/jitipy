import jitipy

interpreter = jitipy.create_interpreter()

result = jitipy.parse_and_execute(interpreter, '1')
print(result)
