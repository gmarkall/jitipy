import jitipy

interpreter = jitipy.create_interpreter()

result = jitipy.parse_and_execute(interpreter, '255')
print(result[0].Data.m_Int)
