cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

project(jitipy)

find_package(PythonExtensions REQUIRED)

add_library(_clanginterpreter MODULE jitipy/_clanginterpreter.cpp)
python_extension_module(_clanginterpreter)
install(TARGETS _clanginterpreter LIBRARY DESTINATION jitipy)
