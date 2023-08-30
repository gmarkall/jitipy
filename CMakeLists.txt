cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

project(jitipy)

find_package(PythonExtensions REQUIRED)
find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in ${LLVM_DIR}")
message(STATUS "Found Clang ${Clang_VERSION}")
message(STATUS "Using ClangConfig.cmake in ${Clang_DIR}")

add_library(_clanginterpreter MODULE jitipy/_clanginterpreter.cpp)
python_extension_module(_clanginterpreter)
install(TARGETS _clanginterpreter LIBRARY DESTINATION jitipy)
