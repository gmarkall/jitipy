cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

project(jitipy)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

find_package(PythonExtensions REQUIRED)
find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in ${LLVM_DIR}")
message(STATUS "Found Clang ${Clang_VERSION}")
message(STATUS "Using ClangConfig.cmake in ${Clang_DIR}")

list(APPEND CMAKE_MODULE_PATH ${LLVM_DIR})
list(APPEND CMAKE_MODULE_PATH ${Clang_DIR})
include(AddLLVM)
include(AddClang)

include_directories(SYSTEM "${LLVM_INCLUDE_DIRS};${CLANG_INCLUDE_DIRS}")

set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Core
  Option
  OrcJIT
  Support
)

add_clang_library(_clanginterpreter jitipy/_clanginterpreter.cpp SHARED)

clang_target_link_libraries(_clanginterpreter PUBLIC
  clangAST
  clangBasic
  clangFrontend
  clangInterpreter
)

install(TARGETS _clanginterpreter LIBRARY DESTINATION jitipy)
