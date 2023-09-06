/*
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 */

#include <iostream>
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Interpreter/Interpreter.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/JITLoaderGDB.h"

extern "C" void*
create_interpreter()
{
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();

  clang::IncrementalCompilerBuilder CB;

  CB.SetCudaSDK("/usr/local/cuda");

  std::unique_ptr<clang::CompilerInstance> CI;

  // Pretend our executable is the clang-repl, since that seems to embody some
  // behaviour (particularly around the resource and include dirs)
  CB.SetMainExecutableName("/home/gmarkall/.local/opt/llvm/main-debug/bin/clang-repl");
  auto compiler_or_error = CB.CreateCudaHost();
  if (auto E = compiler_or_error.takeError())
  {
    std::cerr << toString(std::move(E)) << std::endl;
    return nullptr;
  }

  CI = std::move(*compiler_or_error);

  std::unique_ptr<clang::Interpreter> Interp;
  auto interp_or_error = clang::Interpreter::create(std::move(CI));

  if (auto E = interp_or_error.takeError())
  {
    std::cerr << toString(std::move(E)) << std::endl;
    return nullptr;
  }

  Interp = std::move(*interp_or_error);

  if (auto E = Interp->LoadDynamicLibrary("libcudart.so"))
  {
    std::cerr << toString(std::move(E)) << std::endl;
    return nullptr;
  }

  return Interp.release();
}

extern "C" void
delete_interpreter(void *interpreter)
{
  delete static_cast<clang::Interpreter*>(interpreter);
}

extern "C" void
llvm_shutdown()
{
  llvm::llvm_shutdown();
}

extern "C" bool
parse_and_execute(void *interpreter, const char *line)
{
  auto ip = static_cast<clang::Interpreter*>(interpreter);
  if (auto E = ip->ParseAndExecute(line))
  {
    std::cerr << toString(std::move(E)) << std::endl;
    return true;
  }

  return false;
}

extern "C" bool
load_dynamic_library(void *interpreter, const char *name)
{
  auto ip = static_cast<clang::Interpreter*>(interpreter);
  if (auto E = ip->LoadDynamicLibrary(name))
  {

    std::cerr << toString(std::move(E)) << std::endl;
    return true;
  }

  return false;
}

LLVM_ATTRIBUTE_USED void linkComponents() {
  llvm::errs() << (void *)&llvm_orc_registerJITLoaderGDBWrapper
               << (void *)&llvm_orc_registerJITLoaderGDBAllocAction;
}
