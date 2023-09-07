/*
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 */

#include <iostream>
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Interpreter/Interpreter.h"
#include "clang/Interpreter/Value.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/JITLoaderGDB.h"

extern "C" void*
create_interpreter(const char* resource_path)
{
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();

  clang::IncrementalCompilerBuilder CB;

  CB.SetCudaSDK("/usr/local/cuda");

  std::unique_ptr<clang::CompilerInstance> CI;

  // If the clang installation path does not coincide with the Python
  // installation path, the interpreter won't be able to locate the clang
  // resources because it will be searching in the Python installation path
  // (determined based on the main executable path). We can add the correct
  // path to the resources to the LLVM path so that it can locate the
  // resources.
  if (resource_path)
  {
    CB.SetCompilerArgs({"-resource-dir", resource_path});
  }

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
parse_and_execute(void *interpreter, const char *line, void **result_ptr)
{
  auto ip = static_cast<clang::Interpreter*>(interpreter);
  clang::Value *V = new clang::Value;
  if (auto E = ip->ParseAndExecute(line, V))
  {
    std::cerr << toString(std::move(E)) << std::endl;
    return true;
  }

  *result_ptr = static_cast<void*>(V);
  return false;
}

extern "C" void
delete_value(void *value)
{
  auto vp = static_cast<clang::Value*>(value);
  delete vp;
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
