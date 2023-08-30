/*
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 */

#include <iostream>
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Interpreter/Interpreter.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ManagedStatic.h"

extern "C" void*
create_interpreter()
{
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();

  clang::IncrementalCompilerBuilder CB;

  std::unique_ptr<clang::CompilerInstance> CI;

  auto compiler_or_error = CB.CreateCpp();
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
