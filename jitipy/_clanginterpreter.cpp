/*
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 */

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Interpreter/Interpreter.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ManagedStatic.h"

#include <iostream>

extern "C" void
initialize(int argc, char **argv)
{
  const char* exe = "python";
  const char* inc = "-I/home/gmarkall/.local/opt/llvm/main/lib/clang/18/include";
  const char* fake_argv[] = {exe, inc};
  int fake_argc = 2;
  //llvm::cl::ParseCommandLineOptions(2, fake_argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);
}

extern "C" void*
create_interpreter()
{
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();

  clang::IncrementalCompilerBuilder CB;
  //std::vector<const char *> ClangArgv(0); // = {
  //  "-I/home/gmarkall/.local/opt/llvm/main/lib/clang/18/include"
  //};
  //CB.SetCompilerArgs(ClangArgv);

  std::unique_ptr<clang::CompilerInstance> CI;

  auto compiler_or_error = CB.CreateCpp();
  if (auto E = compiler_or_error.takeError())
  {
    std::cerr << toString(std::move(E)) << std::endl;
    return nullptr;
  }

  CI = std::move(*compiler_or_error);
  CI->LoadRequestedPlugins();

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

extern "C" bool
parse_and_execute(void *interpreter, const char *line)
{
  auto i = static_cast<clang::Interpreter*> (interpreter);
  if (auto Err = i->ParseAndExecute(line))
  {
    // TODO: Add diagnostics handler
    std::cerr << toString(std::move(Err)) << std::endl;
    return true;
  }

  return false;
}
