cmake_minimum_required(VERSION 3.16)
project(Qupiler LANGUAGES C CXX)

# -----------------------------------------------------------------------------
# Common settings
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------------
# Find LLVM and MLIR
# -----------------------------------------------------------------------------
# Default MLIR_DIR to Homebrew LLVM if not provided
if(NOT DEFINED MLIR_DIR)
  set(MLIR_DIR "/opt/homebrew/opt/llvm/lib/cmake/mlir" CACHE PATH
      "Path to MLIRConfig.cmake (e.g. Homebrew LLVM)")
endif()

# Find LLVM (for include paths)
find_package(LLVM REQUIRED CONFIG
  PATHS /opt/homebrew/opt/llvm/lib/cmake/llvm
  NO_DEFAULT_PATH
)
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

# Find MLIR
find_package(MLIR REQUIRED CONFIG
  PATHS ${MLIR_DIR}
  NO_DEFAULT_PATH
)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)

# -----------------------------------------------------------------------------
# 1) Dialect static library
# -----------------------------------------------------------------------------
add_library(dialect STATIC
  dialect/dialect.cpp
  dialect/utils.cpp
)
target_include_directories(dialect PUBLIC
  ${CMAKE_SOURCE_DIR}/dialect
)

# -----------------------------------------------------------------------------
# 2) IR generator executable
# -----------------------------------------------------------------------------
add_executable(direct_ir_gen
  backend/direct_ir_gen.cpp
)
target_link_libraries(direct_ir_gen PRIVATE dialect)
target_include_directories(direct_ir_gen PRIVATE
  ${CMAKE_SOURCE_DIR}/dialect
  ${CMAKE_SOURCE_DIR}/backend
)

# -----------------------------------------------------------------------------
# 3) Quantum optimization library
# -----------------------------------------------------------------------------
add_library(QuantumPassLib STATIC
  backend/passes/quantum_fusion_pass.cpp
)

target_include_directories(QuantumPassLib PUBLIC
  ${CMAKE_SOURCE_DIR}
)

# -----------------------------------------------------------------------------
# 4) Frontend driver
# -----------------------------------------------------------------------------
add_executable(frontend_driver
  frontend/main.cpp
)

target_include_directories(frontend_driver PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)

target_compile_definitions(frontend_driver PRIVATE
  ${LLVM_DEFINITIONS}
)

target_link_libraries(frontend_driver PRIVATE
  QuantumPassLib
  MLIRIR
  MLIRParser
  MLIRTransforms
  MLIRSupport
)

# -----------------------------------------------------------------------------
# Usage:
#
#   mkdir build && cd build
#   cmake -G Ninja ..
#   ninja
#
# Binaries produced in build/:
#   - direct_ir_gen
#   - frontend_driver
#
# Example frontend_driver invocation:
#   cd experiments
#   ../build/frontend_driver simple.c
