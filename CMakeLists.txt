cmake_minimum_required(VERSION 3.21)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)

option(CUNIBS_USE_SYSTEM_CUDA "Use the system CUDA toolkit instead of the CUDA wheels" OFF)

set(CUNIBS_CUDA_LIBS cudart cublasLt cublas)

if(NOT CUNIBS_USE_SYSTEM_CUDA)
  include(CunibsCudaPathfinder)
  cunibs_probe_cuda(LIBS ${CUNIBS_CUDA_LIBS})

  # CMakeDetermineCUDACompiler is skipped entirely once CMakeCUDACompiler.cmake
  # exists, so a stale compiler would survive a moved venv or a switch to a
  # system toolkit until the build directory is wiped.
  if(CMAKE_CUDA_COMPILER AND NOT CMAKE_CUDA_COMPILER STREQUAL CUNIBS_CUDA_NVCC)
    message(FATAL_ERROR
      "The cached CUDA compiler no longer matches the one cuda.pathfinder resolves.\n"
      "  cached: ${CMAKE_CUDA_COMPILER}\n"
      "  probed: ${CUNIBS_CUDA_NVCC}\n"
      "Delete ${CMAKE_BINARY_DIR} and reconfigure.")
  endif()
  set(CMAKE_CUDA_COMPILER "${CUNIBS_CUDA_NVCC}")
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

enable_language(CUDA)

if(NOT CUNIBS_USE_SYSTEM_CUDA AND NOT WIN32)
  cunibs_seed_cuda_toolkit(
    LIBS ${CUNIBS_CUDA_LIBS}
    INCLUDE_DIR "${CUNIBS_CUDA_INCLUDE_DIR}"
    FINGERPRINT "${CUNIBS_CUDA_FINGERPRINT}")
endif()

find_package(CUDAToolkit 12.0 REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Default build type" FORCE)
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_solver_ext LTO FREE_THREADED
  src/cunibs/solver/bindings.cpp
  src/cunibs/solver/solver.cpp
  src/cunibs/solver/cast.cu
  src/cunibs/solver/vcycle.cu
  src/cunibs/solver/aggregate.cu
  src/cunibs/solver/block_cg.cu
  src/cunibs/solver/dadt.cu
  src/cunibs/solver/dadt_element.cu
  src/cunibs/solver/rhs.cu
  src/cunibs/solver/reconstruct.cu
  src/cunibs/solver/adm.cu
  src/cunibs/solver/moments.cu
  src/cunibs/solver/place.cu
)

set_source_files_properties(src/cunibs/solver/dadt.cu PROPERTIES COMPILE_OPTIONS "--use_fast_math")

target_link_libraries(
  _solver_ext
  PRIVATE CUDA::cudart CUDA::cublas
)

set(_CUNIBS_INSTALL_RPATH "$ORIGIN")
if(NOT CUNIBS_USE_SYSTEM_CUDA)
  list(APPEND _CUNIBS_INSTALL_RPATH "$ORIGIN/../../nvidia/cu13/lib")
endif()
set_target_properties(_solver_ext PROPERTIES INSTALL_RPATH "${_CUNIBS_INSTALL_RPATH}")

target_link_options(_solver_ext PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wl,--disable-new-dtags>)

install(TARGETS _solver_ext
  RUNTIME DESTINATION cunibs/solver
  LIBRARY DESTINATION cunibs/solver NAMELINK_SKIP
  ARCHIVE DESTINATION ${SKBUILD_NULL_DIR}
)

nanobind_add_stub(
  _solver_ext_stub
  MODULE _solver_ext
  OUTPUT solver/_solver_ext.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:_solver_ext>
  DEPENDS _solver_ext
  MARKER_FILE solver/py.typed
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/solver/_solver_ext.pyi
  ${CMAKE_CURRENT_BINARY_DIR}/solver/py.typed
  DESTINATION cunibs/solver
)
