cmake_minimum_required(VERSION 3.21)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
endif()

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)

option(CUNIBS_USE_SYSTEM_CUDA "Use the system CUDA toolkit instead of the CUDA wheels" OFF)

set(CUNIBS_CUDA_LIBS cudart)

if(NOT CUNIBS_USE_SYSTEM_CUDA)
  include(CunibsCudaPathfinder)
  cunibs_probe_cuda(LIBS ${CUNIBS_CUDA_LIBS})

  # CMakeDetermineCUDACompiler is skipped entirely once CMakeCUDACompiler.cmake exists,
  # so a stale compiler would otherwise survive a moved venv or a toolkit switch.
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

if(WIN32 AND NOT CUNIBS_USE_SYSTEM_CUDA)
  set(CMAKE_CUDA_RUNTIME_LIBRARY None)
endif()

enable_language(CUDA)

if(NOT CUNIBS_USE_SYSTEM_CUDA)
  if(WIN32)
    include(CunibsWindowsImplib)
    cunibs_generate_cuda_implibs(LIBS ${CUNIBS_CUDA_LIBS})
  endif()
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

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_solver_ext LTO FREE_THREADED
  src/cunibs/solver/bindings.cpp
  src/cunibs/solver/solver.cpp
  src/cunibs/solver/vcycle.cu
  src/cunibs/solver/aggregate.cu
  src/cunibs/solver/block_cg.cu
  src/cunibs/solver/dadt.cu
  src/cunibs/solver/gradient.cu
  src/cunibs/solver/dadt_element.cu
  src/cunibs/solver/rhs.cu
  src/cunibs/solver/stiffness.cu
  src/cunibs/solver/l1.cu
  src/cunibs/solver/reconstruct.cu
  src/cunibs/solver/recovery.cu
  src/cunibs/solver/adm.cu
  src/cunibs/solver/moments.cu
  src/cunibs/solver/place.cu
)

set_source_files_properties(src/cunibs/solver/dadt.cu PROPERTIES COMPILE_OPTIONS "--use_fast_math")

if(MSVC)
  # solver.hpp and several kernels carry UTF-8 bytes; MSVC otherwise decodes the sources
  # in the system ANSI code page.
  set(_msvc_flags /utf-8)
  list(JOIN _msvc_flags "," _msvc_flags_nvcc)
  target_compile_options(_solver_ext PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${_msvc_flags}>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${_msvc_flags_nvcc}>")
else()
  # -Xcompiler only reaches the host half of a .cu, which is where the unused-parameter and
  # sign-compare classes live. Device-side mistakes and ignored CUDA return codes are not
  # covered by this and still have to be caught by hand.
  set(_warn_flags -Wall -Wextra)
  list(JOIN _warn_flags "," _warn_flags_nvcc)
  target_compile_options(_solver_ext PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${_warn_flags}>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${_warn_flags_nvcc}>")
endif()

target_link_libraries(
  _solver_ext
  PRIVATE CUDA::cudart
)

if(NOT WIN32)
  set(_CUNIBS_INSTALL_RPATH "$ORIGIN")
  if(NOT CUNIBS_USE_SYSTEM_CUDA)
    list(APPEND _CUNIBS_INSTALL_RPATH "$ORIGIN/../../nvidia/cu13/lib")
  endif()
  set_target_properties(_solver_ext PROPERTIES INSTALL_RPATH "${_CUNIBS_INSTALL_RPATH}")

  target_link_options(_solver_ext PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wl,--disable-new-dtags>)
endif()

install(TARGETS _solver_ext
  RUNTIME DESTINATION cunibs/solver
  LIBRARY DESTINATION cunibs/solver NAMELINK_SKIP
  ARCHIVE DESTINATION ${SKBUILD_NULL_DIR}
)

# Stub generation imports the freshly built extension, and that import has to go through
# cunibs.solver.__init__, which preloads the CUDA libraries; importing the bare
# _solver_ext skips the preload and fails on Windows, where the loader will not find
# cudart64_13.dll on its own. Hence the staging tree: it is the only layout the dotted
# import resolves against.
#
# The top-level __init__ is empty rather than the real one, which calls
# importlib.metadata.version("cunibs") and so cannot run before installation.
set(_stub_root "${CMAKE_CURRENT_BINARY_DIR}/stub_pkg")
set(_stub_pkg "${_stub_root}/cunibs/solver")
set_target_properties(_solver_ext PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${_stub_pkg}")
file(CONFIGURE OUTPUT "${_stub_root}/cunibs/__init__.py" CONTENT "")
configure_file(src/cunibs/solver/__init__.py "${_stub_pkg}/__init__.py" COPYONLY)
configure_file(src/cunibs/solver/_cuda_preload.py "${_stub_pkg}/_cuda_preload.py" COPYONLY)

nanobind_add_stub(
  _solver_ext_stub
  MODULE cunibs.solver._solver_ext
  OUTPUT solver/_solver_ext.pyi
  PYTHON_PATH "${_stub_root}"
  DEPENDS _solver_ext
  MARKER_FILE solver/py.typed
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/solver/_solver_ext.pyi
  ${CMAKE_CURRENT_BINARY_DIR}/solver/py.typed
  DESTINATION cunibs/solver
)
