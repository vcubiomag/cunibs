cmake_minimum_required(VERSION 3.21)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)

option(CUNIBS_USE_SYSTEM_CUDA "Use the system CUDA toolkit instead of the CUDA wheels" OFF)

if(NOT CUNIBS_USE_SYSTEM_CUDA)
  include(CunibsCudaPathfinder)
  cunibs_probe_cuda()

  set(CMAKE_CUDA_COMPILER "${CUNIBS_CUDA_NVCC}")
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

message(STATUS "cunibs: CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

if(WIN32 AND NOT CUNIBS_USE_SYSTEM_CUDA)
  set(CMAKE_CUDA_RUNTIME_LIBRARY None)
endif()

enable_language(CUDA)

if(CUNIBS_USE_SYSTEM_CUDA)
  find_package(CUDAToolkit 12.9 REQUIRED)
else()
  # FindCUDAToolkit cannot locate the wheels' cudart, which has only a versioned SONAME on Linux and
  # a bare DLL on Windows. Windows links against an import library synthesized from that DLL.
  set(_cudart "${CUNIBS_CUDA_LIB_cudart}")
  if(WIN32)
    include(CunibsWindowsImplib)
    cunibs_generate_cuda_implibs(LIBS cudart)
    set(_cudart "${CUNIBS_CUDA_IMPLIB_cudart}")
  endif()
  add_library(CUDA::cudart UNKNOWN IMPORTED)
  set_target_properties(CUDA::cudart PROPERTIES
    IMPORTED_LOCATION "${_cudart}"
    INTERFACE_INCLUDE_DIRECTORIES "${CUNIBS_CUDA_INCLUDE_DIR}")
endif()

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
  src/cunibs/solver/amg/solver.cpp
  src/cunibs/solver/amg/vcycle.cu
  src/cunibs/solver/amg/aggregate.cu
  src/cunibs/solver/amg/block_cg.cu
  src/cunibs/solver/amg/l1.cu
  src/cunibs/solver/amg/slot_csr.cu
  src/cunibs/solver/fem/gradient.cu
  src/cunibs/solver/fem/pattern.cu
  src/cunibs/solver/fem/rhs.cu
  src/cunibs/solver/fem/stiffness.cu
  src/cunibs/solver/fem/reconstruct.cu
  src/cunibs/solver/fem/recovery.cu
  src/cunibs/solver/fem/reorder.cu
  src/cunibs/solver/fem/raycast.cu
  src/cunibs/solver/fem/skin.cu
  src/cunibs/solver/fem/adm.cu
  src/cunibs/solver/fem/moments.cu
  src/cunibs/solver/fem/summary.cu
  src/cunibs/solver/fem/seed.cu
  src/cunibs/solver/coil/dadt.cu
  src/cunibs/solver/coil/dadt_element.cu
  src/cunibs/solver/coil/place.cu
)

target_include_directories(_solver_ext PRIVATE "${CMAKE_CURRENT_LIST_DIR}/src/cunibs/solver")

set_source_files_properties(src/cunibs/solver/coil/dadt.cu
  PROPERTIES COMPILE_OPTIONS "--use_fast_math")
# Left to itself, nvcc fuses the frame's fp64 products into FMAs differently per architecture, so
# a placement's frame would differ by an ulp between GPU models. The kernel runs once per placement.
set_source_files_properties(src/cunibs/solver/coil/place.cu
  PROPERTIES COMPILE_OPTIONS "--fmad=false")

if(MSVC)
  # solver.hpp and several kernels carry UTF-8 bytes; MSVC otherwise decodes the sources
  # in the system ANSI code page.
  # CCCL refuses to compile under cl.exe's legacy preprocessor.
  set(_msvc_flags /utf-8 /Zc:preprocessor)
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
  # nanobind hides the C++ sources' symbols only. Exported, a .cu file's would include CCCL's unique
  # globals, which bind across every library in the process.
  target_compile_options(_solver_ext PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${_warn_flags}>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${_warn_flags_nvcc},-fvisibility=hidden>")
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
)

# Type checkers resolve cunibs to src/ and cannot follow the editable install's import hook, so a
# stub that exists only in site-packages is invisible to them. The copy in src/ is gitignored, and
# scikit-build-core leaves ignored files out of the wheel, so the wheel takes the stub from install.
add_custom_target(_solver_ext_stub_sync ALL
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
          "${CMAKE_CURRENT_BINARY_DIR}/solver/_solver_ext.pyi"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/cunibs/solver/_solver_ext.pyi"
  VERBATIM
)
add_dependencies(_solver_ext_stub_sync _solver_ext_stub)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/solver/_solver_ext.pyi" DESTINATION cunibs/solver)
