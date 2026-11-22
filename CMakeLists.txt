cmake_minimum_required(VERSION 3.24)

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

message(STATUS "cunibs: CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

enable_language(CUDA)

find_package(CUDAToolkit 12.9 REQUIRED)

# The kernels need CCCL 3. CTK 13 bundles it, and CTK 12 bundles 2.8, so a CTK 12 build
# downloads the release instead.
find_package(CCCL 3.2 CONFIG QUIET HINTS "${CUDAToolkit_LIBRARY_ROOT}")
if(NOT CCCL_FOUND)
  include(FetchContent)
  FetchContent_Declare(cccl
    URL https://github.com/NVIDIA/cccl/releases/download/v3.2.1/cccl-v3.2.1.tar.gz
    URL_HASH SHA256=db8eeb780df099e3e8bb5d0019a9b37533e5fad7c1f80676c4975f290e2c1e4a
    DOWNLOAD_EXTRACT_TIMESTAMP ON)
  FetchContent_MakeAvailable(cccl)
  find_package(CCCL 3.2 CONFIG REQUIRED PATHS "${cccl_SOURCE_DIR}/lib/cmake" NO_DEFAULT_PATH)
endif()
message(STATUS "cunibs: CCCL ${CCCL_VERSION} from ${CCCL_DIR}")

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

# A static cudart needs only the driver at run time, so a CTK 12 build loads next to the
# CUDA 12 and CUDA 13 CuPy alike. CCCL comes first so its headers shadow the toolkit's.
set_target_properties(_solver_ext PROPERTIES CUDA_RUNTIME_LIBRARY Static)
target_link_libraries(_solver_ext PRIVATE CCCL::CCCL CUDA::cudart_static)

install(TARGETS _solver_ext
  RUNTIME DESTINATION cunibs/solver
  LIBRARY DESTINATION cunibs/solver NAMELINK_SKIP
  ARCHIVE DESTINATION ${SKBUILD_NULL_DIR}
)

nanobind_add_stub(
  _solver_ext_stub
  MODULE _solver_ext
  OUTPUT _solver_ext.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:_solver_ext>
  DEPENDS _solver_ext
)

# Type checkers resolve cunibs to src/ and cannot follow the editable install's import hook, so a
# stub that exists only in site-packages is invisible to them. The copy in src/ is gitignored, and
# scikit-build-core leaves ignored files out of the wheel, so the wheel takes the stub from install.
add_custom_target(_solver_ext_stub_sync ALL
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
          "${CMAKE_CURRENT_BINARY_DIR}/_solver_ext.pyi"
          "${CMAKE_CURRENT_SOURCE_DIR}/src/cunibs/solver/_solver_ext.pyi"
  VERBATIM
)
add_dependencies(_solver_ext_stub_sync _solver_ext_stub)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_solver_ext.pyi" DESTINATION cunibs/solver)
