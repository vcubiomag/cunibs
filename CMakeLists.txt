cmake_minimum_required(VERSION 3.21)

find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import nvidia.cu13 as m, os; print(os.path.join(m.__path__[0], 'bin', 'nvcc'))"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE CMAKE_CUDA_COMPILER
)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX CUDA)

get_filename_component(_CUDA_BIN_DIR "${CMAKE_CUDA_COMPILER}" DIRECTORY)
set(_CUDA_LIB_DIR "${_CUDA_BIN_DIR}/../lib")
set(_CUDA_SHIM_LIBS cudart cublas cusparse cusolver)
foreach(_lib ${_CUDA_SHIM_LIBS})
  if(EXISTS "${_CUDA_LIB_DIR}" AND NOT EXISTS "${_CUDA_LIB_DIR}/lib${_lib}.so")
    file(GLOB _versioned "${_CUDA_LIB_DIR}/lib${_lib}.so.[0-9]*")
    list(LENGTH _versioned _count)
    if(_count GREATER 0)
      list(GET _versioned 0 _first)
      file(CREATE_LINK "${_first}" "${_CUDA_LIB_DIR}/lib${_lib}.so" SYMBOLIC)
    endif()
  endif()
endforeach()

find_package(CUDAToolkit 13.2 REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Default build type" FORCE)
endif()

set(AMGX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(AMGX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(AMGX_BUILD_STATIC OFF CACHE BOOL "" FORCE)
set(AMGX_BUILD_SHARED ON  CACHE BOOL "" FORCE)
set(AMGX_INSTALL OFF CACHE BOOL "" FORCE)
set(CMAKE_NO_MPI ON CACHE BOOL "" FORCE)

add_subdirectory(external/AMGX) 

target_compile_options(amgx_libs PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-deprecated-declarations>
  $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_solver_ext LTO FREE_THREADED
  src/cunibs/solver/bindings.cpp
  src/cunibs/solver/solver.cpp
  src/cunibs/solver/dadt.cu
  src/cunibs/solver/rhs.cu
  src/cunibs/solver/reconstruct.cu
)
target_link_libraries(_solver_ext PRIVATE CUDA::cudart AMGX::amgxsh)

set_target_properties(_solver_ext amgxsh PROPERTIES
  INSTALL_RPATH "$ORIGIN;$ORIGIN/../../nvidia/cu13/lib"
)

get_target_property(_amgxsh_soversion amgxsh SOVERSION)
set_target_properties(amgxsh PROPERTIES VERSION ${_amgxsh_soversion})

install(TARGETS _solver_ext amgxsh
  LIBRARY DESTINATION cunibs/solver
  NAMELINK_SKIP
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
