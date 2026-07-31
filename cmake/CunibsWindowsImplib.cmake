# Synthesize MSVC import libraries for the CUDA wheel DLLs.
#
# The Windows wheels ship no usable link-time artifact: the cudart.lib they contain imports
# from the driver-side nvcudart_hybrid64.dll rather than from the cudart64_*.dll in the wheel,
# so linking against it would resolve to a DLL we never load. Derive the import libraries from
# the DLLs themselves instead.

function(cunibs_generate_cuda_implibs)
  cmake_parse_arguments(arg "" "" "LIBS" ${ARGN})

  set(_out_dir "${CMAKE_BINARY_DIR}/cuda_implib")
  set(_script "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/pe_def_from_dll.py")

  # lib.exe under MSVC, llvm-lib under clang-cl; both accept /def.
  if(NOT CMAKE_AR)
    message(FATAL_ERROR
      "Could not locate lib.exe (CMAKE_AR is empty). Configure from a Visual Studio "
      "developer shell, or pass -DCMAKE_AR=<path to lib.exe>.")
  endif()

  foreach(_lib IN LISTS arg_LIBS)
    set(_dll "${CUNIBS_CUDA_LIB_${_lib}}")
    if(NOT _dll OR NOT EXISTS "${_dll}")
      message(FATAL_ERROR "The cuda.pathfinder probe reported no DLL for '${_lib}'")
    endif()

    set(_def "${_out_dir}/${_lib}.def")
    set(_implib "${_out_dir}/${_lib}.lib")
    execute_process(
      COMMAND "${Python_EXECUTABLE}" "${_script}" --dll "${_dll}" --out "${_def}"
      RESULT_VARIABLE _result
      ERROR_VARIABLE _stderr)
    if(NOT _result EQUAL 0)
      message(FATAL_ERROR "Could not extract the exports of '${_lib}':\n${_stderr}")
    endif()

    execute_process(
      COMMAND "${CMAKE_AR}" /nologo "/def:${_def}" /machine:x64 "/out:${_implib}"
      RESULT_VARIABLE _result
      OUTPUT_VARIABLE _stdout
      ERROR_VARIABLE _stderr)
    if(NOT _result EQUAL 0 OR NOT EXISTS "${_implib}")
      message(FATAL_ERROR "lib.exe failed for '${_lib}':\n${_stdout}${_stderr}")
    endif()

    message(STATUS "CUDA ${_lib}: generated import library ${_implib}")
    set(CUNIBS_CUDA_IMPLIB_${_lib} "${_implib}" PARENT_SCOPE)
  endforeach()
endfunction()
