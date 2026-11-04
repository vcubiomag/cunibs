function(cunibs_probe_cuda)
  set(_probe "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/cuda_pathfinder_probe.py")

  execute_process(
    COMMAND "${Python_EXECUTABLE}" "${_probe}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(NOT _result EQUAL 0)
    message(FATAL_ERROR
      "The cuda.pathfinder probe failed (exit ${_result}).\n"
      "  interpreter: ${Python_EXECUTABLE}\n"
      "  probe:       ${_probe}\n"
      "${_stderr}")
  endif()

  string(REPLACE "\n" ";" _lines "${_stdout}")
  foreach(_line IN LISTS _lines)
    string(STRIP "${_line}" _line)
    if(_line STREQUAL "")
      continue()
    endif()
    if(NOT _line MATCHES "^([^=]+)=(.*)$")
      message(FATAL_ERROR "Unparseable cuda.pathfinder probe output: ${_line}")
    endif()
    # Latch both captures: the FOUND_VIA test below overwrites CMAKE_MATCH_*.
    set(_key "${CMAKE_MATCH_1}")
    set(_value "${CMAKE_MATCH_2}")
    set(CUNIBS_CUDA_${_key} "${_value}" PARENT_SCOPE)
    if(_key MATCHES "^FOUND_VIA_(.+)$")
      message(STATUS "CUDA ${CMAKE_MATCH_1}: found via ${_value}")
    endif()
  endforeach()
endfunction()
