# Locate the CUDA toolkit through cuda.pathfinder (PyPI wheels -> conda ->
# system -> CUDA_HOME) rather than assuming a filesystem layout.

function(cunibs_probe_cuda)
  cmake_parse_arguments(arg "" "" "LIBS" ${ARGN})

  set(_probe "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/cuda_pathfinder_probe.py")
  list(JOIN arg_LIBS "," _libs)

  set(_flags "")
  if(NOT WIN32)
    # The Linux wheels ship only versioned SONAMEs (libcudart.so.13) with no .so
    # development symlink, so we link the absolute versioned file instead of
    # letting find_library look for a name that does not exist. The Windows
    # wheels ship unversioned .lib import libraries, which FindCUDAToolkit
    # locates without help.
    list(APPEND _flags --emit-lib-paths)
  endif()

  execute_process(
    COMMAND "${Python_EXECUTABLE}" "${_probe}" --libs "${_libs}" ${_flags}
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

  string(SHA256 _fingerprint "${_stdout}")
  set(CUNIBS_CUDA_FINGERPRINT "${_fingerprint}" PARENT_SCOPE)
endfunction()

# Seed FindCUDAToolkit so it never needs a development symlink, and pre-create
# the imported targets so it cannot widen their interfaces with libraries from
# an unrelated system toolkit. FindCUDAToolkit guards the whole target setup
# with `if(NOT TARGET CUDA::<lib> AND CUDA_<lib>_LIBRARY)`, so targets created
# here survive untouched.
function(cunibs_seed_cuda_toolkit)
  cmake_parse_arguments(arg "" "INCLUDE_DIR;FINGERPRINT" "LIBS" ${ARGN})

  # scikit-build-core reuses build/{wheel_tag} across invocations, so a plain
  # set(... CACHE ...) would be a no-op against the entries a previous
  # environment left behind. Only FORCE when the resolved toolkit actually
  # changed, which leaves manual -DCUDA_<lib>_LIBRARY= overrides intact.
  set(_force "")
  if(NOT arg_FINGERPRINT STREQUAL "${CUNIBS_CUDA_SEEDED_FINGERPRINT}")
    set(_force FORCE)
    set(CUNIBS_CUDA_SEEDED_FINGERPRINT "${arg_FINGERPRINT}"
      CACHE INTERNAL "fingerprint of the CUDA toolkit seeded into the cache" FORCE)
  endif()

  foreach(_lib IN LISTS arg_LIBS)
    set(_path "${CUNIBS_CUDA_LIB_${_lib}}")
    if(NOT _path)
      message(FATAL_ERROR "The cuda.pathfinder probe reported no path for '${_lib}'")
    endif()
    set(CUDA_${_lib}_LIBRARY "${_path}" CACHE FILEPATH "CUDA ${_lib} library" ${_force})
    if(NOT TARGET CUDA::${_lib})
      add_library(CUDA::${_lib} UNKNOWN IMPORTED GLOBAL)
      set_target_properties(CUDA::${_lib} PROPERTIES
        IMPORTED_LOCATION "${_path}"
        INTERFACE_INCLUDE_DIRECTORIES "${arg_INCLUDE_DIR}")
    endif()
  endforeach()

  # A REQUIRED_VARS entry of find_package_handle_standard_args, and the anchor
  # FindCUDAToolkit derives CUDAToolkit_LIBRARY_DIR from. Distinct from
  # CUDA_cudart_LIBRARY above.
  set(CUDA_CUDART "${CUNIBS_CUDA_LIB_cudart}" CACHE FILEPATH "CUDA toolkit anchor" ${_force})

  # FindCUDAToolkit's dependency edges, minus culibos: it is a static library
  # that the wheels do not ship, so leaving it to FindCUDAToolkit means linking
  # a copy from whatever system toolkit happens to be installed.
  set_property(TARGET CUDA::cublas APPEND PROPERTY
    INTERFACE_LINK_LIBRARIES CUDA::cublasLt)
  set_property(TARGET CUDA::cusparse APPEND PROPERTY
    INTERFACE_LINK_LIBRARIES CUDA::nvJitLink)
  set_property(TARGET CUDA::cusolver APPEND PROPERTY
    INTERFACE_LINK_LIBRARIES CUDA::cublas CUDA::cusparse CUDA::cublasLt CUDA::nvJitLink)
endfunction()
