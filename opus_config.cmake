include(opus_functions.cmake)

configure_file(config.h.in config.h @ONLY)
add_definitions(-DHAVE_CONFIG_H)

opus_detect_sse(HAVE_SSE)
opus_detect_neon(HAVE_NEON)

include(CMakeDependentOption)
cmake_dependent_option(OPUS_PRESUME_SSE
                       "Use SSE always (requires CPU with SSE)"
                       ON
                       "HAVE_SSE;OPUS_PRESUME_SSE"
                       OFF)
cmake_dependent_option(OPUS_MAY_HAVE_SSE
                       "Use SSE if available"
                       ON
                       "HAVE_SSE;NOT OPUS_PRESUME_SSE"
                       OFF)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(OPUS_PRESUME_NEON ON)
endif()
cmake_dependent_option(OPUS_PRESUME_NEON
                       "Use NEON always (requires CPU with NEON Support)"
                       ON
                       "HAVE_NEON;OPUS_PRESUME_NEON"
                       OFF)
cmake_dependent_option(OPUS_MAY_HAVE_NEON
                       "Use NEON if available"
                       ON
                       "HAVE_NEON;NOT OPUS_PRESUME_NEON"
                       OFF)

option(FIXED_POINT "Use fixed-point code (for devices with less powerful FPU)"
       OFF)
option(USE_ALLOCA "Use alloca for stack arrays (on non-C99 compilers)" OFF)
option(BUILD_PROGRAMS "Build programs" OFF)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY C_STANDARD 99)

if(MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# It is strongly recommended to uncomment one of these VAR_ARRAYS: Use C99
# variable-length arrays for stack allocation USE_ALLOCA: Use alloca() for stack
# allocation If none is defined, then the fallback is a non-threadsafe global
# array
if(USE_ALLOCA OR MSVC)
  add_definitions(-DUSE_ALLOCA)
else()
  add_definitions(-DVAR_ARRAYS)
endif()

include(CheckLibraryExists)
check_library_exists(m floor "" HAVE_LIBM)
if(HAVE_LIBM)
  list(APPEND CMAKE_REQUIRED_LIBRARIES m)
endif()
