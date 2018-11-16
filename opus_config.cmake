include(opus_functions.cmake)

configure_file(config.h.cmake.in config.h @ONLY)
add_definitions(-DHAVE_CONFIG_H)

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
  list(APPEND OPUS_REQUIRED_LIBRARIES m)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(i[0-9]86|x86|X86|amd64|AMD64|x86_64)")
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(OPUS_CPU_X64 1)
  else()
    set(OPUS_CPU_X86 1)
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
  set(OPUS_CPU_ARM 1)
endif()

opus_supports_cpu_detection(RUNTIME_CPU_CAPABILITY_DETECTION)

if(OPUS_CPU_X86 OR OPUS_CPU_X64)
  opus_detect_sse(COMPILER_SUPPORT_SIMD)
elseif(OPUS_CPU_ARM)
  opus_detect_neon(COMPILER_SUPPORT_NEON)
  if(COMPILER_SUPPORT_NEON)
    option(OPUS_USE_NEON "Option to turn off SSE" ON)
    option(OPUS_MAY_SUPPORT_NEON "Does runtime check for neon support" ON)
    option(OPUS_PRESUME_NEON "Assume target CPU has NEON support" OFF)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
      set(OPUS_PRESUME_NEON ON)
    endif()
  endif()
endif()
