cmake_minimum_required(VERSION 2.8.12)

project(opus)
enable_language(C)
enable_testing()

include(opus.cmake)
include(CheckIncludeFiles)

set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif ()

execute_process(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE UNAME_MACHINE)
if (${UNAME_MACHINE} MATCHES ".86")
    add_definitions(-fPIC)
endif ()
set(ARCH ${UNAME_MACHINE} CACHE STRING "System architecture (retrieved using \"uname -m\")")

set(OPUS_CFLAGS "-Wall")

set(ENABLE_SHARED FALSE CACHE BOOL "Build shared library")
set(ENABLE_STATIC TRUE CACHE BOOL "Build static library")

add_definitions(-DOPUS_BUILD=1)

set(ENABLE_C99_ARRAYS TRUE CACHE BOOL "Enable C99 variable-size arrays")
if (ENABLE_C99_ARRAYS)
    add_definitions(-DVAR_ARRAYS=1)
else ()
    check_include_files(alloca.h HAVE_ALLOCA_H)
    if (HAVE_ALLOCA_H)
        message(STATUS "Enabling alloca.h")
        add_definitions(-DUSE_ALLOCA=1)
    endif (HAVE_ALLOCA_H)
endif ()

set(ENABLE_FIXED_POINT FALSE CACHE BOOL "Compile without the floating point API (for machines with no float library)")
if (ENABLE_FIXED_POINT)
    add_definitions(-DFIXED_POINT=1)
endif ()

set(ENABLE_FLOAT_APPROX FALSE CACHE BOOL "Enable fast approximations for floating point")
if (ENABLE_FLOAT_APPROX)
    message(WARNING "Floating point approximations are not supported on all platforms")
    add_definitions(-DFLOAT_APPROX)
endif (ENABLE_FLOAT_APPROX)

set(ENABLE_CUSTOM_MODES FALSE CACHE BOOL "Enable non-Opus modes, e.g. 44.1 kHz & 2^n frames")
if (ENABLE_CUSTOM_MODES)
    add_definitions(-DCUSTOM_MODES=1)
endif ()

set(ENABLE_ASM TRUE CACHE BOOL "Enable assembly optimizations")

set(ENABLE_ASM_SSE TRUE CACHE BOOL "Enable SSE")
if (ENABLE_ASM_SSE)
    add_definitions(-DOPUS_X86_MAY_HAVE_SSE=1)
    set(OPUS_CFLAGS "${OPUS_CFLAGS} -msse")
endif ()

set(ENABLE_ASM_SSE2 TRUE CACHE BOOL "Enable SSE2")
if (ENABLE_ASM_SSE2)
    add_definitions(-DOPUS_X86_MAY_HAVE_SSE2=1)
    set(OPUS_CFLAGS "${OPUS_CFLAGS} -msse2")
endif ()

set(ENABLE_ASM_SSE41 TRUE CACHE BOOL "Enable SSE41")
if (ENABLE_ASM_SSE41)
    add_definitions(-DOPUS_X86_MAY_HAVE_SSE4_1=1)
    set(OPUS_CFLAGS "${OPUS_CFLAGS} -msse4.1")
endif ()

set(ENABLE_ASM_AVX TRUE CACHE BOOL "Enable AVX")
if (ENABLE_ASM_AVX)
    add_definitions(-DOPUS_X86_MAY_HAVE_AVX=1)
    set(OPUS_CFLAGS "${OPUS_CFLAGS} -mavx")
endif ()

set(ENABLE_ASM_NEON FALSE CACHE BOOL "Enable NEON")
if (ENABLE_ASM_NEON)
    add_definitions(-DOPUS_ARM_ASM=1)
endif ()

set(ENABLE_RTCD FALSE CACHE BOOL "Enable run-time CPU capabilities detection")
if (ENABLE_RTCD)
    add_definitions(-DOPUS_HAVE_RTCD=1)
endif ()

set(ENABLE_INTRINSICS TRUE CACHE BOOL "Enable intrinsics optimizations")

set(ENABLE_ASSERTIONS FALSE CACHE BOOL "Enable additional software error checking")
if (ENABLE_ASSERTIONS)
    add_definitions(-DENABLE_ASSERTIONS=1)
endif ()

set(ENABLE_FUZZING FALSE CACHE BOOL "Causes the encoder to make random decisions")
if (ENABLE_FUZZING)
    add_definitions(-DFUZZING=1)
endif ()

set(ENABLE_EXPERIMENTAL_AMBISONICS FALSE CACHE BOOL "Enable experimental ambisonic encoding and decoding support")
if (ENABLE_EXPERIMENTAL_AMBISONICS)
    add_definitions(-DENABLE_EXPERIMENTAL_AMBISONICS=1)
endif ()

set(ENABLE_DOC FALSE CACHE BOOL "Build API documentation")
set(ENABLE_EXTRA_PROGRAMS FALSE CACHE BOOL "Build extra programs (demo and tests)")

set(OPUS_SOURCES
        src/opus.c
        src/opus_decoder.c
        src/opus_encoder.c
        src/opus_multistream.c
        src/opus_multistream_encoder.c
        src/opus_multistream_decoder.c
        src/repacketizer.c)

set(OPUS_SOURCES_FLOAT
        src/analysis.c
        src/mlp.c
        src/mlp_data.c)

if (NOT FIXED_POINT)
    set(OPUS_SOURCES ${OPUS_SOURCES} ${OPUS_SOURCES_FLOAT})
endif ()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OPUS_CFLAGS}")
set(CMAKE_C_FLAGS_DEBUG "${OPUS_CFLAGS_DEBUG} ${OPUS_CFLAGS}")
set(CMAKE_C_FLAGS_MINSIZEREL "${OPUS_CFLAGS_MINSIZEREL} ${OPUS_CFLAGS}")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "${OPUS_CFLAGS_RELWITHDEBINFO} ${OPUS_CFLAGS}")
set(CMAKE_C_FLAGS_RELEASE "${OPUS_CFLAGS_RELEASE} ${OPUS_CFLAGS}")

add_subdirectory(celt)
add_subdirectory(silk)

include_directories(include)

if (ENABLE_SHARED)
    add_library(opus SHARED ${OPUS_SOURCES})
    target_link_libraries(opus celt silk)
endif ()

if (ENABLE_STATIC)
    add_library(opusstatic STATIC ${OPUS_SOURCES})
    target_link_libraries(opusstatic celt silk)
endif ()

print_configuration()
