cmake_minimum_required(VERSION 3.1)

include(opus_functions.cmake)

get_library_version(OPUS_LIBRARY_VERSION OPUS_LIBRARY_VERSION_MAJOR)
message(STATUS "Opus library version: ${OPUS_LIBRARY_VERSION}")

get_package_version(PACKAGE_VERSION)
message(STATUS "Opus package version: ${PACKAGE_VERSION}")

string(REGEX REPLACE "^v([0-9]+.[0-9]+\\.?([0-9]+)?).*" "\\1" PROJECT_VERSION ${PACKAGE_VERSION})

project(Opus LANGUAGES C VERSION ${PROJECT_VERSION})

include(opus_config.cmake)
include(opus_sources.cmake)
include(GNUInstallDirs)

add_library(opus ${opus_sources} ${opus_sources_float})

set(Opus_PUBLIC_HEADER
  ${CMAKE_CURRENT_SOURCE_DIR}/include/opus.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/opus_custom.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/opus_defines.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/opus_multistream.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/opus_projection.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/opus_types.h
)

set_target_properties(opus
                      PROPERTIES SOVERSION
                                 ${OPUS_LIBRARY_VERSION_MAJOR}
                                 VERSION
                                 ${OPUS_LIBRARY_VERSION}
                                 PUBLIC_HEADER "${Opus_PUBLIC_HEADER}"
                                 )

target_include_directories(
  opus
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        celt
        silk)

target_link_libraries(opus PRIVATE ${CMAKE_REQUIRED_LIBRARIES})

target_compile_definitions(opus PRIVATE CUSTOM_MODES OPUS_BUILD)

if(BUILD_SHARED_LIBS AND MSVC)
  target_compile_definitions(opus PRIVATE DLL_EXPORT)
elseif(BUILD_SHARED)
  include(CheckCCompilerFlag)
  check_c_compiler_flag(-fvisibility=hidden COMPILER_HAS_HIDDEN_VISIBILITY)
  if(COMPILER_HAS_HIDDEN_VISIBILITY)
    set_target_properties(opus PROPERTIES C_VISIBILITY_PRESET hidden)
  endif()
endif()

add_sources_group(opus silk ${silk_sources})
add_sources_group(opus celt ${celt_sources})

if(FIXED_POINT)
  add_sources_group(opus silk ${silk_sources_fixed})
  target_include_directories(
    opus
    PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/silk/fixed>)
  target_compile_definitions(opus PRIVATE FIXED_POINT=1 DISABLE_FLOAT_API)
else()
  add_sources_group(opus silk ${silk_sources_float})
  target_include_directories(
    opus
    PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/silk/float>)
endif()

if(HAVE_SSE)

  add_sources_group(opus celt ${celt_sources_sse})
  add_sources_group(opus celt ${celt_sources_sse2})

  if(SSE4_FLAG_SUPPORTED)
    add_sources_group(opus celt ${celt_sources_sse4_1})
    if(FIXED_POINT)
      add_sources_group(opus silk ${silk_sources_fixed_sse4_1})
    else()
      add_sources_group(opus silk ${silk_sources_sse4_1})
    endif()
  endif()

  if(OPUS_MAY_HAVE_SSE)
    # Do runtime check for SIMD
    target_compile_definitions(opus
                               PRIVATE
                               OPUS_X86_MAY_HAVE_SSE
                               OPUS_X86_MAY_HAVE_SSE2
                               OPUS_HAVE_RTCD)
    if(SSE4_FLAG_SUPPORTED)
      target_compile_definitions(opus PRIVATE OPUS_X86_MAY_HAVE_SSE4_1)
    endif()

  elseif(OPUS_PRESUME_SSE)
    target_compile_definitions(opus
                               PRIVATE
                               OPUS_X86_PRESUME_HAVE_SSE
                               OPUS_X86_PRESUME_HAVE_SSE2)
    if(SSE4_FLAG_SUPPORTED)
      target_compile_definitions(opus PRIVATE OPUS_X86_PRESUME_SSE4_1)
    endif()
  endif()
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(armv7-a)")
  add_sources_group(opus celt ${celt_sources_arm})
endif()

if(HAVE_NEON)

  add_sources_group(opus celt ${celt_sources_arm_neon_intr})
  add_sources_group(opus silk ${silk_sources_arm_neon_intr})

  # silk arm neon depends on main_Fix.h
  target_include_directories(opus PRIVATE silk/fixed)

  if(FIXED_POINT)
    add_sources_group(opus silk ${silk_sources_fixed_arm_neon_intr})
  endif()

  if(OPUS_MAY_HAVE_NEON)
    # Do runtime check for NEON
    target_compile_definitions(opus
                               PRIVATE
                               OPUS_ARM_MAY_HAVE_NEON
                               OPUS_ARM_MAY_HAVE_NEON_INTR
                               OPUS_HAVE_RTCD)
  elseif(OPUS_PRESUME_NEON)
    target_compile_definitions(opus
                               PRIVATE
                               OPUS_ARM_PRESUME_NEON
                               OPUS_ARM_PRESUME_NEON_INTR)
  endif()
endif()

install(TARGETS opus
        EXPORT OpusTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/opus)

set(CMAKE_INSTALL_PACKAGEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
install(
  EXPORT OpusTargets
  NAMESPACE Opus::
  DESTINATION ${CMAKE_INSTALL_PACKAGEDIR}
)

include(CMakePackageConfigHelpers)

set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})
configure_package_config_file(OpusConfig.cmake.in OpusConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_PACKAGEDIR}
  PATH_VARS
    INCLUDE_INSTALL_DIR
  INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}
)
write_basic_package_version_file(OpusConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/OpusConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/OpusConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_PACKAGEDIR}
)

if(BUILD_PROGRAMS)
  # demo
  add_executable(opus_demo ${opus_demo_sources})
  target_include_directories(opus_demo PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(opus_demo PRIVATE opus)

  # compare
  add_executable(opus_compare ${opus_compare_sources})
  target_include_directories(opus_compare PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(opus_compare PRIVATE opus)
endif()

if(BUILD_TESTING)
  enable_testing()

  # tests
  add_executable(test_opus_decode ${test_opus_decode_sources})
  target_include_directories(test_opus_decode PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(test_opus_decode PRIVATE opus)
  if(FIXED_POINT)
    target_compile_definitions(test_opus_decode PRIVATE DISABLE_FLOAT_API)
  endif()
  add_test(test_opus_decode test_opus_decode)

  add_executable(test_opus_padding ${test_opus_padding_sources})
  target_include_directories(test_opus_padding PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(test_opus_padding PRIVATE opus)
  add_test(test_opus_padding test_opus_padding)

  if(NOT BUILD_SHARED_LIBS)
    # disable tests that depends on private API when building shared lib
    add_executable(test_opus_api ${test_opus_api_sources})
    target_include_directories(test_opus_api
                               PRIVATE 
                                      ${CMAKE_CURRENT_BINARY_DIR}
                                      celt)
    target_link_libraries(test_opus_api PRIVATE opus)
    if(FIXED_POINT)
      target_compile_definitions(test_opus_api PRIVATE DISABLE_FLOAT_API)
    endif()
    add_test(test_opus_api test_opus_api)

    add_executable(test_opus_encode ${test_opus_encode_sources})
    target_include_directories(test_opus_encode
                               PRIVATE
                                      ${CMAKE_CURRENT_BINARY_DIR}
                                      celt)
    target_link_libraries(test_opus_encode PRIVATE opus)
    add_test(test_opus_encode test_opus_encode)
  endif()
endif()
