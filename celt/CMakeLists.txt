include_directories(../include)
include_directories(../silk)

set(CELT_SOURCES
        bands.c
        celt.c
        celt_decoder.c
        celt_encoder.c
        celt_lpc.c
        cwrs.c
        entcode.c
        entdec.c
        entenc.c
        kiss_fft.c
        laplace.c
        mathops.c
        mdct.c
        modes.c
        pitch.c
        quant_bands.c
        rate.c
        vq.c)

set(CELT_ASM_SSE_SOURCES
        x86/x86cpu.c
        x86/x86_celt_map.c
        x86/pitch_sse.c)

set(CELT_ASM_SSE2_SOURCES
        x86/pitch_sse2.c
        x86/vq_sse2.c)

set(CELT_ASM_SSE41_SOURCES
        x86/celt_lpc_sse.c
        x86/pitch_sse4_1.c)

set(CELT_ARM_SOURCES
        arm/armcpu.c
        arm/arm_celt_map.c)

if (${UNAME_MACHINE} MATCHES "arm")
    set(CELT_SOURCES ${CELT_SOURCES} ${CELT_ARM_SOURCES})
endif ()

if (ENABLE_ASM)
    if (${ARCH} MATCHES ".86")
        if (ENABLE_ASM_SSE)
            set(CELT_SOURCES ${CELT_SOURCES} ${CELT_ASM_SSE_SOURCES})
        endif ()
        if (ENABLE_ASM_SSE2)
            set(CELT_SOURCES ${CELT_SOURCES} ${CELT_ASM_SSE2_SOURCES})
        endif ()
        if (ENABLE_ASM_SSE41)
            set(CELT_SOURCES ${CELT_SOURCES} ${CELT_ASM_SSE41_SOURCES})
        endif ()
    endif ()
endif ()

add_library(celt STATIC ${CELT_SOURCES})
target_include_directories(celt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (ENABLE_CUSTOM_MODES)
    add_executable(opus_custom_demo opus_custom_demo.c)
    add_test(opus_custom_demo opus_custom_demo)
    target_link_libraries(opus_custom_demo LINK_PUBLIC celt)
endif ()
