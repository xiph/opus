# Copyright (C) 2023 Grinn sp. z o.o.
#
# These computer program listings and specifications, are the property of
# Grinn sp. z o.o. and shall not be reproduced or copied or used in
# whole or in part without written permission from Grinn sp. z o.o.

if(CONFIG_LIBOPUS)

zephyr_library_named(opus)

zephyr_compile_definitions(
  ARM_MATH_CM4
  VAR_ARRAYS
  OPUS_BUILD
  OPUS_ARM_INLINE_MEDIA
  OPUS_ARM_MAY_HAVE_MEDIA
  OPUS_ARM_PRESUME_MEDIA
  OPUS_ARM_INLINE_ASM
  HAVE_LRINTF
  CELT_MODE=1
  OPUS_BITRATE=24000
  OPUS_ARM_ASM
  SKIP_SYSCLK_INIT
  OPUS_CBR=1
  DISABLE_FLOAT_API
  FIXED_POINT=1
)

zephyr_library_cc_option(-Wno-double-promotion)

zephyr_library_sources(
  ${ZEPHYR_CURRENT_MODULE_DIR}/src/opus.c
  ${ZEPHYR_CURRENT_MODULE_DIR}/src/opus_decoder.c
  ${ZEPHYR_CURRENT_MODULE_DIR}/src/opus_encoder.c
  ${ZEPHYR_CURRENT_MODULE_DIR}/src/repacketizer.c
  ${ZEPHYR_CURRENT_MODULE_DIR}/celt/bands.c
  ${ZEPHYR_CURRENT_MODULE_DIR}/celt/celt.c
  ${ZEPHYR_CURRENT_MODULE_DIR}/celt/celt_encoder.c
  ${ZEPHYR_CURRENT_MODULE_DIR}/celt/celt_decoder.c
  ${ZEPHYR_CURRENT_MODULE_DIR}/celt/cwrs.c
  ${ZEPHYR_CURRENT_MODULE_DIR}/celt/entcode.c
  ${ZEPHYR_CURRENT_MODULE_DIR}/celt/entdec.c
  ${ZEPHYR_CURRENT_MODULE_DIR}/celt/entenc.c
  ${ZEPHYR_CURRENT_MODULE_DIR}/celt/kiss_fft.c
  ${ZEPHYR_CURRENT_MODULE_DIR}/celt/laplace.c
  ${ZEPHYR_CURRENT_MODULE_DIR}/celt/mathops.c
  ${ZEPHYR_CURRENT_MODULE_DIR}/celt/mdct.c
  ${ZEPHYR_CURRENT_MODULE_DIR}/celt/modes.c
  ${ZEPHYR_CURRENT_MODULE_DIR}/celt/pitch.c
  ${ZEPHYR_CURRENT_MODULE_DIR}/celt/celt_lpc.c
  ${ZEPHYR_CURRENT_MODULE_DIR}/celt/quant_bands.c
  ${ZEPHYR_CURRENT_MODULE_DIR}/celt/rate.c
  ${ZEPHYR_CURRENT_MODULE_DIR}/celt/vq.c
  ${ZEPHYR_CURRENT_MODULE_DIR}/celt/arm/armcpu.c
  ${ZEPHYR_CURRENT_MODULE_DIR}/celt/arm/arm_celt_map.c
)

zephyr_compile_options(-O2)

zephyr_include_directories(
  ${ZEPHYR_CURRENT_MODULE_DIR}/include
)

target_include_directories(opus PRIVATE
  ${ZEPHYR_CURRENT_MODULE_DIR}/silk
  ${ZEPHYR_CURRENT_MODULE_DIR}/celt
)

endif()
