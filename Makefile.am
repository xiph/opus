
#AM_CFLAGS = -Isrc -Iinterface -fPIC -Wall -O3 
NAME = libSKP_SILK_SDK
AM_CPPFLAGS = $(AM_CFLAGS)

INCLUDES = -I$(top_srcdir)/src_FLP -I$(top_srcdir)/src_SigProc_FIX -I$(top_srcdir)/src_SigProc_FLP -I$(top_srcdir)/src_common -I$(top_srcdir)/interface -I$(top_srcdir)/../celt/libcelt
EXTRA_DIST = Silk_FLP.sln \
Silk_FLP.sln \
src_FIX/src_FIX.vcxproj \
src_FIX/src_FIX.vcxproj.filters \
src_FLP/src_FLP.vcxproj \
src_FLP/src_FLP.vcxproj.filters \
src_common/src_common.vcxproj \
src_common/src_common.vcxproj.filters \
test/Encoder.c \
test/Decoder.c \
test/signalCompare.c \
src_SigProc_FLP/src_SigProc_FLP.vcxproj \
src_SigProc_FLP/src_SigProc_FLP.vcxproj.filters \
src_SigProc_FIX/src_SigProc_FIX.vcxproj \
src_SigProc_FIX/src_SigProc_FIX.vcxproj.filters \
test_silk_dec/test_silk_dec.vcxproj \
test_silk_dec/test_silk_dec.vcxproj.filters \
test_silk_enc/test_silk_enc.vcxproj \
test_silk_enc/test_silk_enc.vcxproj.filters

lib_LTLIBRARIES	= libSKP_SILK_SDK.la
libSKP_SILK_SDK_la_SOURCES = src_common/SKP_Silk_CNG.c \
src_common/SKP_Silk_code_signs.c \
src_common/SKP_Silk_create_init_destroy.c \
src_common/SKP_Silk_decode_core.c \
src_common/SKP_Silk_decode_frame.c \
src_common/SKP_Silk_decode_indices.c \
src_common/SKP_Silk_decode_parameters.c \
src_common/SKP_Silk_decode_pulses.c \
src_common/SKP_Silk_decoder_set_fs.c \
src_common/SKP_Silk_dec_API.c \
src_common/SKP_Silk_detect_SWB_input.c \
src_common/SKP_Silk_enc_API.c \
src_common/SKP_Silk_encode_parameters.c \
src_common/SKP_Silk_encode_pulses.c \
src_common/SKP_Silk_gain_quant.c \
src_common/SKP_Silk_interpolate.c \
src_common/SKP_Silk_LBRR_reset.c \
src_common/SKP_Silk_LP_variable_cutoff.c \
src_common/SKP_Silk_NLSF2A_stable.c \
src_common/SKP_Silk_NLSF_MSVQ_decode.c \
src_common/SKP_Silk_NSQ.c \
src_common/SKP_Silk_NSQ_del_dec.c \
src_common/SKP_Silk_PLC.c \
src_common/SKP_Silk_pulses_to_bytes.c \
src_common/SKP_Silk_range_coder.c \
src_common/SKP_Silk_shell_coder.c \
src_common/SKP_Silk_tables_gain.c \
src_common/SKP_Silk_tables_LTP.c \
src_common/SKP_Silk_tables_NLSF_CB0_10.c \
src_common/SKP_Silk_tables_NLSF_CB0_16.c \
src_common/SKP_Silk_tables_NLSF_CB1_10.c \
src_common/SKP_Silk_tables_NLSF_CB1_16.c \
src_common/SKP_Silk_tables_other.c \
src_common/SKP_Silk_tables_pitch_lag.c \
src_common/SKP_Silk_tables_pulses_per_block.c \
src_common/SKP_Silk_tables_sign.c \
src_common/SKP_Silk_VAD.c \
src_common/SKP_Silk_control_audio_bandwidth.c \
src_common/SKP_Silk_quant_LTP_gains.c \
src_common/SKP_Silk_VQ_WMat_EC.c \
src_FLP/SKP_Silk_apply_sine_window_FLP.c \
src_FLP/SKP_Silk_control_codec_FLP.c \
src_FLP/SKP_Silk_corrMatrix_FLP.c \
src_FLP/SKP_Silk_encode_frame_FLP.c \
src_FLP/SKP_Silk_find_LPC_FLP.c \
src_FLP/SKP_Silk_find_LTP_FLP.c \
src_FLP/SKP_Silk_find_pitch_lags_FLP.c \
src_FLP/SKP_Silk_find_pred_coefs_FLP.c \
src_FLP/SKP_Silk_HP_variable_cutoff_FLP.c \
src_FLP/SKP_Silk_init_encoder_FLP.c \
src_FLP/SKP_Silk_LPC_analysis_filter_FLP.c \
src_FLP/SKP_Silk_LTP_analysis_filter_FLP.c \
src_FLP/SKP_Silk_LTP_scale_ctrl_FLP.c \
src_FLP/SKP_Silk_NLSF_MSVQ_decode_FLP.c \
src_FLP/SKP_Silk_NLSF_MSVQ_encode_FLP.c \
src_FLP/SKP_Silk_NLSF_VQ_rate_distortion_FLP.c \
src_FLP/SKP_Silk_NLSF_VQ_sum_error_FLP.c \
src_FLP/SKP_Silk_NLSF_VQ_weights_laroia_FLP.c \
src_FLP/SKP_Silk_noise_shape_analysis_FLP.c \
src_FLP/SKP_Silk_prefilter_FLP.c \
src_FLP/SKP_Silk_process_gains_FLP.c \
src_FLP/SKP_Silk_process_NLSFs_FLP.c \
src_FLP/SKP_Silk_regularize_correlations_FLP.c \
src_FLP/SKP_Silk_residual_energy_FLP.c \
src_FLP/SKP_Silk_solve_LS_FLP.c \
src_FLP/SKP_Silk_warped_autocorrelation_FLP.c \
src_FLP/SKP_Silk_wrappers_FLP.c \
src_SigProc_FIX/SKP_Silk_A2NLSF.c \
src_SigProc_FIX/SKP_Silk_allpass_int.c \
src_SigProc_FIX/SKP_Silk_ana_filt_bank_1.c \
src_SigProc_FIX/SKP_Silk_apply_sine_window.c \
src_SigProc_FIX/SKP_Silk_array_maxabs.c \
src_SigProc_FIX/SKP_Silk_autocorr.c \
src_SigProc_FIX/SKP_Silk_biquad_alt.c \
src_SigProc_FIX/SKP_Silk_biquad.c \
src_SigProc_FIX/SKP_Silk_burg_modified.c \
src_SigProc_FIX/SKP_Silk_bwexpander_32.c \
src_SigProc_FIX/SKP_Silk_bwexpander.c \
src_SigProc_FIX/SKP_Silk_decode_pitch.c \
src_SigProc_FIX/SKP_Silk_inner_prod_aligned.c \
src_SigProc_FIX/SKP_Silk_k2a.c \
src_SigProc_FIX/SKP_Silk_k2a_Q16.c \
src_SigProc_FIX/SKP_Silk_lin2log.c \
src_SigProc_FIX/SKP_Silk_log2lin.c \
src_SigProc_FIX/SKP_Silk_lowpass_int.c \
src_SigProc_FIX/SKP_Silk_lowpass_short.c \
src_SigProc_FIX/SKP_Silk_LPC_inv_pred_gain.c \
src_SigProc_FIX/SKP_Silk_LPC_stabilize.c \
src_SigProc_FIX/SKP_Silk_LPC_synthesis_filter.c \
src_SigProc_FIX/SKP_Silk_LPC_synthesis_order16.c \
src_SigProc_FIX/SKP_Silk_LSF_cos_table.c \
src_SigProc_FIX/SKP_Silk_MA.c \
src_SigProc_FIX/SKP_Silk_NLSF2A.c \
src_SigProc_FIX/SKP_Silk_NLSF_stabilize.c \
src_SigProc_FIX/SKP_Silk_NLSF_VQ_weights_laroia.c \
src_SigProc_FIX/SKP_Silk_pitch_analysis_core.c \
src_SigProc_FIX/SKP_Silk_pitch_est_tables.c \
src_SigProc_FIX/SKP_Silk_resampler.c \
src_SigProc_FIX/SKP_Silk_resampler_down2_3.c \
src_SigProc_FIX/SKP_Silk_resampler_down2.c \
src_SigProc_FIX/SKP_Silk_resampler_down3.c \
src_SigProc_FIX/SKP_Silk_resampler_private_AR2.c \
src_SigProc_FIX/SKP_Silk_resampler_private_ARMA4.c \
src_SigProc_FIX/SKP_Silk_resampler_private_copy.c \
src_SigProc_FIX/SKP_Silk_resampler_private_down4.c \
src_SigProc_FIX/SKP_Silk_resampler_private_down_FIR.c \
src_SigProc_FIX/SKP_Silk_resampler_private_IIR_FIR.c \
src_SigProc_FIX/SKP_Silk_resampler_private_up2_HQ.c \
src_SigProc_FIX/SKP_Silk_resampler_private_up4.c \
src_SigProc_FIX/SKP_Silk_resampler_rom.c \
src_SigProc_FIX/SKP_Silk_resampler_up2.c \
src_SigProc_FIX/SKP_Silk_scale_copy_vector16.c \
src_SigProc_FIX/SKP_Silk_scale_vector.c \
src_SigProc_FIX/SKP_Silk_schur64.c \
src_SigProc_FIX/SKP_Silk_schur.c \
src_SigProc_FIX/SKP_Silk_sigm_Q15.c \
src_SigProc_FIX/SKP_Silk_sort.c \
src_SigProc_FIX/SKP_Silk_sum_sqr_shift.c \
src_SigProc_FLP/SKP_Silk_allpass_int_FLP.c \
src_SigProc_FLP/SKP_Silk_autocorrelation_FLP.c \
src_SigProc_FLP/SKP_Silk_burg_modified_FLP.c \
src_SigProc_FLP/SKP_Silk_bwexpander_FLP.c \
src_SigProc_FLP/SKP_Silk_decimate2_coarse_FLP.c \
src_SigProc_FLP/SKP_Silk_decimate2_coarsest_FLP.c \
src_SigProc_FLP/SKP_Silk_energy_FLP.c \
src_SigProc_FLP/SKP_Silk_inner_product_FLP.c \
src_SigProc_FLP/SKP_Silk_k2a_FLP.c \
src_SigProc_FLP/SKP_Silk_levinsondurbin_FLP.c \
src_SigProc_FLP/SKP_Silk_LPC_inv_pred_gain_FLP.c \
src_SigProc_FLP/SKP_Silk_pitch_analysis_core_FLP.c \
src_SigProc_FLP/SKP_Silk_scale_copy_vector_FLP.c \
src_SigProc_FLP/SKP_Silk_scale_vector_FLP.c \
src_SigProc_FLP/SKP_Silk_schur_FLP.c \
src_SigProc_FLP/SKP_Silk_sort_FLP.c


libSKP_SILK_SDK_la_CFLAGS = $(AM_CFLAGS)
libSKP_SILK_SDK_la_LDFLAGS = $(LIBS)

library_includedir = $(prefix)/include/silk
library_include_HEADERS	= interface/SKP_debug.h \
interface/SKP_Silk_control.h \
interface/SKP_Silk_errors.h \
interface/SKP_Silk_SDK_API.h \
interface/SKP_Silk_typedef.h \
src_common/SKP_Silk_define.h \
src_common/SKP_Silk_main.h \
src_common/SKP_Silk_perceptual_parameters.h \
src_common/SKP_Silk_PLC.h \
src_common/SKP_Silk_structs.h \
src_common/SKP_Silk_tables.h \
src_common/SKP_Silk_tables_NLSF_CB.h \
src_common/SKP_Silk_tuning_parameters.h \
src_common/SKP_Silk_setup_complexity.h \
src_FIX/SKP_Silk_main_FIX.h \
src_FIX/SKP_Silk_structs_FIX.h \
src_FLP/SKP_Silk_assembler_FLP.h \
src_FLP/SKP_Silk_main_FLP.h \
src_FLP/SKP_Silk_structs_FLP.h \
src_SigProc_FIX/SKP_Silk_common_pitch_est_defines.h \
src_SigProc_FIX/SKP_Silk_Inlines.h \
src_SigProc_FIX/SKP_Silk_MacroCount.h \
src_SigProc_FIX/SKP_Silk_MacroDebug.h \
src_SigProc_FIX/SKP_Silk_macros.h \
src_SigProc_FIX/SKP_Silk_pitch_est_defines.h \
src_SigProc_FIX/SKP_Silk_resampler_private.h \
src_SigProc_FIX/SKP_Silk_resampler_rom.h \
src_SigProc_FIX/SKP_Silk_resampler_structs.h \
src_SigProc_FIX/SKP_Silk_SigProc_FIX.h \
src_SigProc_FLP/SKP_Silk_SigProc_FLP.h

bin_PROGRAMS = Encoder Decoder signalCompare
Encoder_SOURCES = test/Encoder.c $(top_builddir)/interface/SKP_Silk_SDK_API.h
Encoder_LDADD = $(lib_LTLIBRARIES) 
Encoder_LDFLAGS = $(LIBS)

Decoder_SOURCES = test/Decoder.c $(top_builddir)/interface/SKP_Silk_SDK_API.h
Decoder_LDADD = $(lib_LTLIBRARIES) 
Decoder_LDFLAGS = $(LIBS)

signalCompare_SOURCES = test/signalCompare.c $(top_builddir)/interface/SKP_Silk_SDK_API.h
signalCompare_LDADD = $(lib_LTLIBRARIES) 
signalCompare_LDFLAGS = $(LIBS)

LIBS = ../celt/libcelt/libcelt0.la
