const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const fixed_point = b.option(bool, "fixed-point", "Enable fixed-point build") orelse false;

    const module = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    module.addCMacro("HAVE_CONFIG_H", "1");

    const config_h = b.addConfigHeader(.{
        .style = .blank,
        .include_path = "config.h",
    }, .{
        .PACKAGE_NAME = "opus",
        .PACKAGE_VERSION = "1.4",
        .PACKAGE_STRING = "opus 1.4",
        .PACKAGE_BUGREPORT = "opus@xiph.org",
        .PACKAGE_TARNAME = "opus",
        .OPUS_CUSTOM_BUILD = .defined,
        .OPUS_BUILD = .defined,
        .USE_ALLOCA = .defined,
        .NONTHREADSAFE_PSEUDOSTACK = .undef,
        .CUSTOM_MODES = .defined,
        .HAVE_STDINT_H = .defined,
    });

    if (fixed_point) {
        config_h.addValues(.{ .FIXED_POINT = .defined });
    } else {
        config_h.addValues(.{ .FLOAT_POINT = .defined });
    }

    module.addConfigHeader(config_h);
    module.addIncludePath(b.path("include"));
    module.addIncludePath(b.path("celt"));
    module.addIncludePath(b.path("silk"));
    module.addIncludePath(b.path("src"));
    module.addIncludePath(b.path("."));
    if (fixed_point) {
        module.addIncludePath(b.path("silk/fixed"));
    } else {
        module.addIncludePath(b.path("silk/float"));
    }

    module.addCSourceFiles(.{ .files = &opus_sources });
    module.addCSourceFiles(.{ .files = &celt_sources });
    module.addCSourceFiles(.{ .files = &silk_common_sources });
    if (fixed_point) {
        module.addCSourceFiles(.{ .files = &silk_fixed_sources });
    } else {
        module.addCSourceFiles(.{ .files = &silk_float_sources });
    }

    const cpu = target.result.cpu;
    switch (cpu.arch) {
        .x86, .x86_64 => {
            module.addIncludePath(b.path("celt/x86"));
            module.addIncludePath(b.path("silk/x86"));
            if (fixed_point) {
                module.addIncludePath(b.path("silk/fixed/x86"));
            } else {
                module.addIncludePath(b.path("silk/float/x86"));
            }

            module.addCSourceFile(.{ .file = b.path("celt/x86/x86cpu.c") });
            module.addCSourceFile(.{ .file = b.path("celt/x86/x86_celt_map.c") });
            module.addCSourceFile(.{ .file = b.path("silk/x86/x86_silk_map.c") });

            config_h.addValues(.{
                .OPUS_X86_MAY_HAVE_SSE = .defined,
                .OPUS_X86_MAY_HAVE_SSE2 = .defined,
                .OPUS_X86_MAY_HAVE_SSE4_1 = .defined,
                .OPUS_X86_MAY_HAVE_AVX = .defined,
                .OPUS_X86_MAY_HAVE_AVX2 = .defined,
            });

            if (std.Target.x86.featureSetHas(cpu.features, .sse)) {
                module.addCSourceFile(.{ .file = b.path("celt/x86/pitch_sse.c") });
            }
            if (std.Target.x86.featureSetHas(cpu.features, .sse2)) {
                module.addCSourceFile(.{ .file = b.path("celt/x86/pitch_sse2.c") });
                module.addCSourceFile(.{ .file = b.path("celt/x86/vq_sse2.c") });
            }
            if (std.Target.x86.featureSetHas(cpu.features, .sse4_1)) {
                module.addCSourceFile(.{ .file = b.path("celt/x86/celt_lpc_sse4_1.c") });
                module.addCSourceFile(.{ .file = b.path("celt/x86/pitch_sse4_1.c") });
                module.addCSourceFile(.{ .file = b.path("silk/x86/NSQ_sse4_1.c") });
                module.addCSourceFile(.{ .file = b.path("silk/x86/NSQ_del_dec_sse4_1.c") });
                module.addCSourceFile(.{ .file = b.path("silk/x86/VAD_sse4_1.c") });
                module.addCSourceFile(.{ .file = b.path("silk/x86/VQ_WMat_EC_sse4_1.c") });
                if (fixed_point) {
                    module.addCSourceFile(.{ .file = b.path("silk/fixed/x86/burg_modified_FIX_sse4_1.c") });
                    module.addCSourceFile(.{ .file = b.path("silk/fixed/x86/vector_ops_FIX_sse4_1.c") });
                }
            }
            if (std.Target.x86.featureSetHas(cpu.features, .avx)) {
                module.addCSourceFile(.{ .file = b.path("celt/x86/pitch_avx.c") });
            }
            if (std.Target.x86.featureSetHas(cpu.features, .avx2)) {
                module.addCSourceFile(.{ .file = b.path("silk/x86/NSQ_del_dec_avx2.c") });
                if (!fixed_point) {
                    module.addCSourceFile(.{ .file = b.path("silk/float/x86/inner_product_FLP_avx2.c") });
                }
            }
        },
        .arm, .armeb, .thumb, .thumbeb, .aarch64, .aarch64_be => {
            const has_neon = switch (cpu.arch) {
                .arm, .armeb, .thumb, .thumbeb => std.Target.arm.featureSetHas(cpu.features, .neon),
                .aarch64, .aarch64_be => std.Target.aarch64.featureSetHas(cpu.features, .neon),
                else => unreachable,
            };

            if (has_neon) {
                module.addIncludePath(b.path("celt/arm"));
                module.addIncludePath(b.path("silk/arm"));
                if (fixed_point) {
                    module.addIncludePath(b.path("silk/fixed/arm"));
                }

                module.addCSourceFile(.{ .file = b.path("celt/arm/armcpu.c") });
                // Dispatchers.
                module.addCSourceFile(.{ .file = b.path("celt/arm/arm_celt_map.c") });
                module.addCSourceFile(.{ .file = b.path("silk/arm/arm_silk_map.c") });

                // NEON files.
                module.addCSourceFile(.{ .file = b.path("celt/arm/celt_fft_ne10.c") });
                module.addCSourceFile(.{ .file = b.path("celt/arm/celt_mdct_ne10.c") });
                module.addCSourceFile(.{ .file = b.path("celt/arm/celt_neon_intr.c") });
                module.addCSourceFile(.{ .file = b.path("celt/arm/pitch_neon_intr.c") });
                module.addCSourceFile(.{ .file = b.path("silk/arm/biquad_alt_neon_intr.c") });
                module.addCSourceFile(.{ .file = b.path("silk/arm/LPC_inv_pred_gain_neon_intr.c") });
                module.addCSourceFile(.{ .file = b.path("silk/arm/NSQ_del_dec_neon_intr.c") });
                module.addCSourceFile(.{ .file = b.path("silk/arm/NSQ_neon.c") });

                if (fixed_point) {
                    module.addCSourceFile(.{ .file = b.path("silk/fixed/arm/warped_autocorrelation_FIX_neon_intr.c") });
                }

                config_h.addValues(.{
                    .OPUS_ARM_MAY_HAVE_NEON = .defined,
                    .HAVE_ARM_NEON_INTR = .defined,
                });
            }
        },
        else => {},
    }

    const lib = b.addLibrary(.{
        .name = "opus",
        .root_module = module,
    });

    b.installArtifact(lib);
    const header_install_step = b.addInstallDirectory(.{
        .source_dir = b.path("include"),
        .install_dir = .header,
        .install_subdir = "opus",
    });
    b.getInstallStep().dependOn(&header_install_step.step);
}

const opus_sources = [_][]const u8{
    "src/analysis.c",
    "src/extensions.c",
    "src/mapping_matrix.c",
    "src/mlp.c",
    "src/mlp_data.c",
    "src/opus.c",
    "src/opus_decoder.c",
    "src/opus_encoder.c",
    "src/opus_multistream.c",
    "src/opus_multistream_decoder.c",
    "src/opus_multistream_encoder.c",
    "src/opus_projection_decoder.c",
    "src/opus_projection_encoder.c",
    "src/repacketizer.c",
};

const celt_sources = [_][]const u8{
    "celt/bands.c",
    "celt/celt.c",
    "celt/celt_decoder.c",
    "celt/celt_encoder.c",
    "celt/celt_lpc.c",
    "celt/cwrs.c",
    "celt/entcode.c",
    "celt/entdec.c",
    "celt/entenc.c",
    "celt/kiss_fft.c",
    "celt/laplace.c",
    "celt/mathops.c",
    "celt/mdct.c",
    "celt/modes.c",
    "celt/pitch.c",
    "celt/quant_bands.c",
    "celt/rate.c",
    "celt/vq.c",
};

const silk_common_sources = [_][]const u8{
    "silk/A2NLSF.c",
    "silk/ana_filt_bank_1.c",
    "silk/biquad_alt.c",
    "silk/bwexpander.c",
    "silk/bwexpander_32.c",
    "silk/check_control_input.c",
    "silk/CNG.c",
    "silk/code_signs.c",
    "silk/control_audio_bandwidth.c",
    "silk/control_codec.c",
    "silk/control_SNR.c",
    "silk/debug.c",
    "silk/dec_API.c",
    "silk/decode_core.c",
    "silk/decode_frame.c",
    "silk/decode_indices.c",
    "silk/decode_parameters.c",
    "silk/decode_pitch.c",
    "silk/decode_pulses.c",
    "silk/decoder_set_fs.c",
    "silk/enc_API.c",
    "silk/encode_indices.c",
    "silk/encode_pulses.c",
    "silk/gain_quant.c",
    "silk/HP_variable_cutoff.c",
    "silk/init_decoder.c",
    "silk/init_encoder.c",
    "silk/inner_prod_aligned.c",
    "silk/interpolate.c",
    "silk/lin2log.c",
    "silk/LPC_fit.c",
    "silk/log2lin.c",
    "silk/LPC_analysis_filter.c",
    "silk/LPC_inv_pred_gain.c",
    "silk/LP_variable_cutoff.c",
    "silk/NLSF2A.c",
    "silk/NLSF_decode.c",
    "silk/NLSF_del_dec_quant.c",
    "silk/NLSF_encode.c",
    "silk/NLSF_stabilize.c",
    "silk/NLSF_unpack.c",
    "silk/NLSF_VQ.c",
    "silk/NLSF_VQ_weights_laroia.c",
    "silk/NSQ.c",
    "silk/NSQ_del_dec.c",
    "silk/pitch_est_tables.c",
    "silk/PLC.c",
    "silk/process_NLSFs.c",
    "silk/quant_LTP_gains.c",
    "silk/resampler.c",
    "silk/resampler_down2.c",
    "silk/resampler_down2_3.c",
    "silk/resampler_private_AR2.c",
    "silk/resampler_private_down_FIR.c",
    "silk/resampler_private_IIR_FIR.c",
    "silk/resampler_private_up2_HQ.c",
    "silk/resampler_rom.c",
    "silk/shell_coder.c",
    "silk/sigm_Q15.c",
    "silk/sort.c",
    "silk/stereo_decode_pred.c",
    "silk/stereo_encode_pred.c",
    "silk/stereo_find_predictor.c",
    "silk/stereo_LR_to_MS.c",
    "silk/stereo_MS_to_LR.c",
    "silk/stereo_quant_pred.c",
    "silk/sum_sqr_shift.c",
    "silk/tables_gain.c",
    "silk/tables_LTP.c",
    "silk/tables_NLSF_CB_NB_MB.c",
    "silk/tables_NLSF_CB_WB.c",
    "silk/tables_other.c",
    "silk/tables_pitch_lag.c",
    "silk/tables_pulses_per_block.c",
    "silk/table_LSF_cos.c",
    "silk/VAD.c",
    "silk/VQ_WMat_EC.c",
};

const silk_float_sources = [_][]const u8{
    "silk/float/apply_sine_window_FLP.c",
    "silk/float/autocorrelation_FLP.c",
    "silk/float/burg_modified_FLP.c",
    "silk/float/bwexpander_FLP.c",
    "silk/float/corrMatrix_FLP.c",
    "silk/float/encode_frame_FLP.c",
    "silk/float/energy_FLP.c",
    "silk/float/find_LPC_FLP.c",
    "silk/float/find_LTP_FLP.c",
    "silk/float/find_pitch_lags_FLP.c",
    "silk/float/find_pred_coefs_FLP.c",
    "silk/float/inner_product_FLP.c",
    "silk/float/k2a_FLP.c",
    "silk/float/LPC_analysis_filter_FLP.c",
    "silk/float/LPC_inv_pred_gain_FLP.c",
    "silk/float/LTP_analysis_filter_FLP.c",
    "silk/float/LTP_scale_ctrl_FLP.c",
    "silk/float/noise_shape_analysis_FLP.c",
    "silk/float/pitch_analysis_core_FLP.c",
    "silk/float/process_gains_FLP.c",
    "silk/float/regularize_correlations_FLP.c",
    "silk/float/residual_energy_FLP.c",
    "silk/float/scale_copy_vector_FLP.c",
    "silk/float/scale_vector_FLP.c",
    "silk/float/schur_FLP.c",
    "silk/float/sort_FLP.c",
    "silk/float/warped_autocorrelation_FLP.c",
    "silk/float/wrappers_FLP.c",
};

const silk_fixed_sources = [_][]const u8{
    "silk/fixed/apply_sine_window_FIX.c",
    "silk/fixed/autocorr_FIX.c",
    "silk/fixed/burg_modified_FIX.c",
    "silk/fixed/corrMatrix_FIX.c",
    "silk/fixed/encode_frame_FIX.c",
    "silk/fixed/find_LPC_FIX.c",
    "silk/fixed/find_LTP_FIX.c",
    "silk/fixed/find_pitch_lags_FIX.c",
    "silk/fixed/find_pred_coefs_FIX.c",
    "silk/fixed/k2a_FIX.c",
    "silk/fixed/k2a_Q16_FIX.c",
    "silk/fixed/LTP_analysis_filter_FIX.c",
    "silk/fixed/LTP_scale_ctrl_FIX.c",
    "silk/fixed/noise_shape_analysis_FIX.c",
    "silk/fixed/pitch_analysis_core_FIX.c",
    "silk/fixed/process_gains_FIX.c",
    "silk/fixed/regularize_correlations_FIX.c",
    "silk/fixed/residual_energy16_FIX.c",
    "silk/fixed/residual_energy_FIX.c",
    "silk/fixed/schur64_FIX.c",
    "silk/fixed/schur_FIX.c",
    "silk/fixed/vector_ops_FIX.c",
    "silk/fixed/warped_autocorrelation_FIX.c",
};
