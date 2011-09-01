/* Copyright (c) 2010-2011 Xiph.Org Foundation, Skype Limited
   Written by Jean-Marc Valin and Koen Vos */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdarg.h>
#include "celt.h"
#include "entenc.h"
#include "modes.h"
#include "silk_API.h"
#include "stack_alloc.h"
#include "float_cast.h"
#include "opus.h"
#include "arch.h"
#include "opus_private.h"
#include "os_support.h"

#include "silk_tuning_parameters.h"
#ifdef FIXED_POINT
#include "fixed/silk_structs_FIX.h"
#else
#include "float/silk_structs_FLP.h"
#endif

#define MAX_ENCODER_BUFFER 480

struct OpusEncoder {
    int          celt_enc_offset;
    int          silk_enc_offset;
    silk_EncControlStruct silk_mode;
    int          application;
    int          channels;
    int          delay_compensation;
    int          force_mono;
    int          signal_type;
    int          user_bandwidth;
    int          voice_ratio;
    int          Fs;
    int          use_vbr;
    int          vbr_constraint;
    int          bitrate_bps;
    int          user_bitrate_bps;
    int          encoder_buffer;

#define OPUS_ENCODER_RESET_START stream_channels
    int          stream_channels;
    int          hybrid_stereo_width_Q14;
    opus_int32   variable_HP_smth2_Q15;
    opus_val32   hp_mem[4];
    int          mode;
    int          prev_mode;
    int          bandwidth;
    /* Sampling rate (at the API level) */
    int          first;
    opus_val16   delay_buffer[MAX_ENCODER_BUFFER*2];

    int          rangeFinal;
};

/* Transition tables for the voice and audio modes. First column is the
   middle (memoriless) threshold. The second column is the hysteresis
   (difference with the middle) */
static const int voice_bandwidth_thresholds[10] = {
        11000, 1000, /* NB<->MB */
        14000, 1000, /* MB<->WB */
        21000, 2000, /* WB<->SWB */
        29000, 2000, /* SWB<->FB */
};
static const int audio_bandwidth_thresholds[10] = {
        30000,    0, /* MB not allowed */
        20000, 2000, /* MB<->WB */
        26000, 2000, /* WB<->SWB */
        33000, 2000, /* SWB<->FB */
};


int opus_encoder_get_size(int channels)
{
    int silkEncSizeBytes, celtEncSizeBytes;
    int ret;
    ret = silk_Get_Encoder_Size( &silkEncSizeBytes );
    if (ret)
        return 0;
    silkEncSizeBytes = align(silkEncSizeBytes);
    celtEncSizeBytes = celt_encoder_get_size(channels);
    return align(sizeof(OpusEncoder))+silkEncSizeBytes+celtEncSizeBytes;
}

int opus_encoder_init(OpusEncoder* st, int Fs, int channels, int application)
{
    void *silk_enc;
    CELTEncoder *celt_enc;
    int err;
    int ret, silkEncSizeBytes;

    if (channels > 2 || channels < 1)
        return OPUS_BAD_ARG;
    if (application < OPUS_APPLICATION_VOIP || application > OPUS_APPLICATION_AUDIO)
        return OPUS_BAD_ARG;
    if (Fs != 8000 && Fs != 12000 && Fs != 16000 && Fs != 24000 && Fs != 48000)
        return OPUS_BAD_ARG;

    OPUS_CLEAR((char*)st, opus_encoder_get_size(channels));
    /* Create SILK encoder */
    ret = silk_Get_Encoder_Size( &silkEncSizeBytes );
    if (ret)
        return OPUS_BAD_ARG;
    silkEncSizeBytes = align(silkEncSizeBytes);
    st->silk_enc_offset = align(sizeof(OpusEncoder));
    st->celt_enc_offset = st->silk_enc_offset+silkEncSizeBytes;
    silk_enc = (char*)st+st->silk_enc_offset;
    celt_enc = (CELTEncoder*)((char*)st+st->celt_enc_offset);

    st->stream_channels = st->channels = channels;

    st->Fs = Fs;

    ret = silk_InitEncoder( silk_enc, &st->silk_mode );
    if (ret)
        goto failure;

    /* default SILK parameters */
    st->silk_mode.nChannelsAPI              = channels;
    st->silk_mode.nChannelsInternal         = channels;
    st->silk_mode.API_sampleRate            = st->Fs;
    st->silk_mode.maxInternalSampleRate     = 16000;
    st->silk_mode.minInternalSampleRate     = 8000;
    st->silk_mode.desiredInternalSampleRate = 16000;
    st->silk_mode.payloadSize_ms            = 20;
    st->silk_mode.bitRate                   = 25000;
    st->silk_mode.packetLossPercentage      = 0;
    st->silk_mode.complexity                = 10;
    st->silk_mode.useInBandFEC              = 0;
    st->silk_mode.useDTX                    = 0;
    st->silk_mode.useCBR                    = 0;
    st->silk_mode.HP_cutoff_Hz              = 0;

    /* Create CELT encoder */
    /* Initialize CELT encoder */
    err = celt_encoder_init(celt_enc, Fs, channels);
    if (err != OPUS_OK)
        goto failure;
    celt_encoder_ctl(celt_enc, CELT_SET_SIGNALLING(0));

    st->use_vbr = 0;
    st->user_bitrate_bps = OPUS_BITRATE_AUTO;
    st->bitrate_bps = 3000+Fs*channels;
    st->application = application;
    st->signal_type = OPUS_SIGNAL_AUTO;
    st->user_bandwidth = OPUS_BANDWIDTH_AUTO;
    st->voice_ratio = 90;
    st->encoder_buffer = st->Fs/100;

    st->delay_compensation = st->Fs/400;
    /* This part is meant to compensate for the resampler delay as a function
       of the API sampling rate */
    if (st->Fs == 48000)
        st->delay_compensation += 23;
    else if (st->Fs == 24000)
       st->delay_compensation += 15;
    else
       st->delay_compensation += 2;

    st->hybrid_stereo_width_Q14             = 1 << 14;
    st->variable_HP_smth2_Q15 = SKP_LSHIFT( silk_lin2log( VARIABLE_HP_MIN_CUTOFF_HZ ), 8 );
    st->first = 1;
    st->mode = MODE_HYBRID;
    st->bandwidth = OPUS_BANDWIDTH_FULLBAND;

    return OPUS_OK;

failure:
    opus_free(st);
    return OPUS_INTERNAL_ERROR;
}

static unsigned char gen_toc(int mode, int framerate, int bandwidth, int silk_bandwidth, int channels)
{
   int period;
   unsigned char toc;
   period = 0;
   while (framerate < 400)
   {
       framerate <<= 1;
       period++;
   }
   if (mode == MODE_SILK_ONLY)
   {
       toc = (silk_bandwidth-OPUS_BANDWIDTH_NARROWBAND)<<5;
       toc |= (period-2)<<3;
   } else if (mode == MODE_CELT_ONLY)
   {
       int tmp = bandwidth-OPUS_BANDWIDTH_MEDIUMBAND;
       if (tmp < 0)
           tmp = 0;
       toc = 0x80;
       toc |= tmp << 5;
       toc |= period<<3;
   } else /* Hybrid */
   {
       toc = 0x60;
       toc |= (bandwidth-OPUS_BANDWIDTH_SUPERWIDEBAND)<<4;
       toc |= (period-2)<<3;
   }
   toc |= (channels==2)<<2;
   return toc;
}

#ifndef FIXED_POINT
void silk_biquad_float(
    const opus_val16      *in,            /* I:    Input signal                   */
    const opus_int32      *B_Q28,         /* I:    MA coefficients [3]            */
    const opus_int32      *A_Q28,         /* I:    AR coefficients [2]            */
    opus_val32            *S,             /* I/O:  State vector [2]               */
    opus_val16            *out,           /* O:    Output signal                  */
    const opus_int32      len,            /* I:    Signal length (must be even)   */
    int stride
)
{
    /* DIRECT FORM II TRANSPOSED (uses 2 element state vector) */
    opus_int   k;
    opus_val32 vout;
    opus_val32 inval;
    opus_val32 A[2], B[3];

    A[0] = A_Q28[0] * (1./((opus_int32)1<<28));
    A[1] = A_Q28[1] * (1./((opus_int32)1<<28));
    B[0] = B_Q28[0] * (1./((opus_int32)1<<28));
    B[1] = B_Q28[1] * (1./((opus_int32)1<<28));
    B[2] = B_Q28[2] * (1./((opus_int32)1<<28));

    /* Negate A_Q28 values and split in two parts */

    for( k = 0; k < len; k++ ) {
        /* S[ 0 ], S[ 1 ]: Q12 */
        inval = in[ k*stride ];
        vout = S[ 0 ] + B[0]*inval;

        S[ 0 ] = S[1] - vout*A[0] + B[1]*inval;

        S[ 1 ] = - vout*A[1] + B[2]*inval;

        /* Scale back to Q0 and saturate */
        out[ k*stride ] = vout;
    }
}
#endif

static void hp_cutoff(const opus_val16 *in, opus_int32 cutoff_Hz, opus_val16 *out, opus_val32 *hp_mem, int len, int channels, opus_int32 Fs)
{
   opus_int32 B_Q28[ 3 ], A_Q28[ 2 ];
   opus_int32 Fc_Q19, r_Q28, r_Q22;

   SKP_assert( cutoff_Hz <= SKP_int32_MAX / SILK_FIX_CONST( 1.5 * 3.14159 / 1000, 19 ) );
   Fc_Q19 = SKP_DIV32_16( SKP_SMULBB( SILK_FIX_CONST( 1.5 * 3.14159 / 1000, 19 ), cutoff_Hz ), Fs/1000 );
   SKP_assert( Fc_Q19 > 0 && Fc_Q19 < 32768 );

   r_Q28 = SILK_FIX_CONST( 1.0, 28 ) - SKP_MUL( SILK_FIX_CONST( 0.92, 9 ), Fc_Q19 );

   /* b = r * [ 1; -2; 1 ]; */
   /* a = [ 1; -2 * r * ( 1 - 0.5 * Fc^2 ); r^2 ]; */
   B_Q28[ 0 ] = r_Q28;
   B_Q28[ 1 ] = SKP_LSHIFT( -r_Q28, 1 );
   B_Q28[ 2 ] = r_Q28;

   /* -r * ( 2 - Fc * Fc ); */
   r_Q22  = SKP_RSHIFT( r_Q28, 6 );
   A_Q28[ 0 ] = SKP_SMULWW( r_Q22, SKP_SMULWW( Fc_Q19, Fc_Q19 ) - SILK_FIX_CONST( 2.0,  22 ) );
   A_Q28[ 1 ] = SKP_SMULWW( r_Q22, r_Q22 );

#ifdef FIXED_POINT
   silk_biquad_alt( in, B_Q28, A_Q28, hp_mem, out, len, channels );
   if( channels == 2 ) {
       silk_biquad_alt( in+1, B_Q28, A_Q28, hp_mem+2, out+1, len, channels );
   }
#else
   silk_biquad_float( in, B_Q28, A_Q28, hp_mem, out, len, channels );
   if( channels == 2 ) {
       silk_biquad_float( in+1, B_Q28, A_Q28, hp_mem+2, out+1, len, channels );
   }
#endif
}

OpusEncoder *opus_encoder_create(int Fs, int channels, int mode, int *error)
{
   int ret;
   OpusEncoder *st = (OpusEncoder *)opus_alloc(opus_encoder_get_size(channels));
   if (st == NULL)
   {
      if (error)
         *error = OPUS_ALLOC_FAIL;
      return NULL;
   }
   ret = opus_encoder_init(st, Fs, channels, mode);
   if (error)
      *error = ret;
   if (ret != OPUS_OK)
   {
      opus_free(st);
      st = NULL;
   }
   return st;
}
#ifdef FIXED_POINT
int opus_encode(OpusEncoder *st, const opus_val16 *pcm, int frame_size,
                unsigned char *data, int max_data_bytes)
#else
int opus_encode_float(OpusEncoder *st, const opus_val16 *pcm, int frame_size,
                      unsigned char *data, int max_data_bytes)
#endif
{
    void *silk_enc;
    CELTEncoder *celt_enc;
    int i;
    int ret=0;
    int nBytes;
    ec_enc enc;
    int silk_internal_bandwidth=-1;
    int bytes_target;
    int prefill=0;
    int start_band = 0;
    int redundancy = 0;
    int redundancy_bytes = 0;
    int celt_to_silk = 0;
    VARDECL(opus_val16, pcm_buf);
    int nb_compr_bytes;
    int to_celt = 0;
    opus_int32 mono_rate;
    opus_uint32 redundant_rng = 0;
    int cutoff_Hz, hp_freq_smth1;
    ALLOC_STACK;

    st->rangeFinal = 0;
    if (400*frame_size != st->Fs && 200*frame_size != st->Fs && 100*frame_size != st->Fs &&
         50*frame_size != st->Fs &&  25*frame_size != st->Fs &&  50*frame_size != 3*st->Fs)
    {
       RESTORE_STACK;
       return OPUS_BAD_ARG;
    }
    silk_enc = (char*)st+st->silk_enc_offset;
    celt_enc = (CELTEncoder*)((char*)st+st->celt_enc_offset);

    if (st->user_bitrate_bps==OPUS_BITRATE_AUTO)
        st->bitrate_bps = 60*st->Fs/frame_size + st->Fs*st->channels;
    else
        st->bitrate_bps = st->user_bitrate_bps;

    /* Rate-dependent mono-stereo decision */
    if (st->force_mono)
    {
        st->stream_channels = 1;
    } else if (st->mode == MODE_CELT_ONLY && st->channels == 2)
    {
        opus_int32 decision_rate;
        decision_rate = st->bitrate_bps + st->voice_ratio*st->voice_ratio;
        /* Add some hysteresis */
        if (st->stream_channels == 2)
            decision_rate += 4000;
        else
            decision_rate -= 4000;
        if (decision_rate>48000)
            st->stream_channels = 2;
        else
            st->stream_channels = 1;
    } else {
            st->stream_channels = st->channels;
    }

#ifdef FUZZING
    if (st->channels == 2 && (rand()&0x1F)==0)
       st->stream_channels = 3-st->stream_channels;
#endif

    /* Equivalent bit-rate for mono */
    mono_rate = st->bitrate_bps;
    if (st->stream_channels==2)
        mono_rate = 2*mono_rate/3;
    /* Compensate for smaller frame sizes assuming an equivalent overhead
       of 60 bits/frame */
    mono_rate -= 60*(st->Fs/frame_size - 50);

#ifdef FUZZING
    if ((rand()&0xF)==0)
    {
       if ((rand()&0x1)==0)
          st->mode = MODE_CELT_ONLY;
       else
          st->mode = MODE_SILK_ONLY;
    } else {
       if (st->prev_mode==MODE_CELT_ONLY)
          st->mode = MODE_CELT_ONLY;
       else
          st->mode = MODE_SILK_ONLY;
    }
#else
    /* Mode selection depending on application and signal type */
    if (st->application==OPUS_APPLICATION_VOIP)
    {
        opus_int32 threshold;
        threshold = 20000;
        /* OPUS_APPLICATION_VOIP default to auto high-pass */
        st->silk_mode.HP_cutoff_Hz=0;
        /* Hysteresis */
        if (st->prev_mode == MODE_CELT_ONLY)
            threshold -= 4000;
        else if (st->prev_mode>0)
            threshold += 4000;

        /* OPUS_APPLICATION_VOIP defaults to MODE_SILK_ONLY */
        if (st->signal_type == OPUS_SIGNAL_MUSIC && mono_rate > threshold)
            st->mode = MODE_CELT_ONLY;
        else
            st->mode = MODE_SILK_ONLY;
    } else {/* OPUS_APPLICATION_AUDIO */
        opus_int32 threshold;
        /* SILK/CELT threshold is higher for voice than for music */
        threshold = 36000;
        /* OPUS_APPLICATION_AUDIO disables the high-pass */
        st->silk_mode.HP_cutoff_Hz=-1;
        if (st->signal_type == OPUS_SIGNAL_MUSIC)
            threshold -= 20000;
        else if (st->signal_type == OPUS_SIGNAL_VOICE)
            threshold += 8000;

        /* Hysteresis */
        if (st->prev_mode == MODE_CELT_ONLY)
            threshold -= 4000;
        else if (st->prev_mode>0)
            threshold += 4000;

        if (mono_rate>threshold)
            st->mode = MODE_CELT_ONLY;
        else
            st->mode = MODE_SILK_ONLY;
    }
#endif
    /* Override the chosen mode to make sure we meet the requested frame size */
    if (st->mode == MODE_CELT_ONLY && frame_size > st->Fs/50)
       st->mode = MODE_SILK_ONLY;
    if (st->mode != MODE_CELT_ONLY && frame_size < st->Fs/100)
       st->mode = MODE_CELT_ONLY;

    if (st->prev_mode > 0 &&
        ((st->mode != MODE_CELT_ONLY && st->prev_mode == MODE_CELT_ONLY) ||
    (st->mode == MODE_CELT_ONLY && st->prev_mode != MODE_CELT_ONLY)))
    {
        redundancy = 1;
        celt_to_silk = (st->mode != MODE_CELT_ONLY);
        if (!celt_to_silk)
        {
            /* Switch to SILK/hybrid if frame size is 10 ms or more*/
            if (frame_size >= st->Fs/100)
            {
                st->mode = st->prev_mode;
                to_celt = 1;
            } else {
                redundancy=0;
            }
        }
    }
    if (st->mode != MODE_CELT_ONLY && st->prev_mode == MODE_CELT_ONLY)
    {
        silk_EncControlStruct dummy;
        silk_InitEncoder( silk_enc, &dummy);
        prefill=1;
    }

    /* Automatic (rate-dependent) bandwidth selection */
    if (st->mode == MODE_CELT_ONLY || st->first || st->silk_mode.allowBandwidthSwitch)
    {
        const int *bandwidth_thresholds;
        int bandwidth = OPUS_BANDWIDTH_FULLBAND;

        bandwidth_thresholds = st->mode == MODE_CELT_ONLY ? audio_bandwidth_thresholds : voice_bandwidth_thresholds;
        do {
            int threshold, hysteresis;
            threshold = bandwidth_thresholds[2*(bandwidth-OPUS_BANDWIDTH_MEDIUMBAND)];
            hysteresis = bandwidth_thresholds[2*(bandwidth-OPUS_BANDWIDTH_MEDIUMBAND)+1];
            if (!st->first)
            {
                if (st->bandwidth >= bandwidth)
                    threshold -= hysteresis;
                else
                    threshold += hysteresis;
            }
            if (mono_rate >= threshold)
                break;
        } while (--bandwidth>OPUS_BANDWIDTH_NARROWBAND);
        st->bandwidth = bandwidth;
        /* Prevents any transition to SWB/FB until the SILK layer has fully
           switched to WB mode and turned the variable LP filter off */
        if (st->mode != MODE_CELT_ONLY && !st->silk_mode.inWBmodeWithoutVariableLP && st->bandwidth > OPUS_BANDWIDTH_WIDEBAND)
            st->bandwidth = OPUS_BANDWIDTH_WIDEBAND;
    }

    /* Prevents Opus from wasting bits on frequencies that are above
       the Nyquist rate of the input signal */
    if (st->Fs <= 24000 && st->bandwidth > OPUS_BANDWIDTH_SUPERWIDEBAND)
        st->bandwidth = OPUS_BANDWIDTH_SUPERWIDEBAND;
    if (st->Fs <= 16000 && st->bandwidth > OPUS_BANDWIDTH_WIDEBAND)
        st->bandwidth = OPUS_BANDWIDTH_WIDEBAND;
    if (st->Fs <= 12000 && st->bandwidth > OPUS_BANDWIDTH_MEDIUMBAND)
        st->bandwidth = OPUS_BANDWIDTH_MEDIUMBAND;
    if (st->Fs <= 8000 && st->bandwidth > OPUS_BANDWIDTH_NARROWBAND)
        st->bandwidth = OPUS_BANDWIDTH_NARROWBAND;

    if (st->user_bandwidth != OPUS_BANDWIDTH_AUTO)
        st->bandwidth = st->user_bandwidth;

    /* Can't support higher than wideband for >20 ms frames */
    if (frame_size > st->Fs/50 && st->bandwidth > OPUS_BANDWIDTH_WIDEBAND)
       st->bandwidth = OPUS_BANDWIDTH_WIDEBAND;

    /* CELT mode doesn't support mediumband, use wideband instead */
    if (st->mode == MODE_CELT_ONLY && st->bandwidth == OPUS_BANDWIDTH_MEDIUMBAND)
        st->bandwidth = OPUS_BANDWIDTH_WIDEBAND;

    /* Chooses the appropriate mode for speech
       *NEVER* switch to/from CELT-only mode here as this will */
    if (st->mode == MODE_SILK_ONLY && st->bandwidth > OPUS_BANDWIDTH_WIDEBAND)
        st->mode = MODE_HYBRID;
    if (st->mode == MODE_HYBRID && st->bandwidth <= OPUS_BANDWIDTH_WIDEBAND)
        st->mode = MODE_SILK_ONLY;

    bytes_target = st->bitrate_bps * frame_size / (st->Fs * 8) - 1;

    data += 1;

    ec_enc_init(&enc, data, max_data_bytes-1);

    ALLOC(pcm_buf, (st->delay_compensation+frame_size)*st->channels, opus_val16);
    for (i=0;i<st->delay_compensation*st->channels;i++)
       pcm_buf[i] = st->delay_buffer[(st->encoder_buffer-st->delay_compensation)*st->channels+i];

    if (st->mode == MODE_CELT_ONLY)
       hp_freq_smth1 = SKP_LSHIFT( silk_lin2log( VARIABLE_HP_MIN_CUTOFF_HZ ), 8 );
    else
       hp_freq_smth1 = ((silk_encoder*)silk_enc)->state_Fxx[0].sCmn.variable_HP_smth1_Q15;

    st->variable_HP_smth2_Q15 = SKP_SMLAWB( st->variable_HP_smth2_Q15,
          hp_freq_smth1 - st->variable_HP_smth2_Q15, SILK_FIX_CONST( VARIABLE_HP_SMTH_COEF2, 16 ) );

    /* convert from log scale to Hertz */
    cutoff_Hz = silk_log2lin( SKP_RSHIFT( st->variable_HP_smth2_Q15, 8 ) );

    if (st->application == OPUS_APPLICATION_VOIP)
    {
       hp_cutoff(pcm, cutoff_Hz, &pcm_buf[st->delay_compensation*st->channels], st->hp_mem, frame_size, st->channels, st->Fs);
    } else {
       for (i=0;i<frame_size*st->channels;i++)
          pcm_buf[st->delay_compensation*st->channels + i] = pcm[i];
    }

    /* SILK processing */
    if (st->mode != MODE_CELT_ONLY)
    {
#ifdef FIXED_POINT
       const opus_int16 *pcm_silk;
#else
       VARDECL(opus_int16, pcm_silk);
       ALLOC(pcm_silk, st->channels*frame_size, opus_int16);
#endif
        st->silk_mode.bitRate = st->bitrate_bps - 8*st->Fs/frame_size;
        if( st->mode == MODE_HYBRID ) {
            st->silk_mode.bitRate /= st->stream_channels;
            if( st->bandwidth == OPUS_BANDWIDTH_SUPERWIDEBAND ) {
                if( st->Fs == 100 * frame_size ) {
                    /* 24 kHz, 10 ms */
                    st->silk_mode.bitRate = ( ( st->silk_mode.bitRate + 2000 + st->use_vbr * 1000 ) * 2 ) / 3;
                } else {
                    /* 24 kHz, 20 ms */
                    st->silk_mode.bitRate = ( ( st->silk_mode.bitRate + 1000 + st->use_vbr * 1000 ) * 2 ) / 3;
                }
            } else {
                if( st->Fs == 100 * frame_size ) {
                    /* 48 kHz, 10 ms */
                    st->silk_mode.bitRate = ( st->silk_mode.bitRate + 8000 + st->use_vbr * 3000 ) / 2;
                } else {
                    /* 48 kHz, 20 ms */
                    st->silk_mode.bitRate = ( st->silk_mode.bitRate + 9000 + st->use_vbr * 1000 ) / 2;
                }
            }
            st->silk_mode.bitRate *= st->stream_channels;
            /* don't let SILK use more than 80% */
            if( st->silk_mode.bitRate > ( st->bitrate_bps - 8*st->Fs/frame_size ) * 4/5 ) {
                st->silk_mode.bitRate = ( st->bitrate_bps - 8*st->Fs/frame_size ) * 4/5;
            }
        }

        st->silk_mode.payloadSize_ms = 1000 * frame_size / st->Fs;
        st->silk_mode.nChannelsAPI = st->channels;
        st->silk_mode.nChannelsInternal = st->stream_channels;
        if (st->bandwidth == OPUS_BANDWIDTH_NARROWBAND) {
            st->silk_mode.desiredInternalSampleRate = 8000;
        } else if (st->bandwidth == OPUS_BANDWIDTH_MEDIUMBAND) {
            st->silk_mode.desiredInternalSampleRate = 12000;
        } else {
            SKP_assert( st->mode == MODE_HYBRID || st->bandwidth == OPUS_BANDWIDTH_WIDEBAND );
            st->silk_mode.desiredInternalSampleRate = 16000;
        }
        if( st->mode == MODE_HYBRID ) {
            /* Don't allow bandwidth reduction at lowest bitrates in hybrid mode */
            st->silk_mode.minInternalSampleRate = 16000;
        } else {
            st->silk_mode.minInternalSampleRate = 8000;
        }
        st->silk_mode.maxInternalSampleRate = 16000;

        /* Call SILK encoder for the low band */
        nBytes = IMIN(1275, max_data_bytes-1);
        if (prefill)
        {
            int zero=0;
#ifdef FIXED_POINT
            pcm_silk = st->delay_buffer;
#else
            for (i=0;i<st->encoder_buffer*st->channels;i++)
                pcm_silk[i] = FLOAT2INT16(st->delay_buffer[i]);
#endif
            silk_Encode( silk_enc, &st->silk_mode, pcm_silk, st->encoder_buffer, NULL, &zero, 1 );
        }

#ifdef FIXED_POINT
        pcm_silk = pcm_buf+st->delay_compensation*st->channels;
#else
        for (i=0;i<frame_size*st->channels;i++)
            pcm_silk[i] = FLOAT2INT16(pcm_buf[st->delay_compensation*st->channels + i]);
#endif
        ret = silk_Encode( silk_enc, &st->silk_mode, pcm_silk, frame_size, &enc, &nBytes, 0 );
        if( ret ) {
            /*fprintf (stderr, "SILK encode error: %d\n", ret);*/
            /* Handle error */
           return OPUS_INTERNAL_ERROR;
        }
        if (nBytes==0)
        {
           data[-1] = gen_toc(st->mode, st->Fs/frame_size, st->bandwidth, silk_internal_bandwidth, st->stream_channels);
           RESTORE_STACK;
           return 1;
        }
        /* Extract SILK internal bandwidth for signaling in first byte */
        if( st->mode == MODE_SILK_ONLY ) {
            if( st->silk_mode.internalSampleRate == 8000 ) {
                silk_internal_bandwidth = OPUS_BANDWIDTH_NARROWBAND;
            } else if( st->silk_mode.internalSampleRate == 12000 ) {
                silk_internal_bandwidth = OPUS_BANDWIDTH_MEDIUMBAND;
            } else if( st->silk_mode.internalSampleRate == 16000 ) {
                silk_internal_bandwidth = OPUS_BANDWIDTH_WIDEBAND;
            }
        } else {
            SKP_assert( st->silk_mode.internalSampleRate == 16000 );
        }
    }

    /* CELT processing */
    {
        int endband=21;

        switch(st->bandwidth)
        {
            case OPUS_BANDWIDTH_NARROWBAND:
                endband = 13;
                break;
            case OPUS_BANDWIDTH_MEDIUMBAND:
            case OPUS_BANDWIDTH_WIDEBAND:
                endband = 17;
                break;
            case OPUS_BANDWIDTH_SUPERWIDEBAND:
                endband = 19;
                break;
            case OPUS_BANDWIDTH_FULLBAND:
                endband = 21;
                break;
        }
        celt_encoder_ctl(celt_enc, CELT_SET_END_BAND(endband));
        celt_encoder_ctl(celt_enc, CELT_SET_CHANNELS(st->stream_channels));
    }
    if (st->mode != MODE_SILK_ONLY)
    {
        celt_encoder_ctl(celt_enc, OPUS_SET_VBR(0));
        celt_encoder_ctl(celt_enc, OPUS_SET_BITRATE(-1));
        if (st->prev_mode == MODE_SILK_ONLY)
        {
            unsigned char dummy[10];
            celt_encoder_ctl(celt_enc, OPUS_RESET_STATE);
            celt_encoder_ctl(celt_enc, CELT_SET_START_BAND(0));
            celt_encoder_ctl(celt_enc, CELT_SET_PREDICTION(0));
            /* TODO: This wastes CPU a bit compared to just prefilling the buffer */
            celt_encode_with_ec(celt_enc, &st->delay_buffer[(st->encoder_buffer-st->delay_compensation-st->Fs/400)*st->channels], st->Fs/400, dummy, 10, NULL);
        } else {
            celt_encoder_ctl(celt_enc, CELT_SET_PREDICTION(2));
        }

        if (st->mode == MODE_HYBRID)
        {
            int len;

            len = (ec_tell(&enc)+7)>>3;
            if( st->use_vbr ) {
                nb_compr_bytes = len + bytes_target - (st->silk_mode.bitRate * frame_size) / (8 * st->Fs);
            } else {
                /* check if SILK used up too much */
                nb_compr_bytes = len > bytes_target ? len : bytes_target;
            }
        } else {
            if (st->use_vbr)
            {
                celt_encoder_ctl(celt_enc, OPUS_SET_VBR(1));
                celt_encoder_ctl(celt_enc, OPUS_SET_VBR_CONSTRAINT(st->vbr_constraint));
                celt_encoder_ctl(celt_enc, OPUS_SET_BITRATE(st->bitrate_bps));
                nb_compr_bytes = max_data_bytes-1;
            } else {
                nb_compr_bytes = bytes_target;
            }
        }

    } else {
        nb_compr_bytes = 0;
    }

    for (i=0;i<st->encoder_buffer*st->channels;i++)
        st->delay_buffer[i] = pcm_buf[(frame_size+st->delay_compensation-st->encoder_buffer)*st->channels+i];


    if( st->mode == MODE_HYBRID && st->stream_channels == 2 ) {
        /* Apply stereo width reduction (at low bitrates) */
        if( st->hybrid_stereo_width_Q14 < (1 << 14) || st->silk_mode.stereoWidth_Q14 < (1 << 14) ) {
            int width_Q14, delta_Q14, nSamples_8ms, diff;
            nSamples_8ms = ( st->Fs * 8 ) / 1000;
            width_Q14 = (1 << 14 ) - st->hybrid_stereo_width_Q14;
            delta_Q14 = ( st->hybrid_stereo_width_Q14 - st->silk_mode.stereoWidth_Q14 ) / nSamples_8ms;
            for( i = 0; i < nSamples_8ms; i++ ) {
                width_Q14 += delta_Q14;
                diff = pcm_buf[ 2*i+1 ] - (opus_int32)pcm_buf[ 2*i ];
                diff = ( diff * width_Q14 ) >> 15;
                pcm_buf[ 2*i ]   = (opus_int16)( pcm_buf[ 2*i ]   + diff );
                pcm_buf[ 2*i+1 ] = (opus_int16)( pcm_buf[ 2*i+1 ] - diff );
            }
            for( ; i < frame_size; i++ ) {
                diff = pcm_buf[ 2*i+1 ] - (opus_int32)pcm_buf[ 2*i ];
                diff = ( diff * width_Q14 ) >> 15;
                pcm_buf[ 2*i ]   = (opus_int16)( pcm_buf[ 2*i ]   + diff );
                pcm_buf[ 2*i+1 ] = (opus_int16)( pcm_buf[ 2*i+1 ] - diff );
            }
            st->hybrid_stereo_width_Q14 = st->silk_mode.stereoWidth_Q14;
        }
    }

    if (st->mode != MODE_CELT_ONLY)
    {
        /* Check if we have a redundant 0-8 kHz band */
        ec_enc_bit_logp(&enc, redundancy, 12);
        if (redundancy)
        {
            /* Target the same bit-rate for redundancy as for the rest,
               up to a max of 257 bytes */
            redundancy_bytes = IMIN(257, st->bitrate_bps/1600);
            ec_enc_bit_logp(&enc, celt_to_silk, 1);
            if (st->mode == MODE_HYBRID)
                ec_enc_uint(&enc, redundancy_bytes-2, 256);
        }
        start_band = 17;
    }

    if (st->mode == MODE_SILK_ONLY)
    {
        ret = (ec_tell(&enc)+7)>>3;
        ec_enc_done(&enc);
        /*When in LPC only mode it's perfectly
          reasonable to strip off trailing zero bytes as
          the required range decoder behavior is to
          fill these in. This can't be done when the MDCT
          modes are used because the decoder needs to know
          the actual length for allocation purposes.*/
        if(!redundancy)
            while(ret>2&&data[ret-1]==0)ret--;
        nb_compr_bytes = ret;
    } else {
       nb_compr_bytes = IMIN(1275-redundancy_bytes, nb_compr_bytes);
       ec_enc_shrink(&enc, nb_compr_bytes);
    }


    /* 5 ms redundant frame for CELT->SILK */
    if (redundancy && celt_to_silk)
    {
        celt_encoder_ctl(celt_enc, CELT_SET_START_BAND(0));
        celt_encoder_ctl(celt_enc, OPUS_SET_VBR(0));
        celt_encode_with_ec(celt_enc, pcm_buf, st->Fs/200, data+nb_compr_bytes, redundancy_bytes, NULL);
        celt_encoder_ctl(celt_enc, OPUS_GET_FINAL_RANGE(&redundant_rng));
        celt_encoder_ctl(celt_enc, OPUS_RESET_STATE);
    }

    celt_encoder_ctl(celt_enc, CELT_SET_START_BAND(start_band));

    if (st->mode != MODE_SILK_ONLY)
    {
        ret = celt_encode_with_ec(celt_enc, pcm_buf, frame_size, NULL, nb_compr_bytes, &enc);
    }

    /* 5 ms redundant frame for SILK->CELT */
    if (redundancy && !celt_to_silk)
    {
        int N2, N4;
        N2 = st->Fs/200;
        N4 = st->Fs/400;

        celt_encoder_ctl(celt_enc, OPUS_RESET_STATE);
        celt_encoder_ctl(celt_enc, CELT_SET_START_BAND(0));
        celt_encoder_ctl(celt_enc, CELT_SET_PREDICTION(0));

        /* TODO: We could speed up prefilling here */
        celt_encode_with_ec(celt_enc, pcm_buf+st->channels*(frame_size-N2-N4), N4, data+nb_compr_bytes, redundancy_bytes, NULL);

        celt_encode_with_ec(celt_enc, pcm_buf+st->channels*(frame_size-N2), N2, data+nb_compr_bytes, redundancy_bytes, NULL);
        celt_encoder_ctl(celt_enc, OPUS_GET_FINAL_RANGE(&redundant_rng));
    }



    /* Signalling the mode in the first byte */
    data--;
    data[0] = gen_toc(st->mode, st->Fs/frame_size, st->bandwidth, silk_internal_bandwidth, st->stream_channels);

    st->rangeFinal = enc.rng ^ redundant_rng;

    if (to_celt)
        st->prev_mode = MODE_CELT_ONLY;
    else
        st->prev_mode = st->mode;
    st->first = 0;
    RESTORE_STACK;
    return ret+1+redundancy_bytes;
}

#ifdef FIXED_POINT

#ifndef DISABLE_FLOAT_API
int opus_encode_float(OpusEncoder *st, const float *pcm, int frame_size,
      unsigned char *data, int max_data_bytes)
{
   int i, ret;
   VARDECL(opus_int16, in);
   ALLOC_STACK;

   ALLOC(in, frame_size*st->channels, opus_int16);

   for (i=0;i<frame_size*st->channels;i++)
      in[i] = FLOAT2INT16(pcm[i]);
   ret = opus_encode(st, in, frame_size, data, max_data_bytes);
   RESTORE_STACK;
   return ret;
}
#endif

#else
int opus_encode(OpusEncoder *st, const opus_int16 *pcm, int frame_size,
      unsigned char *data, int max_data_bytes)
{
   int i, ret;
   VARDECL(float, in);
   ALLOC_STACK;

   ALLOC(in, frame_size*st->channels, float);

   for (i=0;i<frame_size*st->channels;i++)
      in[i] = (1./32768)*pcm[i];
   ret = opus_encode_float(st, in, frame_size, data, max_data_bytes);
   RESTORE_STACK;
   return ret;
}
#endif


int opus_encoder_ctl(OpusEncoder *st, int request, ...)
{
    int ret;
    CELTEncoder *celt_enc;
    va_list ap;

    ret = OPUS_OK;
    va_start(ap, request);

    celt_enc = (CELTEncoder*)((char*)st+st->celt_enc_offset);

    switch (request)
    {
        case OPUS_SET_APPLICATION_REQUEST:
        {
            opus_int32 value = va_arg(ap, opus_int32);
            st->application = value;
        }
        break;
        case OPUS_GET_APPLICATION_REQUEST:
        {
            opus_int32 *value = va_arg(ap, opus_int32*);
            *value = st->mode;
        }
        break;
        case OPUS_SET_BITRATE_REQUEST:
        {
            opus_int32 value = va_arg(ap, opus_int32);
            if (value != OPUS_BITRATE_AUTO)
            {
                if (value <= 0)
                    goto bad_arg;
                else if (value <= 500)
                    value = 500;
            }
            st->user_bitrate_bps = value;
        }
        break;
        case OPUS_GET_BITRATE_REQUEST:
        {
            opus_int32 *value = va_arg(ap, opus_int32*);
            *value = st->bitrate_bps;
        }
        break;
        case OPUS_SET_FORCE_MONO_REQUEST:
        {
            opus_int32 value = va_arg(ap, opus_int32);
            st->force_mono = value;
        }
        break;
        case OPUS_GET_FORCE_MONO_REQUEST:
        {
            opus_int32 *value = va_arg(ap, opus_int32*);
            *value = !!st->force_mono;
        }
        break;
        case OPUS_SET_BANDWIDTH_REQUEST:
        {
            opus_int32 value = va_arg(ap, opus_int32);
            if (value < OPUS_BANDWIDTH_AUTO || value > OPUS_BANDWIDTH_FULLBAND)
                return OPUS_BAD_ARG;
            st->user_bandwidth = value;
            if (st->user_bandwidth == OPUS_BANDWIDTH_NARROWBAND) {
                st->silk_mode.maxInternalSampleRate = 8000;
            } else if (st->bandwidth == OPUS_BANDWIDTH_MEDIUMBAND) {
                st->silk_mode.maxInternalSampleRate = 12000;
            } else {
                st->silk_mode.maxInternalSampleRate = 16000;
            }
        }
        break;
        case OPUS_GET_BANDWIDTH_REQUEST:
        {
            opus_int32 *value = va_arg(ap, opus_int32*);
            *value = st->bandwidth;
        }
        break;
        case OPUS_SET_DTX_REQUEST:
        {
            opus_int32 value = va_arg(ap, opus_int32);
            st->silk_mode.useDTX = value;
        }
        break;
        case OPUS_GET_DTX_REQUEST:
        {
            opus_int32 *value = va_arg(ap, opus_int32*);
            *value = st->silk_mode.useDTX;
        }
        break;
        case OPUS_SET_COMPLEXITY_REQUEST:
        {
            opus_int32 value = va_arg(ap, opus_int32);
            st->silk_mode.complexity = value;
            celt_encoder_ctl(celt_enc, OPUS_SET_COMPLEXITY(value));
        }
        break;
        case OPUS_GET_COMPLEXITY_REQUEST:
        {
            opus_int32 *value = va_arg(ap, opus_int32*);
            *value = st->silk_mode.complexity;
        }
        break;
        case OPUS_SET_INBAND_FEC_REQUEST:
        {
            opus_int32 value = va_arg(ap, opus_int32);
            st->silk_mode.useInBandFEC = value;
        }
        break;
        case OPUS_GET_INBAND_FEC_REQUEST:
        {
            opus_int32 *value = va_arg(ap, opus_int32*);
            *value = st->silk_mode.useInBandFEC;
        }
        break;
        case OPUS_SET_PACKET_LOSS_PERC_REQUEST:
        {
            opus_int32 value = va_arg(ap, opus_int32);
            if (value < 0 || value > 100)
                return OPUS_BAD_ARG;
            st->silk_mode.packetLossPercentage = value;
            celt_encoder_ctl(celt_enc, OPUS_SET_PACKET_LOSS_PERC(value));
        }
        break;
        case OPUS_GET_PACKET_LOSS_PERC_REQUEST:
        {
            opus_int32 *value = va_arg(ap, opus_int32*);
            *value = st->silk_mode.packetLossPercentage;
        }
        break;
        case OPUS_SET_VBR_REQUEST:
        {
            opus_int32 value = va_arg(ap, opus_int32);
            st->use_vbr = value;
            st->silk_mode.useCBR = 1-value;
        }
        break;
        case OPUS_GET_VBR_REQUEST:
        {
            opus_int32 *value = va_arg(ap, opus_int32*);
            *value = st->use_vbr;
        }
        break;
        case OPUS_SET_VOICE_RATIO_REQUEST:
        {
            opus_int32 value = va_arg(ap, opus_int32);
            if (value>100 || value<0)
                goto bad_arg;
            st->voice_ratio = value;
        }
        break;
        case OPUS_GET_VOICE_RATIO_REQUEST:
        {
            opus_int32 *value = va_arg(ap, opus_int32*);
            *value = st->voice_ratio;
        }
        break;
        case OPUS_SET_VBR_CONSTRAINT_REQUEST:
        {
            opus_int32 value = va_arg(ap, opus_int32);
            st->vbr_constraint = value;
        }
        break;
        case OPUS_GET_VBR_CONSTRAINT_REQUEST:
        {
            opus_int32 *value = va_arg(ap, opus_int32*);
            *value = st->vbr_constraint;
        }
        break;
        case OPUS_SET_SIGNAL_REQUEST:
        {
            opus_int32 value = va_arg(ap, opus_int32);
            st->signal_type = value;
        }
        break;
        case OPUS_GET_SIGNAL_REQUEST:
        {
            opus_int32 *value = va_arg(ap, opus_int32*);
            *value = st->signal_type;
        }
        break;
        case OPUS_GET_LOOKAHEAD_REQUEST:
        {
            opus_int32 *value = va_arg(ap, opus_int32*);
            *value = st->delay_compensation+st->Fs/400;
        }
        break;
        case OPUS_GET_FINAL_RANGE_REQUEST:
        {
            opus_uint32 *value = va_arg(ap, opus_uint32*);
            *value = st->rangeFinal;
        }
        break;
        case OPUS_RESET_STATE:
        {
           void *silk_enc;
           silk_EncControlStruct dummy;
           silk_enc = (char*)st+st->silk_enc_offset;

           OPUS_CLEAR((char*)&st->OPUS_ENCODER_RESET_START,
                 opus_encoder_get_size(st->channels)-
                 ((char*)&st->OPUS_ENCODER_RESET_START - (char*)st));

           celt_encoder_ctl(celt_enc, OPUS_RESET_STATE);
           silk_InitEncoder( silk_enc, &dummy );
           st->stream_channels = st->channels;
           st->hybrid_stereo_width_Q14             = 1 << 14;
           st->first = 1;
           st->mode = MODE_HYBRID;
           st->bandwidth = OPUS_BANDWIDTH_FULLBAND;
           st->variable_HP_smth2_Q15 = SKP_LSHIFT( silk_lin2log( VARIABLE_HP_MIN_CUTOFF_HZ ), 8 );
        }
        break;
        default:
            /* fprintf(stderr, "unknown opus_encoder_ctl() request: %d", request);*/
            ret = OPUS_UNIMPLEMENTED;
            break;
    }
    va_end(ap);
    return ret;
bad_arg:
    va_end(ap);
    return OPUS_BAD_ARG;
}

void opus_encoder_destroy(OpusEncoder *st)
{
    opus_free(st);
}
