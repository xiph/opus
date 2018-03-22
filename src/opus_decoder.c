/* Copyright (c) 2010 Xiph.Org Foundation, Skype Limited
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
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#ifndef OPUS_BUILD
# error "OPUS_BUILD _MUST_ be defined to build Opus. This probably means you need other defines as well, as in a config.h. See the included build files for details."
#endif

#if defined(__GNUC__) && (__GNUC__ >= 2) && !defined(__OPTIMIZE__) && !defined(OPUS_WILL_BE_SLOW)
# pragma message "You appear to be compiling without optimization, if so opus will be very slow."
#endif

#include <stdarg.h>
#include "celt.h"
#include "opus.h"
#include "entdec.h"
#include "modes.h"
#include "API.h"
#include "stack_alloc.h"
#include "float_cast.h"
#include "opus_private.h"
#include "os_support.h"
#include "structs.h"
#include "define.h"
#include "mathops.h"
#include "cpu_support.h"

struct OpusDecoder {
   opus_uint32  crc;
   int          celt_dec_offset;
   int          silk_dec_offset;
   int          channels;
   opus_int32   Fs;          /** Sampling rate (at the API level) */
   silk_DecControlStruct DecControl;
   int          decode_gain;
   int          arch;

   /* Everything beyond this point gets cleared on a reset */
#define OPUS_DECODER_RESET_START stream_channels
   int          stream_channels;

   int          bandwidth;
   int          mode;
   int          prev_mode;
   int          frame_size;
   int          prev_redundancy;
   int          last_packet_duration;
#ifndef FIXED_POINT
   opus_val16   softclip_mem[2];
#endif

   opus_uint32  rangeFinal;
};

static const opus_uint32 crc_lookup[256]={
  0x00000000,0x04c11db7,0x09823b6e,0x0d4326d9,
  0x130476dc,0x17c56b6b,0x1a864db2,0x1e475005,
  0x2608edb8,0x22c9f00f,0x2f8ad6d6,0x2b4bcb61,
  0x350c9b64,0x31cd86d3,0x3c8ea00a,0x384fbdbd,
  0x4c11db70,0x48d0c6c7,0x4593e01e,0x4152fda9,
  0x5f15adac,0x5bd4b01b,0x569796c2,0x52568b75,
  0x6a1936c8,0x6ed82b7f,0x639b0da6,0x675a1011,
  0x791d4014,0x7ddc5da3,0x709f7b7a,0x745e66cd,
  0x9823b6e0,0x9ce2ab57,0x91a18d8e,0x95609039,
  0x8b27c03c,0x8fe6dd8b,0x82a5fb52,0x8664e6e5,
  0xbe2b5b58,0xbaea46ef,0xb7a96036,0xb3687d81,
  0xad2f2d84,0xa9ee3033,0xa4ad16ea,0xa06c0b5d,
  0xd4326d90,0xd0f37027,0xddb056fe,0xd9714b49,
  0xc7361b4c,0xc3f706fb,0xceb42022,0xca753d95,
  0xf23a8028,0xf6fb9d9f,0xfbb8bb46,0xff79a6f1,
  0xe13ef6f4,0xe5ffeb43,0xe8bccd9a,0xec7dd02d,
  0x34867077,0x30476dc0,0x3d044b19,0x39c556ae,
  0x278206ab,0x23431b1c,0x2e003dc5,0x2ac12072,
  0x128e9dcf,0x164f8078,0x1b0ca6a1,0x1fcdbb16,
  0x018aeb13,0x054bf6a4,0x0808d07d,0x0cc9cdca,
  0x7897ab07,0x7c56b6b0,0x71159069,0x75d48dde,
  0x6b93dddb,0x6f52c06c,0x6211e6b5,0x66d0fb02,
  0x5e9f46bf,0x5a5e5b08,0x571d7dd1,0x53dc6066,
  0x4d9b3063,0x495a2dd4,0x44190b0d,0x40d816ba,
  0xaca5c697,0xa864db20,0xa527fdf9,0xa1e6e04e,
  0xbfa1b04b,0xbb60adfc,0xb6238b25,0xb2e29692,
  0x8aad2b2f,0x8e6c3698,0x832f1041,0x87ee0df6,
  0x99a95df3,0x9d684044,0x902b669d,0x94ea7b2a,
  0xe0b41de7,0xe4750050,0xe9362689,0xedf73b3e,
  0xf3b06b3b,0xf771768c,0xfa325055,0xfef34de2,
  0xc6bcf05f,0xc27dede8,0xcf3ecb31,0xcbffd686,
  0xd5b88683,0xd1799b34,0xdc3abded,0xd8fba05a,
  0x690ce0ee,0x6dcdfd59,0x608edb80,0x644fc637,
  0x7a089632,0x7ec98b85,0x738aad5c,0x774bb0eb,
  0x4f040d56,0x4bc510e1,0x46863638,0x42472b8f,
  0x5c007b8a,0x58c1663d,0x558240e4,0x51435d53,
  0x251d3b9e,0x21dc2629,0x2c9f00f0,0x285e1d47,
  0x36194d42,0x32d850f5,0x3f9b762c,0x3b5a6b9b,
  0x0315d626,0x07d4cb91,0x0a97ed48,0x0e56f0ff,
  0x1011a0fa,0x14d0bd4d,0x19939b94,0x1d528623,
  0xf12f560e,0xf5ee4bb9,0xf8ad6d60,0xfc6c70d7,
  0xe22b20d2,0xe6ea3d65,0xeba91bbc,0xef68060b,
  0xd727bbb6,0xd3e6a601,0xdea580d8,0xda649d6f,
  0xc423cd6a,0xc0e2d0dd,0xcda1f604,0xc960ebb3,
  0xbd3e8d7e,0xb9ff90c9,0xb4bcb610,0xb07daba7,
  0xae3afba2,0xaafbe615,0xa7b8c0cc,0xa379dd7b,
  0x9b3660c6,0x9ff77d71,0x92b45ba8,0x9675461f,
  0x8832161a,0x8cf30bad,0x81b02d74,0x857130c3,
  0x5d8a9099,0x594b8d2e,0x5408abf7,0x50c9b640,
  0x4e8ee645,0x4a4ffbf2,0x470cdd2b,0x43cdc09c,
  0x7b827d21,0x7f436096,0x7200464f,0x76c15bf8,
  0x68860bfd,0x6c47164a,0x61043093,0x65c52d24,
  0x119b4be9,0x155a565e,0x18197087,0x1cd86d30,
  0x029f3d35,0x065e2082,0x0b1d065b,0x0fdc1bec,
  0x3793a651,0x3352bbe6,0x3e119d3f,0x3ad08088,
  0x2497d08d,0x2056cd3a,0x2d15ebe3,0x29d4f654,
  0xc5a92679,0xc1683bce,0xcc2b1d17,0xc8ea00a0,
  0xd6ad50a5,0xd26c4d12,0xdf2f6bcb,0xdbee767c,
  0xe3a1cbc1,0xe760d676,0xea23f0af,0xeee2ed18,
  0xf0a5bd1d,0xf464a0aa,0xf9278673,0xfde69bc4,
  0x89b8fd09,0x8d79e0be,0x803ac667,0x84fbdbd0,
  0x9abc8bd5,0x9e7d9662,0x933eb0bb,0x97ffad0c,
  0xafb010b1,0xab710d06,0xa6322bdf,0xa2f33668,
  0xbcb4666d,0xb8757bda,0xb5365d03,0xb1f740b4};

#define POLY 0x82f63b78
opus_uint32 crc32c(const unsigned char *buf, size_t len)
{
    int k;
    opus_uint32 crc = 0;
    crc = ~crc;
    while (len--) {
        crc ^= *buf++;
        for (k = 0; k < 8; k++)
            crc = crc & 1 ? (crc >> 1) ^ POLY : crc >> 1;
    }
    return ~crc;
}

static opus_uint32 opusint_compute_crc(unsigned char *buf, opus_int32 len){
  opus_uint32 crc_reg=0;
  int i;

  //return crc32c(buf, len);
  for(i=0;i<len;i++) crc_reg=(crc_reg<<8)^crc_lookup[((crc_reg >> 24)&0xff)^buf[i]];
  return crc_reg;
}

static void opusint_update_crc(OpusDecoder *st)
{
   int len;
   len = opus_decoder_get_size(st->channels);
   celt_assert(len!=0);
   st->crc = opusint_compute_crc(((unsigned char *)st)+sizeof(st->crc), len-sizeof(st->crc));
}

static void opusint_check_crc(OpusDecoder *st)
{
   int len;
   opus_uint32 crc;
   len = opus_decoder_get_size(st->channels);
   celt_assert(len!=0);
   crc = opusint_compute_crc(((unsigned char *)st)+sizeof(st->crc), len-sizeof(st->crc));
   celt_assert(crc == st->crc);
}

int opus_decoder_get_size(int channels)
{
   int silkDecSizeBytes, celtDecSizeBytes;
   int ret;
   if (channels<1 || channels > 2)
      return 0;
   ret = silk_Get_Decoder_Size( &silkDecSizeBytes );
   if(ret)
      return 0;
   silkDecSizeBytes = align(silkDecSizeBytes);
   celtDecSizeBytes = celt_decoder_get_size(channels);
   return align(sizeof(OpusDecoder))+silkDecSizeBytes+celtDecSizeBytes;
}

int opus_decoder_init(OpusDecoder *st, opus_int32 Fs, int channels)
{
   void *silk_dec;
   CELTDecoder *celt_dec;
   int ret, silkDecSizeBytes;

   if ((Fs!=48000&&Fs!=24000&&Fs!=16000&&Fs!=12000&&Fs!=8000)
    || (channels!=1&&channels!=2))
      return OPUS_BAD_ARG;

   OPUS_CLEAR((char*)st, opus_decoder_get_size(channels));
   /* Initialize SILK decoder */
   ret = silk_Get_Decoder_Size(&silkDecSizeBytes);
   if (ret)
      return OPUS_INTERNAL_ERROR;

   silkDecSizeBytes = align(silkDecSizeBytes);
   st->silk_dec_offset = align(sizeof(OpusDecoder));
   st->celt_dec_offset = st->silk_dec_offset+silkDecSizeBytes;
   silk_dec = (char*)st+st->silk_dec_offset;
   celt_dec = (CELTDecoder*)((char*)st+st->celt_dec_offset);
   st->stream_channels = st->channels = channels;

   st->Fs = Fs;
   st->DecControl.API_sampleRate = st->Fs;
   st->DecControl.nChannelsAPI      = st->channels;

   /* Reset decoder */
   ret = silk_InitDecoder( silk_dec );
   if(ret)return OPUS_INTERNAL_ERROR;

   /* Initialize CELT decoder */
   ret = celt_decoder_init(celt_dec, Fs, channels);
   if(ret!=OPUS_OK)return OPUS_INTERNAL_ERROR;

   celt_decoder_ctl(celt_dec, CELT_SET_SIGNALLING(0));

   st->prev_mode = 0;
   st->frame_size = Fs/400;
   st->arch = opus_select_arch();
   opusint_update_crc(st);
   return OPUS_OK;
}

OpusDecoder *opus_decoder_create(opus_int32 Fs, int channels, int *error)
{
   int ret;
   OpusDecoder *st;
   if ((Fs!=48000&&Fs!=24000&&Fs!=16000&&Fs!=12000&&Fs!=8000)
    || (channels!=1&&channels!=2))
   {
      if (error)
         *error = OPUS_BAD_ARG;
      return NULL;
   }
   st = (OpusDecoder *)opus_alloc(opus_decoder_get_size(channels));
   if (st == NULL)
   {
      if (error)
         *error = OPUS_ALLOC_FAIL;
      return NULL;
   }
   ret = opus_decoder_init(st, Fs, channels);
   if (error)
      *error = ret;
   if (ret != OPUS_OK)
   {
      opus_free(st);
      st = NULL;
   }
   return st;
}

static void smooth_fade(const opus_val16 *in1, const opus_val16 *in2,
      opus_val16 *out, int overlap, int channels,
      const opus_val16 *window, opus_int32 Fs)
{
   int i, c;
   int inc = 48000/Fs;
   for (c=0;c<channels;c++)
   {
      for (i=0;i<overlap;i++)
      {
         opus_val16 w = MULT16_16_Q15(window[i*inc], window[i*inc]);
         out[i*channels+c] = SHR32(MAC16_16(MULT16_16(w,in2[i*channels+c]),
                                   Q15ONE-w, in1[i*channels+c]), 15);
      }
   }
}

static int opus_packet_get_mode(const unsigned char *data)
{
   int mode;
   if (data[0]&0x80)
   {
      mode = MODE_CELT_ONLY;
   } else if ((data[0]&0x60) == 0x60)
   {
      mode = MODE_HYBRID;
   } else {
      mode = MODE_SILK_ONLY;
   }
   return mode;
}

static int opus_decode_frame(OpusDecoder *st, const unsigned char *data,
      opus_int32 len, opus_val16 *pcm, int frame_size, int decode_fec)
{
   void *silk_dec;
   CELTDecoder *celt_dec;
   int i, silk_ret=0, celt_ret=0;
   ec_dec dec;
   opus_int32 silk_frame_size;
   int pcm_silk_size;
   VARDECL(opus_int16, pcm_silk);
   int pcm_transition_silk_size;
   VARDECL(opus_val16, pcm_transition_silk);
   int pcm_transition_celt_size;
   VARDECL(opus_val16, pcm_transition_celt);
   opus_val16 *pcm_transition=NULL;
   int redundant_audio_size;
   VARDECL(opus_val16, redundant_audio);

   int audiosize;
   int mode;
   int transition=0;
   int start_band;
   int redundancy=0;
   int redundancy_bytes = 0;
   int celt_to_silk=0;
   int c;
   int F2_5, F5, F10, F20;
   const opus_val16 *window;
   opus_uint32 redundant_rng = 0;
   int celt_accum;
   ALLOC_STACK;

   silk_dec = (char*)st+st->silk_dec_offset;
   celt_dec = (CELTDecoder*)((char*)st+st->celt_dec_offset);
   F20 = st->Fs/50;
   F10 = F20>>1;
   F5 = F10>>1;
   F2_5 = F5>>1;
   if (frame_size < F2_5)
   {
      RESTORE_STACK;
      return OPUS_BUFFER_TOO_SMALL;
   }
   /* Limit frame_size to avoid excessive stack allocations. */
   frame_size = IMIN(frame_size, st->Fs/25*3);
   /* Payloads of 1 (2 including ToC) or 0 trigger the PLC/DTX */
   if (len<=1)
   {
      data = NULL;
      /* In that case, don't conceal more than what the ToC says */
      frame_size = IMIN(frame_size, st->frame_size);
   }
   if (data != NULL)
   {
      audiosize = st->frame_size;
      mode = st->mode;
      ec_dec_init(&dec,(unsigned char*)data,len);
   } else {
      audiosize = frame_size;
      mode = st->prev_mode;

      if (mode == 0)
      {
         /* If we haven't got any packet yet, all we can do is return zeros */
         for (i=0;i<audiosize*st->channels;i++)
            pcm[i] = 0;
         RESTORE_STACK;
         return audiosize;
      }

      /* Avoids trying to run the PLC on sizes other than 2.5 (CELT), 5 (CELT),
         10, or 20 (e.g. 12.5 or 30 ms). */
      if (audiosize > F20)
      {
         do {
            int ret = opus_decode_frame(st, NULL, 0, pcm, IMIN(audiosize, F20), 0);
            if (ret<0)
            {
               RESTORE_STACK;
               return ret;
            }
            pcm += ret*st->channels;
            audiosize -= ret;
         } while (audiosize > 0);
         RESTORE_STACK;
         return frame_size;
      } else if (audiosize < F20)
      {
         if (audiosize > F10)
            audiosize = F10;
         else if (mode != MODE_SILK_ONLY && audiosize > F5 && audiosize < F10)
            audiosize = F5;
      }
   }

   /* In fixed-point, we can tell CELT to do the accumulation on top of the
      SILK PCM buffer. This saves some stack space. */
#ifdef FIXED_POINT
   celt_accum = (mode != MODE_CELT_ONLY) && (frame_size >= F10);
#else
   celt_accum = 0;
#endif

   pcm_transition_silk_size = ALLOC_NONE;
   pcm_transition_celt_size = ALLOC_NONE;
   if (data!=NULL && st->prev_mode > 0 && (
       (mode == MODE_CELT_ONLY && st->prev_mode != MODE_CELT_ONLY && !st->prev_redundancy)
    || (mode != MODE_CELT_ONLY && st->prev_mode == MODE_CELT_ONLY) )
      )
   {
      transition = 1;
      /* Decide where to allocate the stack memory for pcm_transition */
      if (mode == MODE_CELT_ONLY)
         pcm_transition_celt_size = F5*st->channels;
      else
         pcm_transition_silk_size = F5*st->channels;
   }
   ALLOC(pcm_transition_celt, pcm_transition_celt_size, opus_val16);
   if (transition && mode == MODE_CELT_ONLY)
   {
      pcm_transition = pcm_transition_celt;
      opus_decode_frame(st, NULL, 0, pcm_transition, IMIN(F5, audiosize), 0);
   }
   if (audiosize > frame_size)
   {
      /*fprintf(stderr, "PCM buffer too small: %d vs %d (mode = %d)\n", audiosize, frame_size, mode);*/
      RESTORE_STACK;
      return OPUS_BAD_ARG;
   } else {
      frame_size = audiosize;
   }

   /* Don't allocate any memory when in CELT-only mode */
   pcm_silk_size = (mode != MODE_CELT_ONLY && !celt_accum) ? IMAX(F10, frame_size)*st->channels : ALLOC_NONE;
   ALLOC(pcm_silk, pcm_silk_size, opus_int16);

   /* SILK processing */
   if (mode != MODE_CELT_ONLY)
   {
      int lost_flag, decoded_samples;
      opus_int16 *pcm_ptr;
#ifdef FIXED_POINT
      if (celt_accum)
         pcm_ptr = pcm;
      else
#endif
         pcm_ptr = pcm_silk;

      if (st->prev_mode==MODE_CELT_ONLY)
         silk_InitDecoder( silk_dec );

      /* The SILK PLC cannot produce frames of less than 10 ms */
      st->DecControl.payloadSize_ms = IMAX(10, 1000 * audiosize / st->Fs);

      if (data != NULL)
      {
        st->DecControl.nChannelsInternal = st->stream_channels;
        if( mode == MODE_SILK_ONLY ) {
           if( st->bandwidth == OPUS_BANDWIDTH_NARROWBAND ) {
              st->DecControl.internalSampleRate = 8000;
           } else if( st->bandwidth == OPUS_BANDWIDTH_MEDIUMBAND ) {
              st->DecControl.internalSampleRate = 12000;
           } else if( st->bandwidth == OPUS_BANDWIDTH_WIDEBAND ) {
              st->DecControl.internalSampleRate = 16000;
           } else {
              st->DecControl.internalSampleRate = 16000;
              silk_assert( 0 );
           }
        } else {
           /* Hybrid mode */
           st->DecControl.internalSampleRate = 16000;
        }
     }

     lost_flag = data == NULL ? 1 : 2 * decode_fec;
     decoded_samples = 0;
     do {
        /* Call SILK decoder */
        int first_frame = decoded_samples == 0;
        silk_ret = silk_Decode( silk_dec, &st->DecControl,
                                lost_flag, first_frame, &dec, pcm_ptr, &silk_frame_size, st->arch );
        if( silk_ret ) {
           if (lost_flag) {
              /* PLC failure should not be fatal */
              silk_frame_size = frame_size;
              for (i=0;i<frame_size*st->channels;i++)
                 pcm_ptr[i] = 0;
           } else {
             RESTORE_STACK;
             return OPUS_INTERNAL_ERROR;
           }
        }
        pcm_ptr += silk_frame_size * st->channels;
        decoded_samples += silk_frame_size;
      } while( decoded_samples < frame_size );
   }

   start_band = 0;
   if (!decode_fec && mode != MODE_CELT_ONLY && data != NULL
    && ec_tell(&dec)+17+20*(st->mode == MODE_HYBRID) <= 8*len)
   {
      /* Check if we have a redundant 0-8 kHz band */
      if (mode == MODE_HYBRID)
         redundancy = ec_dec_bit_logp(&dec, 12);
      else
         redundancy = 1;
      if (redundancy)
      {
         celt_to_silk = ec_dec_bit_logp(&dec, 1);
         /* redundancy_bytes will be at least two, in the non-hybrid
            case due to the ec_tell() check above */
         redundancy_bytes = mode==MODE_HYBRID ?
               (opus_int32)ec_dec_uint(&dec, 256)+2 :
               len-((ec_tell(&dec)+7)>>3);
         len -= redundancy_bytes;
         /* This is a sanity check. It should never happen for a valid
            packet, so the exact behaviour is not normative. */
         if (len*8 < ec_tell(&dec))
         {
            len = 0;
            redundancy_bytes = 0;
            redundancy = 0;
         }
         /* Shrink decoder because of raw bits */
         dec.storage -= redundancy_bytes;
      }
   }
   if (mode != MODE_CELT_ONLY)
      start_band = 17;

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
      celt_decoder_ctl(celt_dec, CELT_SET_END_BAND(endband));
      celt_decoder_ctl(celt_dec, CELT_SET_CHANNELS(st->stream_channels));
   }

   if (redundancy)
   {
      transition = 0;
      pcm_transition_silk_size=ALLOC_NONE;
   }

   ALLOC(pcm_transition_silk, pcm_transition_silk_size, opus_val16);

   if (transition && mode != MODE_CELT_ONLY)
   {
      pcm_transition = pcm_transition_silk;
      opus_decode_frame(st, NULL, 0, pcm_transition, IMIN(F5, audiosize), 0);
   }

   /* Only allocation memory for redundancy if/when needed */
   redundant_audio_size = redundancy ? F5*st->channels : ALLOC_NONE;
   ALLOC(redundant_audio, redundant_audio_size, opus_val16);

   /* 5 ms redundant frame for CELT->SILK*/
   if (redundancy && celt_to_silk)
   {
      celt_decoder_ctl(celt_dec, CELT_SET_START_BAND(0));
      celt_decode_with_ec(celt_dec, data+len, redundancy_bytes,
                          redundant_audio, F5, NULL, 0);
      celt_decoder_ctl(celt_dec, OPUS_GET_FINAL_RANGE(&redundant_rng));
   }

   /* MUST be after PLC */
   celt_decoder_ctl(celt_dec, CELT_SET_START_BAND(start_band));

   if (mode != MODE_SILK_ONLY)
   {
      int celt_frame_size = IMIN(F20, frame_size);
      /* Make sure to discard any previous CELT state */
      if (mode != st->prev_mode && st->prev_mode > 0 && !st->prev_redundancy)
         celt_decoder_ctl(celt_dec, OPUS_RESET_STATE);
      /* Decode CELT */
      celt_ret = celt_decode_with_ec(celt_dec, decode_fec ? NULL : data,
                                     len, pcm, celt_frame_size, &dec, celt_accum);
   } else {
      unsigned char silence[2] = {0xFF, 0xFF};
      if (!celt_accum)
      {
         for (i=0;i<frame_size*st->channels;i++)
            pcm[i] = 0;
      }
      /* For hybrid -> SILK transitions, we let the CELT MDCT
         do a fade-out by decoding a silence frame */
      if (st->prev_mode == MODE_HYBRID && !(redundancy && celt_to_silk && st->prev_redundancy) )
      {
         celt_decoder_ctl(celt_dec, CELT_SET_START_BAND(0));
         celt_decode_with_ec(celt_dec, silence, 2, pcm, F2_5, NULL, celt_accum);
      }
   }

   if (mode != MODE_CELT_ONLY && !celt_accum)
   {
#ifdef FIXED_POINT
      for (i=0;i<frame_size*st->channels;i++)
         pcm[i] = SAT16(ADD32(pcm[i], pcm_silk[i]));
#else
      for (i=0;i<frame_size*st->channels;i++)
         pcm[i] = pcm[i] + (opus_val16)((1.f/32768.f)*pcm_silk[i]);
#endif
   }

   {
      const CELTMode *celt_mode;
      celt_decoder_ctl(celt_dec, CELT_GET_MODE(&celt_mode));
      window = celt_mode->window;
   }

   /* 5 ms redundant frame for SILK->CELT */
   if (redundancy && !celt_to_silk)
   {
      celt_decoder_ctl(celt_dec, OPUS_RESET_STATE);
      celt_decoder_ctl(celt_dec, CELT_SET_START_BAND(0));

      celt_decode_with_ec(celt_dec, data+len, redundancy_bytes, redundant_audio, F5, NULL, 0);
      celt_decoder_ctl(celt_dec, OPUS_GET_FINAL_RANGE(&redundant_rng));
      smooth_fade(pcm+st->channels*(frame_size-F2_5), redundant_audio+st->channels*F2_5,
                  pcm+st->channels*(frame_size-F2_5), F2_5, st->channels, window, st->Fs);
   }
   if (redundancy && celt_to_silk)
   {
      for (c=0;c<st->channels;c++)
      {
         for (i=0;i<F2_5;i++)
            pcm[st->channels*i+c] = redundant_audio[st->channels*i+c];
      }
      smooth_fade(redundant_audio+st->channels*F2_5, pcm+st->channels*F2_5,
                  pcm+st->channels*F2_5, F2_5, st->channels, window, st->Fs);
   }
   if (transition)
   {
      if (audiosize >= F5)
      {
         for (i=0;i<st->channels*F2_5;i++)
            pcm[i] = pcm_transition[i];
         smooth_fade(pcm_transition+st->channels*F2_5, pcm+st->channels*F2_5,
                     pcm+st->channels*F2_5, F2_5,
                     st->channels, window, st->Fs);
      } else {
         /* Not enough time to do a clean transition, but we do it anyway
            This will not preserve amplitude perfectly and may introduce
            a bit of temporal aliasing, but it shouldn't be too bad and
            that's pretty much the best we can do. In any case, generating this
            transition it pretty silly in the first place */
         smooth_fade(pcm_transition, pcm,
                     pcm, F2_5,
                     st->channels, window, st->Fs);
      }
   }

   if(st->decode_gain)
   {
      opus_val32 gain;
      gain = celt_exp2(MULT16_16_P15(QCONST16(6.48814081e-4f, 25), st->decode_gain));
      for (i=0;i<frame_size*st->channels;i++)
      {
         opus_val32 x;
         x = MULT16_32_P16(pcm[i],gain);
         pcm[i] = SATURATE(x, 32767);
      }
   }

   if (len <= 1)
      st->rangeFinal = 0;
   else
      st->rangeFinal = dec.rng ^ redundant_rng;

   st->prev_mode = mode;
   st->prev_redundancy = redundancy && !celt_to_silk;

   if (celt_ret>=0)
   {
      if (OPUS_CHECK_ARRAY(pcm, audiosize*st->channels))
         OPUS_PRINT_INT(audiosize);
   }

   RESTORE_STACK;
   return celt_ret < 0 ? celt_ret : audiosize;

}

int opus_decode_native(OpusDecoder *st, const unsigned char *data,
      opus_int32 len, opus_val16 *pcm, int frame_size, int decode_fec,
      int self_delimited, opus_int32 *packet_offset, int soft_clip)
{
   int i, nb_samples;
   int count, offset;
   unsigned char toc;
   int packet_frame_size, packet_bandwidth, packet_mode, packet_stream_channels;
   /* 48 x 2.5 ms = 120 ms */
   opus_int16 size[48];
   if (decode_fec<0 || decode_fec>1)
      return OPUS_BAD_ARG;
   /* For FEC/PLC, frame_size has to be to have a multiple of 2.5 ms */
   if ((decode_fec || len==0 || data==NULL) && frame_size%(st->Fs/400)!=0)
      return OPUS_BAD_ARG;
   if (len==0 || data==NULL)
   {
      int pcm_count=0;
      do {
         int ret;
         ret = opus_decode_frame(st, NULL, 0, pcm+pcm_count*st->channels, frame_size-pcm_count, 0);
         if (ret<0)
            return ret;
         pcm_count += ret;
      } while (pcm_count < frame_size);
      celt_assert(pcm_count == frame_size);
      if (OPUS_CHECK_ARRAY(pcm, pcm_count*st->channels))
         OPUS_PRINT_INT(pcm_count);
      st->last_packet_duration = pcm_count;
      return pcm_count;
   } else if (len<0)
      return OPUS_BAD_ARG;

   packet_mode = opus_packet_get_mode(data);
   packet_bandwidth = opus_packet_get_bandwidth(data);
   packet_frame_size = opus_packet_get_samples_per_frame(data, st->Fs);
   packet_stream_channels = opus_packet_get_nb_channels(data);

   count = opus_packet_parse_impl(data, len, self_delimited, &toc, NULL,
                                  size, &offset, packet_offset);
   if (count<0)
      return count;

   data += offset;

   if (decode_fec)
   {
      int duration_copy;
      int ret;
      /* If no FEC can be present, run the PLC (recursive call) */
      if (frame_size < packet_frame_size || packet_mode == MODE_CELT_ONLY || st->mode == MODE_CELT_ONLY)
         return opus_decode_native(st, NULL, 0, pcm, frame_size, 0, 0, NULL, soft_clip);
      /* Otherwise, run the PLC on everything except the size for which we might have FEC */
      duration_copy = st->last_packet_duration;
      if (frame_size-packet_frame_size!=0)
      {
         ret = opus_decode_native(st, NULL, 0, pcm, frame_size-packet_frame_size, 0, 0, NULL, soft_clip);
         if (ret<0)
         {
            st->last_packet_duration = duration_copy;
            return ret;
         }
         celt_assert(ret==frame_size-packet_frame_size);
      }
      /* Complete with FEC */
      st->mode = packet_mode;
      st->bandwidth = packet_bandwidth;
      st->frame_size = packet_frame_size;
      st->stream_channels = packet_stream_channels;
      ret = opus_decode_frame(st, data, size[0], pcm+st->channels*(frame_size-packet_frame_size),
            packet_frame_size, 1);
      if (ret<0)
         return ret;
      else {
         if (OPUS_CHECK_ARRAY(pcm, frame_size*st->channels))
            OPUS_PRINT_INT(frame_size);
         st->last_packet_duration = frame_size;
         return frame_size;
      }
   }

   if (count*packet_frame_size > frame_size)
      return OPUS_BUFFER_TOO_SMALL;

   /* Update the state as the last step to avoid updating it on an invalid packet */
   st->mode = packet_mode;
   st->bandwidth = packet_bandwidth;
   st->frame_size = packet_frame_size;
   st->stream_channels = packet_stream_channels;

   nb_samples=0;
   for (i=0;i<count;i++)
   {
      int ret;
      ret = opus_decode_frame(st, data, size[i], pcm+nb_samples*st->channels, frame_size-nb_samples, 0);
      if (ret<0)
         return ret;
      celt_assert(ret==packet_frame_size);
      data += size[i];
      nb_samples += ret;
   }
   st->last_packet_duration = nb_samples;
   if (OPUS_CHECK_ARRAY(pcm, nb_samples*st->channels))
      OPUS_PRINT_INT(nb_samples);
#ifndef FIXED_POINT
   if (soft_clip)
      opus_pcm_soft_clip(pcm, nb_samples, st->channels, st->softclip_mem);
   else
      st->softclip_mem[0]=st->softclip_mem[1]=0;
#endif
   return nb_samples;
}

#ifdef FIXED_POINT

int opus_decode(OpusDecoder *st, const unsigned char *data,
      opus_int32 len, opus_val16 *pcm, int frame_size, int decode_fec)
{
   if(frame_size<=0)
      return OPUS_BAD_ARG;
   return opus_decode_native(st, data, len, pcm, frame_size, decode_fec, 0, NULL, 0);
}

#ifndef DISABLE_FLOAT_API
int opus_decode_float(OpusDecoder *st, const unsigned char *data,
      opus_int32 len, float *pcm, int frame_size, int decode_fec)
{
   VARDECL(opus_int16, out);
   int ret, i;
   int nb_samples;
   ALLOC_STACK;

   if(frame_size<=0)
   {
      RESTORE_STACK;
      return OPUS_BAD_ARG;
   }
   if (data != NULL && len > 0 && !decode_fec)
   {
      nb_samples = opus_decoder_get_nb_samples(st, data, len);
      if (nb_samples>0)
         frame_size = IMIN(frame_size, nb_samples);
      else
         return OPUS_INVALID_PACKET;
   }
   ALLOC(out, frame_size*st->channels, opus_int16);

   ret = opus_decode_native(st, data, len, out, frame_size, decode_fec, 0, NULL, 0);
   if (ret > 0)
   {
      for (i=0;i<ret*st->channels;i++)
         pcm[i] = (1.f/32768.f)*(out[i]);
   }
   RESTORE_STACK;
   return ret;
}
#endif


#else
int opus_decode(OpusDecoder *st, const unsigned char *data,
      opus_int32 len, opus_int16 *pcm, int frame_size, int decode_fec)
{
   VARDECL(float, out);
   int ret, i;
   int nb_samples;
   ALLOC_STACK;

   if(frame_size<=0)
   {
      RESTORE_STACK;
      return OPUS_BAD_ARG;
   }

   if (data != NULL && len > 0 && !decode_fec)
   {
      nb_samples = opus_decoder_get_nb_samples(st, data, len);
      if (nb_samples>0)
         frame_size = IMIN(frame_size, nb_samples);
      else
         return OPUS_INVALID_PACKET;
   }
   opusint_check_crc(st);
   ALLOC(out, frame_size*st->channels, float);

   ret = opus_decode_native(st, data, len, out, frame_size, decode_fec, 0, NULL, 1);
   opusint_update_crc(st);
   if (ret > 0)
   {
      for (i=0;i<ret*st->channels;i++)
         pcm[i] = FLOAT2INT16(out[i]);
   }
   RESTORE_STACK;
   return ret;
}

int opus_decode_float(OpusDecoder *st, const unsigned char *data,
      opus_int32 len, opus_val16 *pcm, int frame_size, int decode_fec)
{
   int ret;
   if(frame_size<=0)
      return OPUS_BAD_ARG;
   opusint_check_crc(st);
   ret = opus_decode_native(st, data, len, pcm, frame_size, decode_fec, 0, NULL, 0);
   opusint_update_crc(st);
   return ret;
}

#endif

int opus_decoder_ctl(OpusDecoder *st, int request, ...)
{
   int ret = OPUS_OK;
   va_list ap;
   void *silk_dec;
   CELTDecoder *celt_dec;

   opusint_check_crc(st);
   silk_dec = (char*)st+st->silk_dec_offset;
   celt_dec = (CELTDecoder*)((char*)st+st->celt_dec_offset);


   va_start(ap, request);

   switch (request)
   {
   case OPUS_GET_BANDWIDTH_REQUEST:
   {
      opus_int32 *value = va_arg(ap, opus_int32*);
      if (!value)
      {
         goto bad_arg;
      }
      *value = st->bandwidth;
   }
   break;
   case OPUS_GET_FINAL_RANGE_REQUEST:
   {
      opus_uint32 *value = va_arg(ap, opus_uint32*);
      if (!value)
      {
         goto bad_arg;
      }
      *value = st->rangeFinal;
   }
   break;
   case OPUS_RESET_STATE:
   {
      OPUS_CLEAR((char*)&st->OPUS_DECODER_RESET_START,
            sizeof(OpusDecoder)-
            ((char*)&st->OPUS_DECODER_RESET_START - (char*)st));

      celt_decoder_ctl(celt_dec, OPUS_RESET_STATE);
      silk_InitDecoder( silk_dec );
      st->stream_channels = st->channels;
      st->frame_size = st->Fs/400;
   }
   break;
   case OPUS_GET_SAMPLE_RATE_REQUEST:
   {
      opus_int32 *value = va_arg(ap, opus_int32*);
      if (!value)
      {
         goto bad_arg;
      }
      *value = st->Fs;
   }
   break;
   case OPUS_GET_PITCH_REQUEST:
   {
      opus_int32 *value = va_arg(ap, opus_int32*);
      if (!value)
      {
         goto bad_arg;
      }
      if (st->prev_mode == MODE_CELT_ONLY)
         celt_decoder_ctl(celt_dec, OPUS_GET_PITCH(value));
      else
         *value = st->DecControl.prevPitchLag;
   }
   break;
   case OPUS_GET_GAIN_REQUEST:
   {
      opus_int32 *value = va_arg(ap, opus_int32*);
      if (!value)
      {
         goto bad_arg;
      }
      *value = st->decode_gain;
   }
   break;
   case OPUS_SET_GAIN_REQUEST:
   {
       opus_int32 value = va_arg(ap, opus_int32);
       if (value<-32768 || value>32767)
       {
          goto bad_arg;
       }
       st->decode_gain = value;
   }
   break;
   case OPUS_GET_LAST_PACKET_DURATION_REQUEST:
   {
      opus_int32 *value = va_arg(ap, opus_int32*);
      if (!value)
      {
         goto bad_arg;
      }
      *value = st->last_packet_duration;
   }
   break;
   case OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST:
   {
       opus_int32 value = va_arg(ap, opus_int32);
       if(value<0 || value>1)
       {
          goto bad_arg;
       }
       celt_decoder_ctl(celt_dec, OPUS_SET_PHASE_INVERSION_DISABLED(value));
   }
   break;
   case OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST:
   {
       opus_int32 *value = va_arg(ap, opus_int32*);
       if (!value)
       {
          goto bad_arg;
       }
       celt_decoder_ctl(celt_dec, OPUS_GET_PHASE_INVERSION_DISABLED(value));
   }
   break;
   default:
      /*fprintf(stderr, "unknown opus_decoder_ctl() request: %d", request);*/
      ret = OPUS_UNIMPLEMENTED;
      break;
   }

   va_end(ap);
   opusint_update_crc(st);
   return ret;
bad_arg:
   va_end(ap);
   return OPUS_BAD_ARG;
}

void opus_decoder_destroy(OpusDecoder *st)
{
   opus_free(st);
}


int opus_packet_get_bandwidth(const unsigned char *data)
{
   int bandwidth;
   if (data[0]&0x80)
   {
      bandwidth = OPUS_BANDWIDTH_MEDIUMBAND + ((data[0]>>5)&0x3);
      if (bandwidth == OPUS_BANDWIDTH_MEDIUMBAND)
         bandwidth = OPUS_BANDWIDTH_NARROWBAND;
   } else if ((data[0]&0x60) == 0x60)
   {
      bandwidth = (data[0]&0x10) ? OPUS_BANDWIDTH_FULLBAND :
                                   OPUS_BANDWIDTH_SUPERWIDEBAND;
   } else {
      bandwidth = OPUS_BANDWIDTH_NARROWBAND + ((data[0]>>5)&0x3);
   }
   return bandwidth;
}

int opus_packet_get_nb_channels(const unsigned char *data)
{
   return (data[0]&0x4) ? 2 : 1;
}

int opus_packet_get_nb_frames(const unsigned char packet[], opus_int32 len)
{
   int count;
   if (len<1)
      return OPUS_BAD_ARG;
   count = packet[0]&0x3;
   if (count==0)
      return 1;
   else if (count!=3)
      return 2;
   else if (len<2)
      return OPUS_INVALID_PACKET;
   else
      return packet[1]&0x3F;
}

int opus_packet_get_nb_samples(const unsigned char packet[], opus_int32 len,
      opus_int32 Fs)
{
   int samples;
   int count = opus_packet_get_nb_frames(packet, len);

   if (count<0)
      return count;

   samples = count*opus_packet_get_samples_per_frame(packet, Fs);
   /* Can't have more than 120 ms */
   if (samples*25 > Fs*3)
      return OPUS_INVALID_PACKET;
   else
      return samples;
}

int opus_decoder_get_nb_samples(const OpusDecoder *dec,
      const unsigned char packet[], opus_int32 len)
{
   return opus_packet_get_nb_samples(packet, len, dec->Fs);
}
