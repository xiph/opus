/* Copyright (c) 2011 Xiph.Org Foundation
   Written by Jean-Marc Valin */
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
#include "config.h"
#endif

#include "opus_multistream.h"
#include "opus.h"
#include "opus_private.h"
#include "stack_alloc.h"
#include <stdarg.h>
#include "float_cast.h"
#include "os_support.h"
#include "analysis.h"

struct OpusMSEncoder {
   TonalityAnalysisState analysis;
   ChannelLayout layout;
   int variable_duration;
   opus_int32 bitrate_bps;
   opus_val32 subframe_mem[3];
   /* Encoder states go here */
};


static int validate_encoder_layout(const ChannelLayout *layout)
{
   int s;
   for (s=0;s<layout->nb_streams;s++)
   {
      if (s < layout->nb_coupled_streams)
      {
         if (get_left_channel(layout, s, -1)==-1)
            return 0;
         if (get_right_channel(layout, s, -1)==-1)
            return 0;
      } else {
         if (get_mono_channel(layout, s, -1)==-1)
            return 0;
      }
   }
   return 1;
}


opus_int32 opus_multistream_encoder_get_size(int nb_streams, int nb_coupled_streams)
{
   int coupled_size;
   int mono_size;

   if(nb_streams<1||nb_coupled_streams>nb_streams||nb_coupled_streams<0)return 0;
   coupled_size = opus_encoder_get_size(2);
   mono_size = opus_encoder_get_size(1);
   return align(sizeof(OpusMSEncoder))
        + nb_coupled_streams * align(coupled_size)
        + (nb_streams-nb_coupled_streams) * align(mono_size);
}



int opus_multistream_encoder_init(
      OpusMSEncoder *st,
      opus_int32 Fs,
      int channels,
      int streams,
      int coupled_streams,
      const unsigned char *mapping,
      int application
)
{
   int coupled_size;
   int mono_size;
   int i, ret;
   char *ptr;

   if ((channels>255) || (channels<1) || (coupled_streams>streams) ||
       (coupled_streams+streams>255) || (streams<1) || (coupled_streams<0))
      return OPUS_BAD_ARG;

   st->layout.nb_channels = channels;
   st->layout.nb_streams = streams;
   st->layout.nb_coupled_streams = coupled_streams;
   st->subframe_mem[0]=st->subframe_mem[1]=st->subframe_mem[2]=0;
   OPUS_CLEAR(&st->analysis,1);

   st->bitrate_bps = OPUS_AUTO;
   st->variable_duration = OPUS_FRAMESIZE_ARG;
   for (i=0;i<st->layout.nb_channels;i++)
      st->layout.mapping[i] = mapping[i];
   if (!validate_layout(&st->layout) || !validate_encoder_layout(&st->layout))
      return OPUS_BAD_ARG;
   ptr = (char*)st + align(sizeof(OpusMSEncoder));
   coupled_size = opus_encoder_get_size(2);
   mono_size = opus_encoder_get_size(1);

   for (i=0;i<st->layout.nb_coupled_streams;i++)
   {
      ret = opus_encoder_init((OpusEncoder*)ptr, Fs, 2, application);
      if(ret!=OPUS_OK)return ret;
      ptr += align(coupled_size);
   }
   for (;i<st->layout.nb_streams;i++)
   {
      ret = opus_encoder_init((OpusEncoder*)ptr, Fs, 1, application);
      if(ret!=OPUS_OK)return ret;
      ptr += align(mono_size);
   }
   return OPUS_OK;
}

OpusMSEncoder *opus_multistream_encoder_create(
      opus_int32 Fs,
      int channels,
      int streams,
      int coupled_streams,
      const unsigned char *mapping,
      int application,
      int *error
)
{
   int ret;
   OpusMSEncoder *st;
   if ((channels>255) || (channels<1) || (coupled_streams>streams) ||
       (coupled_streams+streams>255) || (streams<1) || (coupled_streams<0))
   {
      if (error)
         *error = OPUS_BAD_ARG;
      return NULL;
   }
   st = (OpusMSEncoder *)opus_alloc(opus_multistream_encoder_get_size(streams, coupled_streams));
   if (st==NULL)
   {
      if (error)
         *error = OPUS_ALLOC_FAIL;
      return NULL;
   }
   ret = opus_multistream_encoder_init(st, Fs, channels, streams, coupled_streams, mapping, application);
   if (ret != OPUS_OK)
   {
      opus_free(st);
      st = NULL;
   }
   if (error)
      *error = ret;
   return st;
}

typedef void (*opus_copy_channel_in_func)(
  opus_val16 *dst,
  int dst_stride,
  const void *src,
  int src_stride,
  int src_channel,
  int frame_size
);

/* Max size in case the encoder decides to return three frames */
#define MS_FRAME_TMP (3*1275+7)
static int opus_multistream_encode_native
(
    OpusMSEncoder *st,
    opus_copy_channel_in_func copy_channel_in,
    const void *pcm,
    int frame_size,
    unsigned char *data,
    opus_int32 max_data_bytes,
    int lsb_depth
#ifndef FIXED_POINT
    , downmix_func downmix
    , const void *pcm_analysis
#endif
)
{
   opus_int32 Fs;
   int coupled_size;
   int mono_size;
   int s;
   char *ptr;
   int tot_size;
   VARDECL(opus_val16, buf);
   unsigned char tmp_data[MS_FRAME_TMP];
   OpusRepacketizer rp;
   int orig_frame_size;
   int coded_channels;
   opus_int32 channel_rate;
   opus_int32 complexity;
   AnalysisInfo analysis_info;
   const CELTMode *celt_mode;
   ALLOC_STACK;

   ptr = (char*)st + align(sizeof(OpusMSEncoder));
   opus_encoder_ctl((OpusEncoder*)ptr, OPUS_GET_SAMPLE_RATE(&Fs));
   opus_encoder_ctl((OpusEncoder*)ptr, OPUS_GET_COMPLEXITY(&complexity));
   opus_encoder_ctl((OpusEncoder*)ptr, CELT_GET_MODE(&celt_mode));

   if (400*frame_size < Fs)
   {
      RESTORE_STACK;
      return OPUS_BAD_ARG;
   }
   orig_frame_size = IMIN(frame_size,Fs/50);
#ifndef FIXED_POINT
   analysis_info.valid = 0;
   if (complexity >= 7 && Fs==48000)
   {
      opus_int32 delay_compensation;
      int channels;

      channels = st->layout.nb_streams + st->layout.nb_coupled_streams;
      opus_encoder_ctl((OpusEncoder*)ptr, OPUS_GET_LOOKAHEAD(&delay_compensation));
      delay_compensation -= Fs/400;

      frame_size = run_analysis(&st->analysis, celt_mode, pcm, pcm_analysis,
            frame_size, st->variable_duration, channels, Fs, st->bitrate_bps, delay_compensation, lsb_depth, downmix, &analysis_info);
   } else
#endif
   {
      frame_size = frame_size_select(frame_size, st->variable_duration, Fs);
   }
   /* Validate frame_size before using it to allocate stack space.
      This mirrors the checks in opus_encode[_float](). */
   if (400*frame_size != Fs && 200*frame_size != Fs &&
       100*frame_size != Fs &&  50*frame_size != Fs &&
        25*frame_size != Fs &&  50*frame_size != 3*Fs)
   {
      RESTORE_STACK;
      return OPUS_BAD_ARG;
   }
   ALLOC(buf, 2*frame_size, opus_val16);
   coupled_size = opus_encoder_get_size(2);
   mono_size = opus_encoder_get_size(1);

   if (max_data_bytes < 4*st->layout.nb_streams-1)
   {
      RESTORE_STACK;
      return OPUS_BUFFER_TOO_SMALL;
   }

   /* Compute bitrate allocation between streams (this could be a lot better) */
   coded_channels = st->layout.nb_streams + st->layout.nb_coupled_streams;
   if (st->bitrate_bps==OPUS_AUTO)
   {
      channel_rate = Fs+60*Fs/orig_frame_size;
   } else if (st->bitrate_bps==OPUS_BITRATE_MAX)
   {
      channel_rate = 300000;
   } else {
      channel_rate = st->bitrate_bps/coded_channels;
   }
#ifndef FIXED_POINT
   if (st->variable_duration==OPUS_FRAMESIZE_VARIABLE && frame_size != Fs/50)
   {
      opus_int32 bonus;
      bonus = 60*(Fs/frame_size-50);
      channel_rate += bonus;
   }
#endif
   ptr = (char*)st + align(sizeof(OpusMSEncoder));
   for (s=0;s<st->layout.nb_streams;s++)
   {
      OpusEncoder *enc;
      enc = (OpusEncoder*)ptr;
      if (s < st->layout.nb_coupled_streams)
         ptr += align(coupled_size);
      else
         ptr += align(mono_size);
      opus_encoder_ctl(enc, OPUS_SET_BITRATE(channel_rate * (s < st->layout.nb_coupled_streams ? 2 : 1)));
   }

   ptr = (char*)st + align(sizeof(OpusMSEncoder));
   /* Counting ToC */
   tot_size = 0;
   for (s=0;s<st->layout.nb_streams;s++)
   {
      OpusEncoder *enc;
      int len;
      int curr_max;

      opus_repacketizer_init(&rp);
      enc = (OpusEncoder*)ptr;
      if (s < st->layout.nb_coupled_streams)
      {
         int left, right;
         left = get_left_channel(&st->layout, s, -1);
         right = get_right_channel(&st->layout, s, -1);
         (*copy_channel_in)(buf, 2,
            pcm, st->layout.nb_channels, left, frame_size);
         (*copy_channel_in)(buf+1, 2,
            pcm, st->layout.nb_channels, right, frame_size);
         ptr += align(coupled_size);
      } else {
         int chan = get_mono_channel(&st->layout, s, -1);
         (*copy_channel_in)(buf, 1,
            pcm, st->layout.nb_channels, chan, frame_size);
         ptr += align(mono_size);
      }
      /* number of bytes left (+Toc) */
      curr_max = max_data_bytes - tot_size;
      /* Reserve three bytes for the last stream and four for the others */
      curr_max -= IMAX(0,4*(st->layout.nb_streams-s-1)-1);
      curr_max = IMIN(curr_max,MS_FRAME_TMP);
      len = opus_encode_native(enc, buf, frame_size, tmp_data, curr_max, lsb_depth
#ifndef FIXED_POINT
            , &analysis_info
#endif
            );
      if (len<0)
      {
         RESTORE_STACK;
         return len;
      }
      /* We need to use the repacketizer to add the self-delimiting lengths
         while taking into account the fact that the encoder can now return
         more than one frame at a time (e.g. 60 ms CELT-only) */
      opus_repacketizer_cat(&rp, tmp_data, len);
      len = opus_repacketizer_out_range_impl(&rp, 0, opus_repacketizer_get_nb_frames(&rp), data, max_data_bytes-tot_size, s != st->layout.nb_streams-1);
      data += len;
      tot_size += len;
   }
   RESTORE_STACK;
   return tot_size;

}

#if !defined(DISABLE_FLOAT_API)
static void opus_copy_channel_in_float(
  opus_val16 *dst,
  int dst_stride,
  const void *src,
  int src_stride,
  int src_channel,
  int frame_size
)
{
   const float *float_src;
   opus_int32 i;
   float_src = (const float *)src;
   for (i=0;i<frame_size;i++)
#if defined(FIXED_POINT)
      dst[i*dst_stride] = FLOAT2INT16(float_src[i*src_stride+src_channel]);
#else
      dst[i*dst_stride] = float_src[i*src_stride+src_channel];
#endif
}
#endif

static void opus_copy_channel_in_short(
  opus_val16 *dst,
  int dst_stride,
  const void *src,
  int src_stride,
  int src_channel,
  int frame_size
)
{
   const opus_int16 *short_src;
   opus_int32 i;
   short_src = (const opus_int16 *)src;
   for (i=0;i<frame_size;i++)
#if defined(FIXED_POINT)
      dst[i*dst_stride] = short_src[i*src_stride+src_channel];
#else
      dst[i*dst_stride] = (1/32768.f)*short_src[i*src_stride+src_channel];
#endif
}

#ifdef FIXED_POINT
int opus_multistream_encode(
    OpusMSEncoder *st,
    const opus_val16 *pcm,
    int frame_size,
    unsigned char *data,
    opus_int32 max_data_bytes
)
{
   return opus_multistream_encode_native(st, opus_copy_channel_in_short,
      pcm, frame_size, data, max_data_bytes, 16);
}

#ifndef DISABLE_FLOAT_API
int opus_multistream_encode_float(
    OpusMSEncoder *st,
    const float *pcm,
    int frame_size,
    unsigned char *data,
    opus_int32 max_data_bytes
)
{
   return opus_multistream_encode_native(st, opus_copy_channel_in_float,
      pcm, frame_size, data, max_data_bytes, 16);
}
#endif

#else

int opus_multistream_encode_float
(
    OpusMSEncoder *st,
    const opus_val16 *pcm,
    int frame_size,
    unsigned char *data,
    opus_int32 max_data_bytes
)
{
   int channels = st->layout.nb_streams + st->layout.nb_coupled_streams;
   return opus_multistream_encode_native(st, opus_copy_channel_in_float,
      pcm, frame_size, data, max_data_bytes, 24, downmix_float, pcm+channels*st->analysis.analysis_offset);
}

int opus_multistream_encode(
    OpusMSEncoder *st,
    const opus_int16 *pcm,
    int frame_size,
    unsigned char *data,
    opus_int32 max_data_bytes
)
{
   int channels = st->layout.nb_streams + st->layout.nb_coupled_streams;
   return opus_multistream_encode_native(st, opus_copy_channel_in_short,
      pcm, frame_size, data, max_data_bytes, 16, downmix_int, pcm+channels*st->analysis.analysis_offset);
}
#endif

int opus_multistream_encoder_ctl(OpusMSEncoder *st, int request, ...)
{
   va_list ap;
   int coupled_size, mono_size;
   char *ptr;
   int ret = OPUS_OK;

   va_start(ap, request);

   coupled_size = opus_encoder_get_size(2);
   mono_size = opus_encoder_get_size(1);
   ptr = (char*)st + align(sizeof(OpusMSEncoder));
   switch (request)
   {
   case OPUS_SET_BITRATE_REQUEST:
   {
      opus_int32 value = va_arg(ap, opus_int32);
      if (value<0 && value!=OPUS_AUTO && value!=OPUS_BITRATE_MAX)
         goto bad_arg;
      st->bitrate_bps = value;
   }
   break;
   case OPUS_GET_BITRATE_REQUEST:
   {
      int s;
      opus_int32 *value = va_arg(ap, opus_int32*);
      *value = 0;
      for (s=0;s<st->layout.nb_streams;s++)
      {
         opus_int32 rate;
         OpusEncoder *enc;
         enc = (OpusEncoder*)ptr;
         if (s < st->layout.nb_coupled_streams)
            ptr += align(coupled_size);
         else
            ptr += align(mono_size);
         opus_encoder_ctl(enc, request, &rate);
         *value += rate;
      }
   }
   break;
   case OPUS_GET_LSB_DEPTH_REQUEST:
   case OPUS_GET_VBR_REQUEST:
   case OPUS_GET_APPLICATION_REQUEST:
   case OPUS_GET_BANDWIDTH_REQUEST:
   case OPUS_GET_COMPLEXITY_REQUEST:
   case OPUS_GET_PACKET_LOSS_PERC_REQUEST:
   case OPUS_GET_DTX_REQUEST:
   case OPUS_GET_VOICE_RATIO_REQUEST:
   case OPUS_GET_VBR_CONSTRAINT_REQUEST:
   case OPUS_GET_SIGNAL_REQUEST:
   case OPUS_GET_LOOKAHEAD_REQUEST:
   case OPUS_GET_SAMPLE_RATE_REQUEST:
   case OPUS_GET_INBAND_FEC_REQUEST:
   case OPUS_GET_FORCE_CHANNELS_REQUEST:
   {
      OpusEncoder *enc;
      /* For int32* GET params, just query the first stream */
      opus_int32 *value = va_arg(ap, opus_int32*);
      enc = (OpusEncoder*)ptr;
      ret = opus_encoder_ctl(enc, request, value);
   }
   break;
   case OPUS_GET_FINAL_RANGE_REQUEST:
   {
      int s;
      opus_uint32 *value = va_arg(ap, opus_uint32*);
      opus_uint32 tmp;
      *value=0;
      for (s=0;s<st->layout.nb_streams;s++)
      {
         OpusEncoder *enc;
         enc = (OpusEncoder*)ptr;
         if (s < st->layout.nb_coupled_streams)
            ptr += align(coupled_size);
         else
            ptr += align(mono_size);
         ret = opus_encoder_ctl(enc, request, &tmp);
         if (ret != OPUS_OK) break;
         *value ^= tmp;
      }
   }
   break;
   case OPUS_SET_LSB_DEPTH_REQUEST:
   case OPUS_SET_COMPLEXITY_REQUEST:
   case OPUS_SET_VBR_REQUEST:
   case OPUS_SET_VBR_CONSTRAINT_REQUEST:
   case OPUS_SET_BANDWIDTH_REQUEST:
   case OPUS_SET_SIGNAL_REQUEST:
   case OPUS_SET_APPLICATION_REQUEST:
   case OPUS_SET_INBAND_FEC_REQUEST:
   case OPUS_SET_PACKET_LOSS_PERC_REQUEST:
   case OPUS_SET_DTX_REQUEST:
   case OPUS_SET_FORCE_MODE_REQUEST:
   case OPUS_SET_FORCE_CHANNELS_REQUEST:
   {
      int s;
      /* This works for int32 params */
      opus_int32 value = va_arg(ap, opus_int32);
      for (s=0;s<st->layout.nb_streams;s++)
      {
         OpusEncoder *enc;

         enc = (OpusEncoder*)ptr;
         if (s < st->layout.nb_coupled_streams)
            ptr += align(coupled_size);
         else
            ptr += align(mono_size);
         ret = opus_encoder_ctl(enc, request, value);
         if (ret != OPUS_OK)
            break;
      }
   }
   break;
   case OPUS_MULTISTREAM_GET_ENCODER_STATE_REQUEST:
   {
      int s;
      opus_int32 stream_id;
      OpusEncoder **value;
      stream_id = va_arg(ap, opus_int32);
      if (stream_id<0 || stream_id >= st->layout.nb_streams)
         ret = OPUS_BAD_ARG;
      value = va_arg(ap, OpusEncoder**);
      for (s=0;s<stream_id;s++)
      {
         if (s < st->layout.nb_coupled_streams)
            ptr += align(coupled_size);
         else
            ptr += align(mono_size);
      }
      *value = (OpusEncoder*)ptr;
   }
   break;
   case OPUS_SET_EXPERT_FRAME_DURATION_REQUEST:
   {
       opus_int32 value = va_arg(ap, opus_int32);
       st->variable_duration = value;
   }
   break;
   case OPUS_GET_EXPERT_FRAME_DURATION_REQUEST:
   {
       opus_int32 *value = va_arg(ap, opus_int32*);
       *value = st->variable_duration;
   }
   break;
   default:
      ret = OPUS_UNIMPLEMENTED;
      break;
   }

   va_end(ap);
   return ret;
bad_arg:
   va_end(ap);
   return OPUS_BAD_ARG;
}

void opus_multistream_encoder_destroy(OpusMSEncoder *st)
{
    opus_free(st);
}


