/* Copyright (c) 2022 Amazon */
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


#include "opus_types.h"
#include "opus_defines.h"
#include "arch.h"
#include "os_support.h"
#include "opus_private.h"


/* Given an extension payload (i.e., excluding the initial ID byte), advance
    data to the next extension and return the length of the remaining
    extensions.
   N.B., a "Repeat These Extensions" extension (ID==2) does not advance past
    the repeated extension payloads.
   That requires higher-level logic. */
static opus_int32 skip_extension_payload(const unsigned char **pdata,
 opus_int32 len, opus_int32 *pheader_size, int id_byte)
{
   const unsigned char *data;
   opus_int32 header_size;
   int id, L;
   data = *pdata;
   header_size = 0;
   id = id_byte>>1;
   L = id_byte&1;
   if ((id == 0 && L == 1) || id == 2)
   {
      /* Nothing to do. */
   } else if (id > 0 && id < 32)
   {
      if (len < L)
         return -1;
      data += L;
      len -= L;
   } else {
      if (L==0)
      {
         data += len;
         len = 0;
      } else {
         opus_int32 bytes=0;
         opus_int32 lacing;
         do {
            if (len < 1)
               return -1;
            lacing = *data++;
            bytes += lacing;
            header_size++;
            len -= lacing + 1;
         } while (lacing == 255);
         if (len < 0)
            return -1;
         data += bytes;
      }
   }
   *pdata = data;
   *pheader_size = header_size;
   return len;
}

/* Given an extension, advance data to the next extension and return the
   length of the remaining extensions.
   N.B., a "Repeat These Extensions" extension (ID==2) only advances past the
    extension ID byte.
   Higher-level logic is required to skip the extension payloads that come
    after it.*/
static opus_int32 skip_extension(const unsigned char **pdata, opus_int32 len,
 opus_int32 *pheader_size)
{
   const unsigned char *data;
   int id_byte;
   if (len == 0) {
      *pheader_size = 0;
      return 0;
   }
   if (len < 1)
      return -1;
   data = *pdata;
   id_byte = *data++;
   len--;
   len = skip_extension_payload(&data, len, pheader_size, id_byte);
   if (len >= 0) {
      *pdata = data;
      (*pheader_size)++;
   }
   return len;
}

void opus_extension_iterator_init(OpusExtensionIterator *iter,
 const unsigned char *data, opus_int32 len, opus_int32 nb_frames) {
   celt_assert(len >= 0);
   celt_assert(data != NULL || len == 0);
   celt_assert(nb_frames >= 0 && nb_frames <= 48);
   iter->repeat_data_end = iter->repeat_data = iter->curr_data = iter->data =
    data;
   iter->src_data = NULL;
   iter->curr_len = iter->len = len;
   iter->repeat_len = iter->src_len = 0;
   iter->frame_max = iter->nb_frames = nb_frames;
   iter->repeat_frame = iter->curr_frame = 0;
   iter->repeat_l = 0;
}

/* Reset the iterator so it can start iterating again from the first
    extension. */
void opus_extension_iterator_reset(OpusExtensionIterator *iter) {
   iter->repeat_data_end = iter->repeat_data = iter->curr_data = iter->data;
   iter->curr_len = iter->len;
   iter->repeat_frame = iter->curr_frame = 0;
}

/* Tell the iterator not to return any extensions for frames of index
    frame_max or larger.
   This can allow it to stop iterating early if these extensions are not
    needed. */
void opus_extension_iterator_set_frame_max(OpusExtensionIterator *iter,
 int frame_max) {
   iter->frame_max = frame_max;
}

/* Return the next extension (excluding real padding, separators, and repeat
    indicators, but including the repeated extensions) in bitstream order.
   Due to the extension repetition mechanism, extensions are not necessarily
    returned in frame order. */
int opus_extension_iterator_next(OpusExtensionIterator *iter,
 opus_extension_data *ext) {
   opus_int32 header_size;
   if (iter->curr_len < 0) {
      return OPUS_INVALID_PACKET;
   }
   /* Checking this here allows opus_extension_iterator_set_frame_max() to be
       called at any point. */
   if (iter->curr_frame >= iter->frame_max) {
      return 0;
   }
   if (iter->repeat_frame > 0) {
      /* We are in the process of repeating some extensions. */
      for (;iter->repeat_frame < iter->nb_frames; iter->repeat_frame++) {
         while (iter->src_len > 0) {
            const unsigned char *curr_data0;
            int repeat_id_byte;
            repeat_id_byte = *iter->src_data;
            iter->src_len = skip_extension(&iter->src_data, iter->src_len,
             &header_size);
            /* We skipped this extension earlier, so it should not fail now. */
            celt_assert(iter->src_len >= 0);
            /* Don't repeat padding or frame separators with a 0 increment. */
            if (repeat_id_byte <= 3) continue;
            /* If the "Repeat These Extensions" extension had L == 0 and this
                is the last repeated extension, and it is a long extension,
                then force decoding the payload with L = 0. */
            if (iter->repeat_l == 0
             && iter->repeat_frame + 1 >= iter->nb_frames
             && iter->src_data == iter->repeat_data_end
             && repeat_id_byte >= 64) {
               repeat_id_byte &= ~1;
            }
            curr_data0 = iter->curr_data;
            iter->curr_len = skip_extension_payload(&iter->curr_data,
             iter->curr_len, &header_size, repeat_id_byte);
            if (iter->curr_len < 0) {
               return OPUS_INVALID_PACKET;
            }
            celt_assert(iter->curr_data - iter->data
             == iter->len - iter->curr_len);
            /* If we were asked to stop at frame_max, skip extensions for later
                frames. */
            if (iter->repeat_frame >= iter->frame_max) {
               if (iter->repeat_l == 0) {
                  /* If L == 0, there will be no more extensions after these
                      repeats, so we can just stop. */
                  iter->repeat_frame = 0;
                  iter->curr_len = 0;
                  return 0;
               }
               continue;
            }
            if (ext != NULL) {
               ext->id = repeat_id_byte >> 1;
               ext->frame = iter->repeat_frame;
               ext->data = curr_data0 + header_size;
               ext->len = iter->curr_data - curr_data0 - header_size;
            }
            return 1;
         }
         /* We finished repeating the extensions for this frame. */
         iter->src_data = iter->repeat_data;
         iter->src_len = iter->repeat_len;
      }
      /* We finished repeating extensions. */
      iter->repeat_data_end = iter->repeat_data = iter->curr_data;
      /* Even if there is more data (because there was nothing to repeat or
          because the last extension was a short extension and we did not use
          all the data), when L == 0 we are done decoding extensions. */
      if (iter->repeat_l == 0) {
         iter->curr_len = 0;
      }
      iter->repeat_frame = 0;
   }
   while (iter->curr_len > 0) {
      const unsigned char *curr_data0;
      int id;
      int L;
      curr_data0 = iter->curr_data;
      id = *curr_data0>>1;
      L = *curr_data0&1;
      iter->curr_len = skip_extension(&iter->curr_data, iter->curr_len,
       &header_size);
      if (iter->curr_len < 0) {
         return OPUS_INVALID_PACKET;
      }
      celt_assert(iter->curr_data - iter->data == iter->len - iter->curr_len);
      if (id == 1) {
         if (L == 0) {
            iter->curr_frame++;
         }
         else {
            /* A frame increment of 0 is a no-op. */
            if (!curr_data0[1]) continue;
            iter->curr_frame += curr_data0[1];
         }
         if (iter->curr_frame >= iter->nb_frames) {
            iter->curr_len = -1;
            return OPUS_INVALID_PACKET;
         }
         /* If we were asked to stop at frame_max, skip extensions for later
             frames. */
         if (iter->curr_frame >= iter->frame_max) {
            iter->curr_len = 0;
         }
         iter->repeat_data_end = iter->repeat_data = iter->curr_data;
      }
      else if (id == 2) {
         iter->repeat_l = L;
         iter->repeat_frame = iter->curr_frame + 1;
         iter->repeat_len = curr_data0 - iter->repeat_data;
         iter->src_data = iter->repeat_data;
         iter->src_len = iter->repeat_len;
         return opus_extension_iterator_next(iter, ext);
      }
      else if (id > 2) {
         /* Update the stopping point for repeating extension data.
            This lets us detect when we have hit the last repeated extension,
             for purposes of modifying the L flag if it is a long extension.
            Only extensions which can have a non-empty payload count, as we can
             still use L=0 to code a final long extension if there cannot be
             any more payload data, even if there are more short L=0
             extensions (or padding). */
         if (L || id >= 32) iter->repeat_data_end = iter->curr_data;
         if (ext != NULL) {
            ext->id = id;
            ext->frame = iter->curr_frame;
            ext->data = curr_data0 + header_size;
            ext->len = iter->curr_data - curr_data0 - header_size;
         }
         return 1;
      }
   }
   return 0;
}

int opus_extension_iterator_find(OpusExtensionIterator *iter,
 opus_extension_data *ext, int id) {
   opus_extension_data curr_ext;
   int ret;
   for(;;) {
      ret = opus_extension_iterator_next(iter, &curr_ext);
      if (ret <= 0) {
         return ret;
      }
      if (curr_ext.id == id) {
         *ext = curr_ext;
         return ret;
      }
   }
}

/* Count the number of extensions, excluding real padding, separators, and
    repeat indicators, but including the repeated extensions. */
opus_int32 opus_packet_extensions_count(const unsigned char *data,
 opus_int32 len, int nb_frames)
{
   OpusExtensionIterator iter;
   int count;
   opus_extension_iterator_init(&iter, data, len, nb_frames);
   for (count=0; opus_extension_iterator_next(&iter, NULL) > 0; count++);
   return count;
}

/* Count the number of extensions for each frame, excluding real padding and
    separators and repeat indicators, but including the repeated extensions. */
opus_int32 opus_packet_extensions_count_ext(const unsigned char *data,
 opus_int32 len, opus_int32 *nb_frame_exts, int nb_frames) {
   OpusExtensionIterator iter;
   opus_extension_data ext;
   int count;
   opus_extension_iterator_init(&iter, data, len, nb_frames);
   OPUS_CLEAR(nb_frame_exts, nb_frames);
   for (count=0; opus_extension_iterator_next(&iter, &ext) > 0; count++) {
      nb_frame_exts[ext.frame]++;
   }
   return count;
}

/* Extract extensions from Opus padding (excluding real padding, separators,
    and repeat indicators, but including the repeated extensions) in bitstream
    order.
   Due to the extension repetition mechanism, extensions are not necessarily
    returned in frame order. */
opus_int32 opus_packet_extensions_parse(const unsigned char *data,
 opus_int32 len, opus_extension_data *extensions, opus_int32 *nb_extensions,
 int nb_frames) {
   OpusExtensionIterator iter;
   int count;
   int ret;
   celt_assert(nb_extensions != NULL);
   celt_assert(extensions != NULL || *nb_extensions == 0);
   opus_extension_iterator_init(&iter, data, len, nb_frames);
   for (count=0;; count++) {
      opus_extension_data ext;
      ret = opus_extension_iterator_next(&iter, &ext);
      if (ret <= 0) break;
      if (count == *nb_extensions) {
         return OPUS_BUFFER_TOO_SMALL;
      }
      extensions[count] = ext;
   }
   *nb_extensions = count;
   return ret;
}

/* Extract extensions from Opus padding (excluding real padding, separators,
    and repeat indicators, but including the repeated extensions) in frame
    order.
   nb_frame_exts must be filled with the output of
    opus_packet_extensions_count_ext(). */
opus_int32 opus_packet_extensions_parse_ext(const unsigned char *data,
 opus_int32 len, opus_extension_data *extensions, opus_int32 *nb_extensions,
 const opus_int32 *nb_frame_exts, int nb_frames) {
   OpusExtensionIterator iter;
   opus_extension_data ext;
   opus_int32 nb_frames_cum[49];
   int count;
   int prev_total;
   int ret;
   celt_assert(nb_extensions != NULL);
   celt_assert(extensions != NULL || *nb_extensions == 0);
   celt_assert(nb_frames <= 48);
   /* Convert the frame extension count array to a cumulative sum. */
   prev_total = 0;
   for (count=0; count<nb_frames; count++) {
      int total;
      total = nb_frame_exts[count] + prev_total;
      nb_frames_cum[count] = prev_total;
      prev_total = total;
   }
   nb_frames_cum[count] = prev_total;
   opus_extension_iterator_init(&iter, data, len, nb_frames);
   for (count=0;; count++) {
      opus_int32 idx;
      ret = opus_extension_iterator_next(&iter, &ext);
      if (ret <= 0) break;
      idx = nb_frames_cum[ext.frame]++;
      if (idx >= *nb_extensions) {
         return OPUS_BUFFER_TOO_SMALL;
      }
      celt_assert(idx < nb_frames_cum[ext.frame + 1]);
      extensions[idx] = ext;
   }
   *nb_extensions = count;
   return ret;
}

opus_int32 opus_packet_extensions_generate(unsigned char *data, opus_int32 len, const opus_extension_data  *extensions, opus_int32 nb_extensions, int pad)
{
   int max_frame=0;
   opus_int32 i;
   int frame;
   int curr_frame = 0;
   opus_int32 pos = 0;
   opus_int32 written = 0;

   celt_assert(len >= 0);

   for (i=0;i<nb_extensions;i++)
   {
      max_frame = IMAX(max_frame, extensions[i].frame);
      if (extensions[i].id < 3 || extensions[i].id > 127)
         return OPUS_BAD_ARG;
   }
   if (max_frame >= 48) return OPUS_BAD_ARG;
   for (frame=0;frame<=max_frame;frame++)
   {
      for (i=0;i<nb_extensions;i++)
      {
         if (extensions[i].frame == frame)
         {
            /* Insert separator when needed. */
            if (frame != curr_frame) {
               int diff = frame - curr_frame;
               if (len-pos < 2)
                  return OPUS_BUFFER_TOO_SMALL;
               if (diff == 1) {
                  if (data) data[pos] = 0x02;
                  pos++;
               } else {
                  if (data) data[pos] = 0x03;
                  pos++;
                  if (data) data[pos] = diff;
                  pos++;
               }
               curr_frame = frame;
            }
            if (extensions[i].id < 32)
            {
               if (extensions[i].len < 0 || extensions[i].len > 1)
                  return OPUS_BAD_ARG;
               if (len-pos < extensions[i].len+1)
                  return OPUS_BUFFER_TOO_SMALL;
               if (data) data[pos] = (extensions[i].id<<1) + extensions[i].len;
               pos++;
               if (extensions[i].len > 0) {
                  if (data) data[pos] = extensions[i].data[0];
                  pos++;
               }
            } else {
               int last;
               opus_int32 length_bytes;
               if (extensions[i].len < 0)
                  return OPUS_BAD_ARG;
               last = (written == nb_extensions - 1);
               length_bytes = 1 + extensions[i].len/255;
               if (last)
                  length_bytes = 0;
               if (len-pos < 1 + length_bytes + extensions[i].len)
                  return OPUS_BUFFER_TOO_SMALL;
               if (data) data[pos] = (extensions[i].id<<1) + !last;
               pos++;
               if (!last)
               {
                  opus_int32 j;
                  for (j=0;j<extensions[i].len/255;j++) {
                     if (data) data[pos] = 255;
                     pos++;
                  }
                  if (data) data[pos] = extensions[i].len % 255;
                  pos++;
               }
               if (data) OPUS_COPY(&data[pos], extensions[i].data, extensions[i].len);
               pos += extensions[i].len;
            }
            written++;
         }
      }
   }
   /* If we need to pad, just prepend 0x01 bytes. Even better would be to fill the
      end with zeros, but that requires checking that turning the last extesion into
      an L=1 case still fits. */
   if (pad && pos < len)
   {
      opus_int32 padding = len - pos;
      if (data) {
         OPUS_MOVE(data+padding, data, pos);
         for (i=0;i<padding;i++)
            data[i] = 0x01;
      }
      pos += padding;
   }
   return pos;
}

#if 0
#include <stdio.h>
int main()
{
   opus_extension_data ext[] = {{2, 0, (const unsigned char *)"a", 1},
   {32, 10, (const unsigned char *)"DRED", 4},
   {33, 1, (const unsigned char *)"NOT DRED", 8},
   {3, 4, (const unsigned char *)NULL, 0}
   };
   opus_extension_data ext2[10];
   int i, len;
   int nb_ext = 10;
   unsigned char packet[10000];
   len = opus_packet_extensions_generate(packet, 32, ext, 4, 1);
   for (i=0;i<len;i++)
   {
      printf("%#04x ", packet[i]);
      if (i%16 == 15)
         printf("\n");
   }
   printf("\n");
   printf("count = %d\n", opus_packet_extensions_count(packet, len));
   opus_packet_extensions_parse(packet, len, ext2, &nb_ext);
   for (i=0;i<nb_ext;i++)
   {
      int j;
      printf("%d %d {", ext2[i].id, ext2[i].frame);
      for (j=0;j<ext2[i].len;j++) printf("%#04x ", ext2[i].data[j]);
      printf("} %d\n", ext2[i].len);
   }
}
#endif
