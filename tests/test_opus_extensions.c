/* Copyright (c) 2023 Amazon
   Written by Michael Klingbeil */
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifndef _WIN32
#include <unistd.h>
#else
#include <process.h>
#define getpid _getpid
#endif

#include "../src/opus_private.h"
#include "test_opus_common.h"

void test_extensions_generate_success(void)
{
   static const opus_extension_data ext[] = {
      {(const unsigned char *)"a", 1, 3, 0, 1},
      {(const unsigned char *)"DRED", 4, 32, 10, 1},
      {(const unsigned char *)"NOT DRED", 8, 33, 1, 1},
      {(const unsigned char *)NULL, 0, 4, 4, 1}
   };

   int result;
   unsigned char packet[32];
   const unsigned char *p = packet;
   result = opus_packet_extensions_generate(packet, 23+4, ext, 4, 11, 1);
   expect_true(result == 23+4, "expected length 23+4");

   /* expect padding */
   expect_true(p[0] == 1 && p[1] == 1 && p[2] == 1 && p[3] == 1, "expected padding");
   p += 4;

   /* extension ID=3 */
   expect_true((p[0] >> 1) == 3, "expected extension id 3");
   /* For extension IDs 1 through 31, L=0 means that no data follows the
      extension, whereas L=1 means that exactly one byte of extension data follows. */
   expect_true((p[0] & 0x01) == 1, "expected L-bit set");
   /* content */
   expect_true(p[1] == 'a', "expected extension content");
   p += 2;

   /* next byte should increment the frame count, ID=1, L=0 */
   expect_true(p[0] == 0x02, "bad frame separator");
   p += 1;
   /* extension ID=33 */
   expect_true((p[0] >> 1) == 33, "expected extension id 33");
   /* For IDs 32 to 127, L=0 signals that the extension data takes up the
      rest of the padding, and L=1 signals that a length indicator follows. */
   expect_true((p[0] & 0x01) == 1, "expected L-bit set");
   /* content */
   expect_true(p[1] == ext[2].len, "expected length");
   p += 2;
   expect_true(0 == memcmp(p, ext[2].data, ext[2].len), "expected extension content");
   p += ext[2].len;

   /* advance to frame 4, increment by 3 */
   /* next byte should increment the frame count, ID=1, L=1 */
   expect_true(p[0] == 0x03, "bad frame separator");
   expect_true(p[1] == 0x03, "bad frame increment");
   p += 2;
   /* extension ID=4 */
   expect_true((p[0] >> 1) == 4, "expected extension id 4");
   /* For extension IDs 1 through 31, L=0 means that no data follows the
      extension, whereas L=1 means that exactly one byte of extension data follows. */
   expect_true((p[0] & 0x01) == 0, "expected L-bit unset");
   p += 1;

   /* advance to frame 10, increment by 6 */
   /* next byte should increment the frame count, ID=1, L=1 */
   expect_true(p[0] == 0x03, "bad frame separator");
   expect_true(p[1] == 0x06, "bad frame increment");
   p += 2;
   /* extension ID=32 */
   expect_true((p[0] >> 1) == 32, "expected extension id 32");
   /* For IDs 32 to 127, L=0 signals that the extension data takes up the
      rest of the padding */
   expect_true((p[0] & 0x01) == 0, "expected L-bit unset");
   p += 1;
   expect_true(0 == memcmp(p, ext[1].data, ext[1].len), "expected extension content");
}

void test_extensions_generate_zero(void)
{
   int result;
   unsigned char packet[32];

   /* zero length packet, zero extensions */
   result = opus_packet_extensions_generate(packet, 0, NULL, 0, 0, 1);
   expect_true(result == 0, "expected length 0");
}

void test_extensions_generate_no_padding(void)
{
   static const opus_extension_data ext[] = {
      {(const unsigned char *)"a", 1, 3, 0, 1},
      {(const unsigned char *)"DRED", 4, 32, 10, 1},
      {(const unsigned char *)"NOT DRED", 8, 33, 1, 1},
      {(const unsigned char *)NULL, 0, 4, 4, 1}
   };

   int result;
   unsigned char packet[32];
   result =
    opus_packet_extensions_generate(packet, sizeof(packet), ext, 4, 11, 0);
   expect_true(result == 23, "expected length 23");
}

void test_extensions_generate_fail(void)
{
   static const opus_extension_data ext[] = {
      {(const unsigned char *)"a", 1, 3, 0, 1},
      {(const unsigned char *)"DRED", 4, 32, 10, 1},
      {(const unsigned char *)"NOT DRED", 8, 33, 1, 1},
      {(const unsigned char *)NULL, 0, 4, 4, 1}
   };

   int result;
   unsigned char packet[100];

   /* buffer too small */
   {
      opus_int32 len;
      /* this failure can occur at lots of points, so iterate to check as many
          as possible */
      for (len=0;len<23;len++)
      {
         size_t i;
         for (i=len;i<sizeof(packet);i++) packet[i] = 0xFE;
         result = opus_packet_extensions_generate(packet, len, ext, 4, 11, 1);
         expect_true(result == OPUS_BUFFER_TOO_SMALL,
          "expected OPUS_BUFFER_TOO_SMALL");
         for (i=len;i<sizeof(packet);i++)
         {
            expect_true(packet[i] == 0xFE,
             "expected 0xFE padding to be undisturbed");
         }
      }
   }

   /* invalid id */
   {
      static const opus_extension_data id_too_big[] = {
         {(const unsigned char *)"a", 1, 256, 0, 1},
      };
      result = opus_packet_extensions_generate(packet, sizeof(packet),
       id_too_big, 1, 11, 1);
      expect_true(result == OPUS_BAD_ARG, "expected OPUS_BAD_ARG");
   }

   /* invalid id */
   {
      static const opus_extension_data id_too_small[] = {
         {(const unsigned char *)"a", 1, 2, 0, 1},
      };
      result = opus_packet_extensions_generate(packet, sizeof(packet),
       id_too_small, 1, 11, 1);
      expect_true(result == OPUS_BAD_ARG, "expected OPUS_BAD_ARG");
   }

   /* frame count too big */
   {
      static const opus_extension_data frame_too_big[] = {
         {(const unsigned char *)"a", 1, 33, 11, 1},
      };
      result = opus_packet_extensions_generate(packet, sizeof(packet),
       frame_too_big, 1, 49, 1);
      expect_true(result == OPUS_BAD_ARG, "expected OPUS_BAD_ARG");
   }

   /* frame index too small */
   {
      static const opus_extension_data frame_too_big[] = {
         {(const unsigned char *)"a", 1, 33, -1, 1},
      };
      result = opus_packet_extensions_generate(packet, sizeof(packet),
       frame_too_big, 1, 11, 1);
      expect_true(result == OPUS_BAD_ARG, "expected OPUS_BAD_ARG");
   }

   /* frame index too big */
   {
      static const opus_extension_data frame_too_big[] = {
         {(const unsigned char *)"a", 1, 33, 11, 1},
      };
      result = opus_packet_extensions_generate(packet, sizeof(packet),
       frame_too_big, 1, 11, 1);
      expect_true(result == OPUS_BAD_ARG, "expected OPUS_BAD_ARG");
   }

   /* size too big for extension IDs 1 through 31 */
   {
      static const opus_extension_data size_too_big[] = {
         {(const unsigned char *)"abcd", 4, 3, 0, 1},
      };
      result = opus_packet_extensions_generate(packet, sizeof(packet),
       size_too_big, 1, 1, 1);
      expect_true(result == OPUS_BAD_ARG, "expected OPUS_BAD_ARG");
   }

   /* negative size for extension IDs 1 through 31 */
   {
      static const opus_extension_data neg_size[] = {
         {NULL, -4, 3, 0, 1},
      };
      result = opus_packet_extensions_generate(packet, sizeof(packet),
       neg_size, 1, 1, 1);
      expect_true(result == OPUS_BAD_ARG, "expected OPUS_BAD_ARG");
   }

   /* negative size for extension IDs 32 through 127 */
   {
      static const opus_extension_data neg_size_33[] = {
         {NULL, -4, 33, 0, 1},
      };
      result = opus_packet_extensions_generate(packet, sizeof(packet),
       neg_size_33, 1, 1, 1);
      expect_true(result == OPUS_BAD_ARG, "expected OPUS_BAD_ARG");
   }

   /* length + lacing overflow for non-final long extension */
   {
      static const opus_extension_data huge_len[] = {
         {(const unsigned char *)"a", 0x7FFFFFFF, 33, 0, 1},
         {NULL, 0, 3, 0, 1}
      };
      memset(packet + 16, 0xFE, sizeof(packet) - 16);
      result = opus_packet_extensions_generate(packet, 16, huge_len, 2, 1, 0);
      expect_true(result == OPUS_BUFFER_TOO_SMALL && packet[16] == 0xFE,
       "expected OPUS_BUFFER_TOO_SMALL without signed overflow");
      expect_true(opus_packet_extensions_generate(NULL, 0x7FFFFFFF, huge_len,
       2, 1, 0) == OPUS_BUFFER_TOO_SMALL,
       "expected OPUS_BUFFER_TOO_SMALL on size query overflow");
      expect_true(opus_packet_extensions_generate(packet, -1, NULL, 0, 1, 0)
       == OPUS_BAD_ARG
       && opus_packet_extensions_generate(packet, 16, NULL, 1, 1, 0)
       == OPUS_BAD_ARG, "expected OPUS_BAD_ARG for negative len or NULL exts");
   }
}

void test_extensions_parse_success(void)
{
   static const opus_extension_data ext[] = {
      {(const unsigned char *)"a", 1, 3, 0, 1},
      {(const unsigned char *)"DRED", 4, 32, 10, 1},
      {(const unsigned char *)"NOT DRED", 8, 33, 1, 1},
      {(const unsigned char *)NULL, 0, 4, 4, 1}
   };
   opus_extension_data ext_out[10];
   opus_int32 nb_ext;
   int nb_frames;
   int len, result;
   unsigned char packet[32];

   nb_ext = 10;
   nb_frames = 11;
   len = opus_packet_extensions_generate(packet, 32, ext, 4, 11, 1);
   expect_true(len == 32, "expected length 32");
   result = opus_packet_extensions_count(packet, len, nb_frames);
   expect_true(result == 4, "expected opus_packet_extensions_count 4");
   result = opus_packet_extensions_parse(packet, len, ext_out, &nb_ext,
    nb_frames);
   expect_true(nb_ext == 4, "expected 4 extensions");

   expect_true(ext_out[0].id == 3, "expected id 3");
   expect_true(ext_out[0].frame == 0, "expected frame 0");
   expect_true(ext_out[0].len == 1, "expected len 1");
   expect_true(0 == memcmp(ext_out[0].data, ext[0].data, 1), "expected data");

   expect_true(ext_out[1].id == 33, "expected id 33");
   expect_true(ext_out[1].frame == 1, "expected frame 1");
   expect_true(ext_out[1].len == 8, "expected len 8");
   expect_true(0 == memcmp(ext_out[1].data, ext[2].data, 8), "expected data");

   expect_true(ext_out[2].id == 4, "expected id 4");
   expect_true(ext_out[2].frame == 4, "expected frame 4");
   expect_true(ext_out[2].len == 0, "expected len 0");

   expect_true(ext_out[3].id == 32, "expected id 32");
   expect_true(ext_out[3].frame == 10, "expected frame 10");
   expect_true(ext_out[3].len == 4, "expected len 4");
   expect_true(0 == memcmp(ext_out[3].data, ext[1].data, 4), "expected data");
}

void test_extensions_parse_zero(void)
{
   static const opus_extension_data ext[] = {
      {(const unsigned char *)"DRED", 4, 32, 1, 1},
   };
   opus_int32 nb_ext;
   int nb_frames;
   int len, result;
   unsigned char packet[32];

   len = opus_packet_extensions_generate(packet, 32, ext, 1, 2, 1);
   expect_true(len == 32, "expected length 32");

   nb_ext = 0;
   nb_frames = 2;
   result = opus_packet_extensions_parse(packet, len, NULL, &nb_ext, nb_frames);
   expect_true(result == OPUS_BUFFER_TOO_SMALL, "expected OPUS_BUFFER_TOO_SMALL");
}

void test_extensions_parse_fail(void)
{
   static const opus_extension_data ext[] = {
      {(const unsigned char *)"a", 1, 3, 0, 1},
      {(const unsigned char *)"NOT DRED", 8, 33, 1, 1},
      {(const unsigned char *)NULL, 0, 4, 4, 1},
      {(const unsigned char *)"DRED", 4, 32, 10, 1},
      {(const unsigned char *)"DRED", 4, 32, 9, 1},
      {(const unsigned char *)"b", 1, 4, 9, 1},
      {(const unsigned char *)"c", 1, 4, 10, 1}
   };
   opus_extension_data ext_out[10];
   opus_int32 nb_ext;
   int nb_frames;
   int len, result;
   unsigned char packet[32];

   /* create invalid length */
   len = opus_packet_extensions_generate(packet, sizeof(packet), ext, 4, 11, 0);
   packet[4] = 255;
   nb_ext = 10;
   nb_frames = 11;
   result = opus_packet_extensions_parse(packet, len, ext_out, &nb_ext,
    nb_frames);
   expect_true(result == OPUS_INVALID_PACKET, "expected OPUS_INVALID_PACKET");
   /* note, opus_packet_extensions_count stops at the invalid frame increment
      and tells us that we have 1 extension */
   result = opus_packet_extensions_count(packet, len, nb_frames);
   expect_true(result == 1, "expected opus_packet_extensions_count to return 1");

   /* create invalid frame increment */
   nb_ext = 10;
   len = opus_packet_extensions_generate(packet, sizeof(packet), ext, 4, 11, 0);
   /* first by reducing the number of frames */
   result = opus_packet_extensions_parse(packet, len, ext_out, &nb_ext, 5);
   expect_true(result == OPUS_INVALID_PACKET, "expected OPUS_INVALID_PACKET");

   /* note, opus_packet_extensions_count stops at the invalid frame increment
      and tells us that we have 3 extensions */
   result = opus_packet_extensions_count(packet, len, 5);
   expect_true(result == 3, "expected opus_packet_extensions_count to return 3");

   /* then by increasing the increment */
   packet[14] = 255;
   result = opus_packet_extensions_parse(packet, len, ext_out, &nb_ext,
    nb_frames);
   expect_true(result == OPUS_INVALID_PACKET, "expected OPUS_INVALID_PACKET");

   /* note, opus_packet_extensions_count stops at the invalid frame increment
      and tells us that we have 2 extensions */
   result = opus_packet_extensions_count(packet, len, nb_frames);
   expect_true(result == 2, "expected opus_packet_extensions_count to return 2");

   /* not enough space */
   nb_ext = 1;
   len = opus_packet_extensions_generate(packet, sizeof(packet), ext, 4, 11, 0);
   result = opus_packet_extensions_parse(packet, len, ext_out, &nb_ext,
    nb_frames);
   expect_true(result == OPUS_BUFFER_TOO_SMALL, "expected OPUS_BUFFER_TOO_SMALL");

   /* create repeated L=0 long extension without enough room for all of the
       short extension payloads that need to follow it */
   len = opus_packet_extensions_generate(packet, sizeof(packet), ext, 7, 11, 0);
   len -= 5;
   nb_ext = 10;
   nb_frames = 11;
   result = opus_packet_extensions_parse(packet, len, ext_out, &nb_ext,
    nb_frames);
   expect_true(result == OPUS_INVALID_PACKET, "expected OPUS_INVALID_PACKET");
   /* note, opus_packet_extensions_count stops at the invalid long extension
       and tells us that we have 5 extensions */
   result = opus_packet_extensions_count(packet, len, nb_frames);
   expect_true(result == 5, "expected opus_packet_extensions_count to return 5");

   /* overflow for long extension length */
   {
      /* about 8 MB */
#define LENSIZE ((1U<<31)/255 + 1)
      unsigned char *buf = (unsigned char *)malloc(LENSIZE+1);
      len = LENSIZE+1;
      buf[0] = 33<<1 | 1;
      memset(buf + 1, 0xFF, LENSIZE - 1);
      buf[LENSIZE] = 0xFE;
      result = opus_packet_extensions_parse(buf, len, ext_out, &nb_ext, 1);
      expect_true(result == OPUS_INVALID_PACKET, "expected OPUS_INVALID_PACKET");
      free(buf);
   }
}

/* Check that the same extensions with the same data appear in both ext_in and
    ext_out, with extensions in ext_out appearing in frame order, and the order
    within a frame matching the order within a frame in ext_in. */
static void check_ext_data(const opus_extension_data *ext_in,
 const opus_extension_data *ext_out, opus_int32 nb_ext)
{
   opus_int32 i;
   opus_int32 j;
   int prev_frame;
   prev_frame = -1;
   j = 0;
   for (i=0;i<nb_ext;i++)
   {
      expect_true(ext_out[i].frame >= prev_frame,
       "expected parsed extensions to be returned in frame order");
      if (ext_out[i].frame > prev_frame) {
         j = 0;
      }
      while (j < nb_ext && ext_in[j].frame != ext_out[i].frame) j++;
      expect_true(j < nb_ext,
       "expected enough extensions matching this frame");
      expect_true(ext_in[j].id == ext_out[i].id,
       "expected extension IDs to match");
      expect_true(ext_in[j].len == ext_out[i].len,
       "expected extension lengths to match");
      /* The len check allows ext_in[j].data to be NULL without triggering
          undefined behavior. */
      expect_true(ext_in[j].len == 0 ||
       memcmp(ext_in[j].data, ext_out[i].data, ext_out[i].len) == 0,
       "expected extension data to match");
      prev_frame = ext_out[i].frame;
      j++;
   }
}

#define NB_EXT (13)

void test_extensions_repeating(void)
{
   static const opus_extension_data ext[] = {
      {(const unsigned char *)"a", 1, 3, 0, 1},
      {(const unsigned char *)"b", 1, 3, 1, 1},
      {(const unsigned char *)"c", 1, 3, 2, 1},
      {(const unsigned char *)"d", 1, 4, 0, 1},
      {(const unsigned char *)NULL, 0, 4, 1, 1},
      {(const unsigned char *)NULL, 0, 4, 2, 1},
      {(const unsigned char *)"DRED2", 5, 32, 2, 1},
      {(const unsigned char *)"DRED", 4, 32, 1, 1},
      {(const unsigned char *)NULL, 0, 5, 1, 1},
      {(const unsigned char *)NULL, 0, 5, 2, 1},
      {(const unsigned char *)"f", 1, 6, 2, 1},
      {(const unsigned char *)"e", 1, 6, 1, 1},
      {(const unsigned char *)"DREDthree", 9, 32, 2, 1}
   };
   static const opus_int32 encoded_len[] = {
      0,
      2,
      /* nb_ext = 2: don't try to repeat if the same extension is not used in
          every frame */
      5,
      /* nb_ext = 3: do repeat if the same extension is used in every frame */
      5,
      7,
      9,
      /* nb_ext = 6: do not repeat short extensions if the lengths do not match
          ... but do repeat after the first frame when the lengths do match. */
      10,
      /* nb_ext = 7: code repeated extensions with L=0 to skip a frame
          separator because they are all short extensions. */
      16,
      /* nb_ext = 8: repeat multiple extensions in the same frame.
         code the last repeated extension with L=0 if it is a long
          extension. */
      21,
      23,
      /* nb_ext = 10: code the last repeated long extension with L=0 even if it
          is followed by repeated short extensions with L=0. */
      22,
      /* nb_ext = 11: don't use L=0 to skip a frame separator if repeats end
          on a short extension if there was a preceding L=0 long extension. */
      26,
      /* nb_ext = 12: code the last repeated long extension with L=0 even if
          it is followed by repeated short extensions with L=1. */
      25,
      /* nb_ext = 13: don't use L=0 to skip a frame separator if repeats end
          on a short extension if there was a preceding L=1 long extension. */
      37
   };
   opus_int32 nb_ext;
   for (nb_ext = 0; nb_ext <= NB_EXT; nb_ext++) {
      opus_extension_data ext_out[NB_EXT];
      opus_int32 nb_frame_exts[48];
      opus_int32 nb_ext_out;
      int len, result;
      unsigned char packet[64];
      len = opus_packet_extensions_generate(packet, sizeof(packet), ext,
       nb_ext, 3, 0);
      expect_true(len == encoded_len[nb_ext],
       "expected extension encoding length to match");
      result = opus_packet_extensions_count_ext(packet, len, nb_frame_exts, 3);
      expect_true(result == nb_ext, "expected extension count to match");
      nb_ext_out = NB_EXT;
      result = opus_packet_extensions_parse_ext(packet, len, ext_out,
       &nb_ext_out, nb_frame_exts, 3);
      expect_true(result == 0, "expected extension parsing to succeed");
      expect_true(nb_ext_out == nb_ext, "expected extension count to match");
      check_ext_data(ext, ext_out, nb_ext);
      /* Special case some modifications to test things our generator will
          never produce. */
      if (nb_ext == 6) {
         /* allow a repeat in the last frame, as well as trailing junk after an
             L=0 repeat that MUST be ignored. */
         packet[len++] = 2<<1|0;
         packet[len++] = 3<<1|0;
      }
      else if (nb_ext == 8) {
         /* insert padding before the repeat indicator */
         memmove(packet+16, packet+15, len-15);
         packet[15] = 0x1;
         len++;
         /* don't repeat the padding and continue decoding last extension with
             L=0 */
      }
      else if (nb_ext == 10) {
         /* use a frame separator with an increment of 0 as padding.
            This should _not_ change which extensions get repeated. */
         memmove(packet+17, packet+15, len-15);
         packet[15] = 0x3;
         packet[16] = 0;
         len += 2;
      }
      else if (nb_ext == 13) {
         /* use L=0 to skip a frame separator if a repeat has no extensions at
             all */
         packet[26] = 2<<1|0;
      }
      else continue;
      result = opus_packet_extensions_count_ext(packet, len, nb_frame_exts, 3);
      expect_true(result == nb_ext, "expected extension count to match");
      nb_ext_out = NB_EXT;
      result = opus_packet_extensions_parse_ext(packet, len, ext_out,
       &nb_ext_out, nb_frame_exts, 3);
      expect_true(result == 0, "expected extension parsing to succeed");
      expect_true(nb_ext_out == nb_ext, "expected extension count to match");
      check_ext_data(ext, ext_out, nb_ext);
      if (nb_ext == 8) {
         /* allow multiple repeat indicators in the same frame */
         memmove(packet+10, packet+9, len-9);
         packet[9] = 2<<1|1;
         len++;
         /* even when there are no new extensions to repeat */
         memmove(packet+6, packet+5, len-5);
         packet[5] = 2<<1|1;
         len++;
      }
      else continue;
      result = opus_packet_extensions_count_ext(packet, len, nb_frame_exts, 3);
      expect_true(result == nb_ext, "expected extension count to match");
      nb_ext_out = NB_EXT;
      result = opus_packet_extensions_parse_ext(packet, len, ext_out,
       &nb_ext_out, nb_frame_exts, 3);
      expect_true(result == 0, "expected extension parsing to succeed");
      expect_true(nb_ext_out == nb_ext, "expected extension count to match");
      check_ext_data(ext, ext_out, nb_ext);
   }
}

#define NB_RANDOM_EXTENSIONS 100000000
#define MAX_EXTENSION_SIZE 200
/* The worst case is a 48-frame packet filled entirely with 0-byte short
    extensions followed by an RTE. */
#define MAX_NB_EXTENSIONS ((MAX_EXTENSION_SIZE - 1)*48)

void test_random_extensions_parse(void)
{
   int i;
   for (i=0;i<NB_RANDOM_EXTENSIONS;i++)
   {
      opus_extension_data ext_out[MAX_NB_EXTENSIONS];
      int nb_frames;
      opus_int32 nb_ext;
      unsigned char payload[MAX_EXTENSION_SIZE];
      int len;
      int j;
      int result;
      len = fast_rand()%(MAX_EXTENSION_SIZE+1);
      for (j=0;j<len;j++)
         payload[j] = fast_rand()&0xFF;
      nb_ext = fast_rand()%(MAX_NB_EXTENSIONS+1);
      /* The success rate is near 60.4% almost independent of the number of
          frames we choose, so pick it randomly.
         nb_frames == 1 is slightly (~0.2%) less likely to succeed, while
          nb_frames == 2 is slightly (~0.1%) more likely to succeed, followed
          by nb_frames == 3 being very slightly (~0.02%) more likely. */
      nb_frames = (fast_rand()%48) + 1;
      result =
       opus_packet_extensions_parse(payload, len, ext_out, &nb_ext, nb_frames);
      expect_true(result == OPUS_OK || result == OPUS_BUFFER_TOO_SMALL || result == OPUS_INVALID_PACKET, "expected OPUS_OK, OPUS_BUFFER_TOO_SMALL or OPUS_INVALID_PACKET");
      /* Even if parsing fails, check that the extensions that got extracted make sense. */
      for (j=0;j<nb_ext;j++)
      {
         expect_true(ext_out[j].frame >= 0 && ext_out[j].frame < nb_frames,
          "expected frame between 0 and nb_frames-1");
         expect_true(ext_out[j].id >= 2 && ext_out[j].id <= 127, "expected id between 2 and 127");
         expect_true(ext_out[j].data >= payload && ext_out[j].data+ext_out[j].len <= payload+len, "expected data to be within packet");
      }
      /* If parsing succeeds, check to see if we can round-trip these
          extensions. */
      if (result == OPUS_OK) {
         opus_extension_data ext_out2[MAX_NB_EXTENSIONS];
         unsigned char payload2[MAX_EXTENSION_SIZE+1];
         opus_int32 nb_frame_exts[48];
         opus_int32 nb_ext_out;
         int orig_len = len;
         len = opus_packet_extensions_generate(payload2, sizeof(payload2),
          ext_out, nb_ext, nb_frames, 0);
         expect_true(len >= 0, "expected extension generation to succeed");
         result = opus_packet_extensions_count_ext(payload2, len,
          nb_frame_exts, nb_frames);
         expect_true(result == nb_ext, "expected extension count to match");
         nb_ext_out = MAX_NB_EXTENSIONS;
         result = opus_packet_extensions_parse_ext(payload2, len, ext_out2,
          &nb_ext_out, nb_frame_exts, nb_frames);
         expect_true(result == OPUS_OK,
          "expected extension parsing to succeed");
         expect_true(nb_ext == nb_ext_out,
          "expected extension count to match");
         check_ext_data(ext_out, ext_out2, nb_ext);
         if ((i & 63) == 0) {
            opus_extension_data ext_rep[MAX_EXTENSION_SIZE];
            opus_extension_data ext_sub_ref[MAX_NB_EXTENSIONS];
            opus_int32 nb_cnt = -1, nb_rep = orig_len;
            opus_int32 nb_sub_cnt = -1, nb_sub = orig_len, nb_sub_ref = 0;
            int fmin = fast_rand() % (nb_frames + 1);
            int fmax = fmin + (fast_rand() % (nb_frames - fmin + 1));
            int k;
            expect_true(opus_packet_extensions_parse_rep(payload, orig_len,
             NULL, &nb_cnt, nb_frames, 0, nb_frames) == OPUS_OK,
             "expected parse_rep count pass to succeed");
            result = opus_packet_extensions_parse_rep(payload, orig_len,
             ext_rep, &nb_rep, nb_frames, 0, nb_frames);
            expect_true(result == OPUS_OK && nb_rep == nb_cnt
             && nb_rep <= orig_len,
             "expected parse_rep to match count pass within orig_len");
            if (nb_rep > 0) {
               opus_int32 short_nb = nb_rep - 1;
               expect_true(opus_packet_extensions_parse_rep(payload, orig_len,
                ext_out2, &short_nb, nb_frames, 0, nb_frames)
                == OPUS_BUFFER_TOO_SMALL && short_nb == nb_rep - 1,
                "expected OPUS_BUFFER_TOO_SMALL without modifying nb_extensions");
            }
            len = opus_packet_extensions_generate(payload2, sizeof(payload2),
             ext_rep, nb_rep, nb_frames, 0);
            expect_true(len >= 0, "expected generate from parse_rep to succeed");
            result = opus_packet_extensions_count_ext(payload2, len,
             nb_frame_exts, nb_frames);
            expect_true(result == nb_ext, "expected count_ext to match");
            nb_ext_out = MAX_NB_EXTENSIONS;
            result = opus_packet_extensions_parse_ext(payload2, len, ext_out2,
             &nb_ext_out, nb_frame_exts, nb_frames);
            expect_true(result == OPUS_OK && nb_ext == nb_ext_out,
             "expected parse_ext from parse_rep to match");
            check_ext_data(ext_out, ext_out2, nb_ext);

            /* Also verify [fmin, fmax) subrange filtering in parse_rep. */
            expect_true(opus_packet_extensions_parse_rep(payload, orig_len,
             NULL, &nb_sub_cnt, nb_frames, fmin, fmax) == OPUS_OK,
             "expected subrange parse_rep count pass to succeed");
            result = opus_packet_extensions_parse_rep(payload, orig_len,
             ext_rep, &nb_sub, nb_frames, fmin, fmax);
            expect_true(result == OPUS_OK && nb_sub == nb_sub_cnt
             && nb_sub <= nb_rep,
             "expected subrange parse_rep to match count pass");
            if (nb_sub > 0) {
               opus_int32 rand_nb = fast_rand() % nb_sub;
               opus_int32 short_nb = rand_nb;
               expect_true(opus_packet_extensions_parse_rep(payload, orig_len,
                ext_out2, &short_nb, nb_frames, fmin, fmax)
                == OPUS_BUFFER_TOO_SMALL && short_nb == rand_nb,
                "expected subrange OPUS_BUFFER_TOO_SMALL");
            }
            for (k = 0; k < nb_sub; k++) {
               expect_true(ext_rep[k].frame >= fmin
                && ext_rep[k].repeat >= 1
                && ext_rep[k].frame + ext_rep[k].repeat <= fmax,
                "expected subrange entry within [fmin, fmax)");
               ext_rep[k].frame = (opus_int8)(ext_rep[k].frame - fmin);
            }
            for (k = 0; k < nb_ext; k++) {
               if (ext_out[k].frame >= fmin && ext_out[k].frame < fmax) {
                  ext_sub_ref[nb_sub_ref] = ext_out[k];
                  ext_sub_ref[nb_sub_ref++].frame =
                   (opus_int8)(ext_out[k].frame - fmin);
               }
            }
            len = opus_packet_extensions_generate(payload2, sizeof(payload2),
             ext_rep, nb_sub, fmax - fmin, 0);
            expect_true(len >= 0, "expected subrange generate to succeed");
            result = opus_packet_extensions_count_ext(payload2, len,
             nb_frame_exts, fmax - fmin);
            expect_true(result == nb_sub_ref,
             "expected subrange count_ext to match");
            nb_ext_out = MAX_NB_EXTENSIONS;
            result = opus_packet_extensions_parse_ext(payload2, len, ext_out2,
             &nb_ext_out, nb_frame_exts, fmax - fmin);
            expect_true(result == OPUS_OK && nb_sub_ref == nb_ext_out,
             "expected subrange parse_ext to match");
            check_ext_data(ext_sub_ref, ext_out2, nb_sub_ref);
         }
      } else if (result == OPUS_INVALID_PACKET && (i & 63) == 0) {
         opus_int32 nb_cnt = -17, nb_rep = MAX_NB_EXTENSIONS;
         expect_true(opus_packet_extensions_parse_rep(payload, len, NULL,
          &nb_cnt, nb_frames, 0, nb_frames) == OPUS_INVALID_PACKET
          && nb_cnt == -17
          && opus_packet_extensions_parse_rep(payload, len, ext_out,
          &nb_rep, nb_frames, 0, nb_frames) == OPUS_INVALID_PACKET
          && nb_rep == MAX_NB_EXTENSIONS,
          "expected parse_rep to reject invalid packet without modifying nb");
      }
   }
}

void test_opus_repacketizer_out_range_impl(void)
{
   OpusRepacketizer rp;
   unsigned char packet[1024];
   unsigned char packet_out[1024];
   opus_int16 size[48];
   const unsigned char *padding;
   opus_int32 padding_len;
   opus_extension_data ext_out[10];
   int i;
   opus_int32 nb_ext;
   int res, len;
   int first_count = 0, second_count = 0;
   static const opus_extension_data ext[] = {
      {(const unsigned char *)"abcdefg", 7, 33, 0, 1},
      {(const unsigned char *)"uvwxyz", 6, 100, 0, 1},
   };

   opus_repacketizer_init(&rp);

   memset(packet, 0, sizeof(packet));
   /* Hybrid Packet with 20 msec frames, Code 3 */
   packet[0] = (15 << 3) | 3;
   /* Code 3, padding bit set, 1 frame */
   packet[1] = 1 << 6 | 1;
   packet[2] = 0;
   packet[3] = 0;

   /* generate 2 extensions, id 33 and 100 */
   len = opus_packet_extensions_generate(&packet[4], sizeof(packet)-4, ext, 2,
    1, 0);
   /* update the padding length */
   packet[2] = len;

   /* concatenate 3 frames */
   res = opus_repacketizer_cat(&rp, packet, 4+len);
   /* for the middle frame, no padding, no extensions */
   packet[1] = 1;
   res = opus_repacketizer_cat(&rp, packet, 4);
   /* switch back to extensions for the last frame extensions */
   packet[1] = 1 << 6 | 1;
   res = opus_repacketizer_cat(&rp, packet, 4+len);

   expect_true(rp.nb_frames == 3, "Expected 3 frames");
   res = opus_repacketizer_out_range_impl(&rp,
      0, 3, /* begin, end */
      packet_out, /* unsigned char *data */
      sizeof(packet_out), /* opus_int32 maxlen */
      0, /*int self_delimited */
      0, /* int pad */
      NULL, /* const opus_extension_data *extensions */
      0 /* int nb_extensions */);
   expect_true(res > 0, "expected valid packet length");

   /* now verify that we have the expected extensions */
   res = opus_packet_parse_impl(packet_out, res, 0, NULL, NULL, size,
      NULL, NULL, &padding, &padding_len);
   nb_ext = 10;
   res = opus_packet_extensions_parse(padding, padding_len, ext_out, &nb_ext,
    3);
   expect_true(nb_ext == 4, "Expected 4 extensions");
   for (i = 0 ; i < nb_ext; i++)
   {
      if (ext_out[i].id == 33)
      {
         opus_test_assert(ext_out[i].len == ext[0].len);
         opus_test_assert(0 == memcmp(ext_out[i].data, ext[0].data, ext[0].len));
         first_count++;
      }
      else if (ext_out[i].id == 100)
      {
         opus_test_assert(ext_out[i].len == ext[1].len);
         opus_test_assert(0 == memcmp(ext_out[i].data, ext[1].data, ext[1].len));
         second_count++;
      }
      if (i < 2)
      {
         opus_test_assert(ext_out[i].frame == 0);
      } else {
         opus_test_assert(ext_out[i].frame == 2);
      }
   }
   opus_test_assert(first_count == 2);
   opus_test_assert(second_count == 2);
}

static opus_int32 build_code3_ext_packet(unsigned char *pkt, int nb_frames,
 const unsigned char *ext_bytes, opus_int32 ext_len)
{
   opus_int32 pos = 0;
   opus_int32 rem;
   /* CELT-only, FB, 2.5 ms, Code 3 */
   pkt[pos++] = 0xE3;
   if (ext_len > 0)
   {
      pkt[pos++] = (unsigned char)(0x40 | (nb_frames & 0x3F));
      rem = ext_len;
      while (rem >= 254)
      {
         pkt[pos++] = 255;
         rem -= 254;
      }
      pkt[pos++] = (unsigned char)rem;
      memcpy(pkt + pos, ext_bytes, ext_len);
      pos += ext_len;
   } else {
      pkt[pos++] = (unsigned char)(nb_frames & 0x3F);
   }
   return pos;
}

void test_repacketizer_out_range_slice(void)
{
   /* 3-frame packet with two RTEs in frame 0:
      RTE #1 (0x05): id=3(0B), id=32("ab"/"cd"/"ef"), id=4(0B)
      RTE #2 (0x04): id=5("X"/"Y"/"Z"), id=33("1"/"22"/"333" with L=0), id=6(0B) */
   static const unsigned char two_rte[] = {
      0x06, (32<<1)|1, 2, 'a', 'b', 0x08, 0x05, 2, 'c', 'd', 2, 'e', 'f',
      0x0B, 'X', (33<<1)|1, 1, '1', 0x0C, 0x04, 'Y', 2, '2', '2', 'Z', '3', '3', '3'
   };
   static const unsigned char f0_only[] = { 0x06 };
   unsigned char pkt[128], sub_pkt[128];
   OpusRepacketizer rp;
   opus_int32 pkt_len, out_len, pad_len;
   const unsigned char *pad_ptr;
   opus_int16 sz[48];
   int begin, end;

   pkt_len = build_code3_ext_packet(pkt, 3, two_rte, sizeof(two_rte));
   opus_repacketizer_init(&rp);
   expect_true(opus_repacketizer_cat(&rp, pkt, pkt_len) == OPUS_OK, "cat a");
   expect_true(opus_repacketizer_cat(&rp, pkt, pkt_len) == OPUS_OK, "cat b");

   for (begin = 0; begin < 6; begin++)
   {
      for (end = begin + 1; end <= 6; end++)
      {
         opus_int32 nb_frame_exts[48], nb_parsed = 48;
         opus_extension_data parsed[48];
         int f, nf = end - begin;

         out_len = opus_repacketizer_out_range(&rp, begin, end, sub_pkt,
          sizeof(sub_pkt));
         expect_true(out_len > 0, "expected out_range to succeed");
         expect_true(opus_packet_parse_impl(sub_pkt, out_len, 0, NULL, NULL,
          sz, NULL, NULL, &pad_ptr, &pad_len) == nf, "expected nf frames");
         expect_true(opus_packet_extensions_count_ext(pad_ptr, pad_len,
          nb_frame_exts, nf) == 6 * nf, "expected 6*nf extensions");
         expect_true(opus_packet_extensions_parse_ext(pad_ptr, pad_len,
          parsed, &nb_parsed, nb_frame_exts, nf) == OPUS_OK, "parse_ext");
         for (f = 0; f < nf; f++)
         {
            int s = (begin + f) % 3;
            expect_true(parsed[6*f].id == 3 && parsed[6*f].len == 0
             && parsed[6*f + 1].id == 32 && parsed[6*f + 1].len == 2
             && parsed[6*f + 1].data[0] == 'a' + 2*s
             && parsed[6*f + 2].id == 4 && parsed[6*f + 2].len == 0
             && parsed[6*f + 3].id == 5 && parsed[6*f + 3].data[0] == 'X' + s
             && parsed[6*f + 4].id == 33 && parsed[6*f + 4].len == s + 1
             && parsed[6*f + 4].data[0] == '1' + s
             && parsed[6*f + 5].id == 6 && parsed[6*f + 5].len == 0,
             "expected exact within-frame order across sliced RTEs");
         }
      }
   }

   /* Slicing out an extension-free trailing frame from a multi-frame packet
      with frame-0 extensions should emit no padding. */
   pkt_len = build_code3_ext_packet(pkt, 2, f0_only, sizeof(f0_only));
   opus_repacketizer_init(&rp);
   expect_true(opus_repacketizer_cat(&rp, pkt, pkt_len) == OPUS_OK, "cat 2f");
   out_len = opus_repacketizer_out_range(&rp, 1, 2, sub_pkt, sizeof(sub_pkt));
   expect_true(out_len == 1
    && opus_packet_parse_impl(sub_pkt, out_len, 0, NULL, NULL, sz, NULL, NULL,
    &pad_ptr, &pad_len) == 1 && pad_len == 0,
    "expected unpadded Code 0 packet when sliced range has no extensions");

   /* Subrange [2, 4) of a 5-frame packet whose frame-0 RTE contains 0x01
      padding, a 0-increment separator, 0-byte shorts, an L=0 long extension,
      and a trailing 1-byte short extension. */
   {
      static const unsigned char f0_pad_sep_rte[] = {
         0x01, 0x03, 0x00, 0x06, (32<<1)|1, 2, 'a', 'b', 0x09, 'X', 0x08, 0x04,
         2, 'c', 'd', 'Y',
         2, 'e', 'f', 'Z',
         2, 'g', 'h', 'W',
         'q', 'r', 's', 'V'
      };
      opus_int32 nb_frame_exts[48], nb_parsed = 16;
      opus_extension_data parsed[16];
      pkt_len = build_code3_ext_packet(pkt, 5, f0_pad_sep_rte,
       sizeof(f0_pad_sep_rte));
      opus_repacketizer_init(&rp);
      expect_true(opus_repacketizer_cat(&rp, pkt, pkt_len) == OPUS_OK,
       "cat 5f pad+sep RTE");
      out_len = opus_repacketizer_out_range(&rp, 2, 4, sub_pkt,
       sizeof(sub_pkt));
      expect_true(out_len > 0
       && opus_packet_parse_impl(sub_pkt, out_len, 0, NULL, NULL, sz, NULL,
       NULL, &pad_ptr, &pad_len) == 2
       && opus_packet_extensions_count_ext(pad_ptr, pad_len, nb_frame_exts, 2)
       == 8
       && opus_packet_extensions_parse_ext(pad_ptr, pad_len, parsed,
       &nb_parsed, nb_frame_exts, 2) == OPUS_OK
       && parsed[0].id == 3 && parsed[0].len == 0
       && parsed[1].id == 32 && parsed[1].len == 2
       && memcmp(parsed[1].data, "ef", 2) == 0
       && parsed[2].id == 4 && parsed[2].len == 1 && parsed[2].data[0] == 'Z'
       && parsed[3].id == 4 && parsed[3].len == 0
       && parsed[4].id == 3 && parsed[4].len == 0
       && parsed[5].id == 32 && parsed[5].len == 2
       && memcmp(parsed[5].data, "gh", 2) == 0
       && parsed[6].id == 4 && parsed[6].len == 1 && parsed[6].data[0] == 'W'
       && parsed[7].id == 4 && parsed[7].len == 0,
       "expected exact [2, 4) slice of padded/0-sep L=0 RTE");
   }
}

void test_extension_iterator_frame_max(void)
{
   static const unsigned char l0_rte_then_ext[] = { 0x06, 0x04, (124<<1)|0 };
   static const unsigned char last_f_l0_rte_junk[] = {
      0x02, 0x06, 0x04, (124<<1)|0
   };
   OpusExtensionIterator iter;
   opus_extension_data fext;

   /* frame_max must stop when an L=0 RTE advances curr_frame to frame_max */
   opus_extension_iterator_init(&iter, l0_rte_then_ext, sizeof(l0_rte_then_ext),
    3);
   opus_extension_iterator_set_frame_max(&iter, 1);
   expect_true(opus_extension_iterator_find(&iter, &fext, 124) == 0,
    "expected frame_max=1 to stop after L=0 RTE");

   /* Setting frame_max > nb_frames must still clamp to nb_frames and ignore
      trailing junk after an L=0 RTE in the last frame. */
   opus_extension_iterator_init(&iter, last_f_l0_rte_junk,
    sizeof(last_f_l0_rte_junk), 2);
   opus_extension_iterator_set_frame_max(&iter, 5);
   expect_true(opus_extension_iterator_find(&iter, &fext, 124) == 0,
    "expected frame_max > nb_frames to clamp to nb_frames");
}

void test_extensions_rep_and_pathological(void)
{
   static const opus_extension_data bad_rep[] = {
      {NULL, 0, 4, 0, 0},
      {NULL, 0, 4, 0, -1},
      {NULL, 0, 4, 2, 3},
      {(const unsigned char *)"A", 1, 4, 0, 2},
      {NULL, 0, 33, 0, 2}
   };
   static const unsigned char empty_f1_rte[] = { 0x06, 0x02, 0x05 };
#ifdef NONTHREADSAFE_PSEUDOSTACK
   const opus_int32 EXT_SZ = 2000;
# ifdef ENABLE_VALGRIND
   const opus_int32 max_cap = IMIN((opus_int32)32768,
    (opus_int32)(GLOBAL_STACK_SIZE / (2 * sizeof(opus_extension_data))));
# else
   const opus_int32 max_cap = IMIN((opus_int32)32768,
    (opus_int32)(GLOBAL_STACK_SIZE / sizeof(opus_extension_data)));
# endif
#else
   const opus_int32 EXT_SZ = 16000;
   const opus_int32 max_cap = 32768;
#endif
   const opus_int32 slice_exts = IMIN((opus_int32)20000, (max_cap * 3) / 4);
   unsigned char *ext_buf = (unsigned char *)malloc(40256);
   unsigned char *pkt1 = (unsigned char *)malloc(40512);
   unsigned char pkt2[8];
   unsigned char *out_small = (unsigned char *)malloc(65536);
   unsigned char *out_large = (unsigned char *)malloc(1500000);
   OpusRepacketizer rp;
   opus_int32 len1, len2, res, pad_len, nb_rep, pos;
   const unsigned char *pad_ptr;
   opus_int16 sz[48];
   int k;
   OpusExtensionIterator iter;
   opus_extension_data fext;

   expect_true(sizeof(opus_extension_data) <= sizeof(const unsigned char *) + 8,
    "expected compact opus_extension_data layout");

   for (k = 0; k < 5; k++)
   {
      expect_true(opus_packet_extensions_generate(pkt1, 128, &bad_rep[k], 1, 4,
       0) == OPUS_BAD_ARG, "expected OPUS_BAD_ARG for invalid repeat");
   }
   nb_rep = -1;
   expect_true(opus_packet_extensions_parse_rep(pkt1, 0, &fext, &nb_rep, 1,
    0, 1) == OPUS_BAD_ARG
    && opus_packet_extensions_parse_rep(NULL, 1, NULL, &nb_rep, 1, 0, 1)
    == OPUS_BAD_ARG
    && opus_packet_extensions_parse_rep(pkt1, 0, NULL, &nb_rep, 49, 0, 49)
    == OPUS_BAD_ARG
    && opus_packet_extensions_parse_rep(pkt1, 0, NULL, &nb_rep, 4, -1, 2)
    == OPUS_BAD_ARG
    && opus_packet_extensions_parse_rep(pkt1, 0, NULL, &nb_rep, 4, 3, 2)
    == OPUS_BAD_ARG
    && opus_packet_extensions_parse_rep(pkt1, 0, NULL, &nb_rep, 4, 0, 5)
    == OPUS_BAD_ARG, "expected OPUS_BAD_ARG from parse_rep");

   /* Empty RTE after frame separator must not repeat prior frame's ext */
   nb_rep = 1;
   expect_true(opus_packet_extensions_parse_rep(empty_f1_rte,
    sizeof(empty_f1_rte), &fext, &nb_rep, 4, 0, 4) == OPUS_OK
    && nb_rep == 1 && fext.id == 3 && fext.frame == 0 && fext.repeat == 1,
    "expected empty frame-1 RTE not to affect frame-0 extension");

   /* Out-of-frame-order repeat > 1 + long extension in generate() */
   {
      static const opus_extension_data ooo_rep[] = {
         {NULL, 0, 4, 1, 2},
         {(const unsigned char *)"ABCD", 4, 32, 2, 1},
         {NULL, 0, 4, 0, 1}
      };
      opus_int32 nb_frame_exts[48], nb_p = 8;
      opus_extension_data parsed[8];
      len1 = opus_packet_extensions_generate(pkt1, 32, ooo_rep, 3, 3, 0);
      expect_true(len1 == 8
       && opus_packet_extensions_count_ext(pkt1, len1, nb_frame_exts, 3) == 4
       && opus_packet_extensions_parse_ext(pkt1, len1, parsed, &nb_p,
       nb_frame_exts, 3) == OPUS_OK
       && parsed[0].id == 4 && parsed[0].frame == 0
       && parsed[1].id == 4 && parsed[1].frame == 1
       && parsed[2].id == 4 && parsed[2].frame == 2
       && parsed[3].id == 32 && parsed[3].frame == 2 && parsed[3].len == 4
       && memcmp(parsed[3].data, "ABCD", 4) == 0,
       "expected out-of-frame-order collapsed generate() to round-trip");
   }

   /* Subrange W=1 with two RTEs in frame 0 (0B short + 0B long + 1B short)
      and L=0 0B RTE advancing curr_frame into frame_min without separator */
   {
      static const unsigned char two_f0_w1[] = {
         0x06, (32<<1)|1, 0, 0x07, 'a', 0x05, 0, 'b', 0, 'c', 0, 'd',
         0x08, 0x09, '1', 0x04, '2', '3', '4'
      };
      static const unsigned char l0_into_fmin[] = {
         0x06, 0x04, 0x09, 'Q', 0x04, 'R'
      };
      opus_extension_data rep_out[8];
      nb_rep = 8;
      expect_true(opus_packet_extensions_parse_rep(two_f0_w1,
       sizeof(two_f0_w1), rep_out, &nb_rep, 4, 2, 3) == OPUS_OK
       && nb_rep == 5
       && rep_out[0].id == 3 && rep_out[0].len == 0 && rep_out[0].repeat == 1
       && rep_out[1].id == 32 && rep_out[1].len == 0 && rep_out[1].repeat == 1
       && rep_out[2].id == 3 && rep_out[2].len == 1 && rep_out[2].data[0] == 'c'
       && rep_out[3].id == 4 && rep_out[3].len == 0 && rep_out[3].repeat == 1
       && rep_out[4].id == 4 && rep_out[4].len == 1 && rep_out[4].data[0] == '3',
       "expected exact W=1 subrange across two frame-0 RTE groups");
      nb_rep = 8;
      expect_true(opus_packet_extensions_parse_rep(l0_into_fmin,
       sizeof(l0_into_fmin), rep_out, &nb_rep, 3, 1, 3) == OPUS_OK
       && nb_rep == 3
       && rep_out[0].id == 3 && rep_out[0].frame == 1 && rep_out[0].repeat == 2
       && rep_out[1].id == 4 && rep_out[1].frame == 1 && rep_out[1].data[0] == 'Q'
       && rep_out[2].id == 4 && rep_out[2].frame == 2 && rep_out[2].data[0] == 'R',
       "expected L=0 0B RTE at frame_min-1 to advance cleanly into frame_min");
   }

   /* 32768 (or pseudostack) collapsed extension cap -> OPUS_ALLOC_FAIL */
   len2 = build_code3_ext_packet(pkt2, 1, NULL, 0);
   opus_repacketizer_init(&rp);
   expect_true(opus_repacketizer_cat(&rp, pkt2, len2) == OPUS_OK, "cat 1 cap");
   fext.data = NULL;
   fext.len = 0;
   fext.id = 3;
   fext.frame = 0;
   fext.repeat = 1;
   expect_true(opus_repacketizer_out_range_impl(&rp, 0, 1, out_small, 65536,
    0, 0, NULL, 32769) == OPUS_BAD_ARG,
    "expected OPUS_BAD_ARG for NULL extensions before cap check");
   expect_true(opus_repacketizer_out_range_impl(&rp, 0, 1, out_small, 65536,
    0, 0, &fext, 32769) == OPUS_ALLOC_FAIL
    && opus_repacketizer_out_range_impl(&rp, 0, 1, out_small, 65536,
    0, 0, &fext, max_cap + 1) == OPUS_ALLOC_FAIL,
    "expected OPUS_ALLOC_FAIL when nb_extensions exceeds cap");

   memset(ext_buf, 0x06, 32769);
   ext_buf[32769] = 0x04;
   len1 = build_code3_ext_packet(pkt1, 48, ext_buf, 32770);
   opus_repacketizer_init(&rp);
   expect_true(opus_repacketizer_cat(&rp, pkt1, len1) == OPUS_OK, "cat 32769");
   expect_true(opus_repacketizer_out(&rp, out_small, 65536) == OPUS_ALLOC_FAIL,
    "expected OPUS_ALLOC_FAIL for 32769 collapsed extensions");

   memset(ext_buf, 0x06, max_cap + 1);
   ext_buf[max_cap + 1] = 0x04;
   len1 = build_code3_ext_packet(pkt1, 48, ext_buf, max_cap + 2);
   opus_repacketizer_init(&rp);
   expect_true(opus_repacketizer_cat(&rp, pkt1, len1) == OPUS_OK, "cat cap+1");
   expect_true(opus_repacketizer_out(&rp, out_small, 65536) == OPUS_ALLOC_FAIL,
    "expected OPUS_ALLOC_FAIL at max_cap + 1 collapsed extensions");

   ext_buf[max_cap] = 0x04;
   len1 = build_code3_ext_packet(pkt1, 48, ext_buf, max_cap + 1);
   opus_repacketizer_init(&rp);
   expect_true(opus_repacketizer_cat(&rp, pkt1, len1) == OPUS_OK, "cat cap");
   res = opus_repacketizer_out(&rp, out_small, 65536);
   expect_true(res == len1 && memcmp(pkt1, out_small, len1) == 0,
    "expected exact round-trip at collapsed extension cap");

   /* 2-frame packet with slice_exts in frame 0 and slice_exts in frame 1:
      full [0, 2) exceeds max_cap, while sliced [0, 1) and [1, 2) succeed. */
   memset(ext_buf, 0x06, slice_exts);
   ext_buf[slice_exts] = 0x02;
   memset(ext_buf + slice_exts + 1, 0x06, slice_exts);
   len1 = build_code3_ext_packet(pkt1, 2, ext_buf, 2 * slice_exts + 1);
   opus_repacketizer_init(&rp);
   expect_true(opus_repacketizer_cat(&rp, pkt1, len1) == OPUS_OK, "cat 2x");
   expect_true(opus_repacketizer_out(&rp, out_small, 65536) == OPUS_ALLOC_FAIL,
    "expected OPUS_ALLOC_FAIL when unsliced packet exceeds cap");
   expect_true(opus_repacketizer_out_range(&rp, 0, 1, out_small, 65536) > 0
    && opus_repacketizer_out_range(&rp, 1, 2, out_small, 65536) > 0,
    "expected sliced single frames within cap to succeed");

   /* 48 concatenated 25 kB zero-padded packets (only 48 collapsed exts total,
      fits even in NONTHREADSAFE_PSEUDOSTACK). */
   ext_buf[0] = 0x06;
   memset(ext_buf + 1, 0, 24999);
   len1 = build_code3_ext_packet(pkt1, 1, ext_buf, 25000);
   opus_repacketizer_init(&rp);
   for (k = 0; k < 48; k++)
      expect_true(opus_repacketizer_cat(&rp, pkt1, len1) == OPUS_OK, "cat zp");
   res = opus_repacketizer_out(&rp, out_small, 65536);
   expect_true(res == 5, "expected 48x 0x06 + zero-padding to collapse to 5B");

   memset(ext_buf, 0x06, EXT_SZ - 1);
   ext_buf[EXT_SZ - 1] = 0x04;

   /* 48-frame 0-byte RTE round-trip + iterator_find */
   len1 = build_code3_ext_packet(pkt1, 48, ext_buf, EXT_SZ);
   opus_repacketizer_init(&rp);
   expect_true(opus_repacketizer_cat(&rp, pkt1, len1) == OPUS_OK, "cat 48");
   res = opus_repacketizer_out(&rp, out_small, 65536);
   expect_true(res == len1 && memcmp(pkt1, out_small, len1) == 0,
    "expected exact 16 kB RTE packet round-trip");
   opus_extension_iterator_init(&iter, ext_buf, EXT_SZ, 48);
   expect_true(opus_extension_iterator_find(&iter, &fext, 4) == 0,
    "expected iterator_find(4) == 0 on 0-byte RTE");

   /* Two 24-frame RTE packets merged -> collapses to single 48-frame RTE */
   len1 = build_code3_ext_packet(pkt1, 24, ext_buf, EXT_SZ);
   opus_repacketizer_init(&rp);
   expect_true(opus_repacketizer_cat(&rp, pkt1, len1) == OPUS_OK, "cat 24a");
   expect_true(opus_repacketizer_cat(&rp, pkt1, len1) == OPUS_OK, "cat 24b");
   res = opus_repacketizer_out(&rp, out_small, 65536);
   expect_true(res == build_code3_ext_packet(pkt1, 48, ext_buf, EXT_SZ),
    "expected merged 24f+24f RTE packet to collapse");

   /* 47-frame 0-byte RTE + 1 empty frame */
   len1 = build_code3_ext_packet(pkt1, 47, ext_buf, EXT_SZ);
   len2 = build_code3_ext_packet(pkt2, 1, NULL, 0);
   opus_repacketizer_init(&rp);
   expect_true(opus_repacketizer_cat(&rp, pkt1, len1) == OPUS_OK, "cat 47");
   expect_true(opus_repacketizer_cat(&rp, pkt2, len2) == OPUS_OK, "cat 1");
   expect_true(opus_repacketizer_out(&rp, out_small, 4000)
    == OPUS_BUFFER_TOO_SMALL, "expected BUFFER_TOO_SMALL on 4 kB buffer");
   expect_true(opus_repacketizer_out(&rp, out_small, 65536)
    == OPUS_BUFFER_TOO_SMALL, "expected BUFFER_TOO_SMALL on 64 kB buffer");
   res = opus_repacketizer_out(&rp, out_large, 1500000);
   expect_true(res > 47 * (EXT_SZ - 1), "expected expanded output packet");
   expect_true(opus_packet_parse_impl(out_large, res, 0, NULL, NULL, sz,
    NULL, NULL, &pad_ptr, &pad_len) == 48
    && opus_packet_extensions_count(pad_ptr, pad_len, 48)
    == 47 * (EXT_SZ - 1), "expected 47*(EXT_SZ-1) unrepeated extensions");

   /* 1 empty frame + 47-frame 0-byte RTE -> stays ~16 kB via frame-1 RTE */
   opus_repacketizer_init(&rp);
   expect_true(opus_repacketizer_cat(&rp, pkt2, len2) == OPUS_OK, "cat 1f");
   expect_true(opus_repacketizer_cat(&rp, pkt1, len1) == OPUS_OK, "cat 47f");
   res = opus_repacketizer_out(&rp, out_small, 65536);
   expect_true(res == len1 + 1 || res == len1 + 2,
    "expected 1f+47f RTE packet to stay compact");
   expect_true(opus_packet_parse_impl(out_small, res, 0, NULL, NULL, sz,
    NULL, NULL, &pad_ptr, &pad_len) == 48
    && opus_packet_extensions_count(pad_ptr, pad_len, 48)
    == 47 * (EXT_SZ - 1), "expected 47*(EXT_SZ-1) instances in 1f+47f");

   /* Leading 1-byte RTE followed by alternating [0x06, 0x05] -> single RTE */
   pos = 0;
   ext_buf[pos++] = 0x07;
   ext_buf[pos++] = 'X';
   ext_buf[pos++] = 0x05;
   memset(ext_buf + pos, 'X', 47);
   pos += 47;
   for (k = 0; k < (EXT_SZ - pos) / 2; k++)
   {
      ext_buf[pos + 2*k] = 0x06;
      ext_buf[pos + 2*k + 1] = 0x05;
   }
   len1 = build_code3_ext_packet(pkt1, 48, ext_buf,
    pos + 2 * ((EXT_SZ - pos) / 2));
   opus_repacketizer_init(&rp);
   expect_true(opus_repacketizer_cat(&rp, pkt1, len1) == OPUS_OK, "cat alt");
   res = opus_repacketizer_out(&rp, out_small, 65536);
   expect_true(res > 0 && res < EXT_SZ, "expected merged single RTE");

   /* Mixed 1B + 0B + long RTE + trailing frame-47 extension */
   pos = 0;
   ext_buf[pos++] = 0x07;
   ext_buf[pos++] = 'A';
   memset(ext_buf + pos, 0x06, EXT_SZ - 205);
   pos += EXT_SZ - 205;
   ext_buf[pos++] = (33 << 1) | 1;
   ext_buf[pos++] = 1;
   ext_buf[pos++] = 'Z';
   ext_buf[pos++] = 0x05;
   for (k = 1; k <= 47; k++)
   {
      ext_buf[pos++] = 'A';
      ext_buf[pos++] = 1;
      ext_buf[pos++] = 'Z';
   }
   ext_buf[pos++] = 0x03;
   ext_buf[pos++] = 47;
   ext_buf[pos++] = (124 << 1) | 0;
   len1 = build_code3_ext_packet(pkt1, 48, ext_buf, pos);
   opus_repacketizer_init(&rp);
   expect_true(opus_repacketizer_cat(&rp, pkt1, len1) == OPUS_OK, "cat mix");
   res = opus_repacketizer_out(&rp, out_small, 65536);
   expect_true(res > 0 && res <= len1, "expected compact mixed RTE output");
   {
      opus_int16 *pcm = (opus_int16 *)malloc(sizeof(opus_int16) * 5760);
      int err;
      OpusDecoder *dec = opus_decoder_create(48000, 1, &err);
      expect_true(err == OPUS_OK
       && opus_decode(dec, pkt1, len1, pcm, 5760, 0) == 5760,
       "expected opus_decode to succeed on 16 kB mixed RTE + late QEXT");
      opus_decoder_destroy(dec);
#ifdef ENABLE_QEXT
      {
         opus_int16 in_pcm[1920], pcm1[1920], pcm2[1920], pcm3[1920];
         unsigned char f0_pkt[1280], f1_pkt[1280], cat_pkt[2600];
         opus_int32 len_f0, len_f1, len_cat;
         opus_uint32 rng1 = 0, rng2 = 1;
         OpusEncoder *enc = opus_encoder_create(48000, 1,
          OPUS_APPLICATION_RESTRICTED_LOWDELAY, &err);
         OpusDecoder *dec1 = opus_decoder_create(48000, 1, &err);
         OpusDecoder *dec2 = opus_decoder_create(48000, 1, &err);
         OpusDecoder *dec3 = opus_decoder_create(48000, 1, &err);
         for (k = 0; k < 1920; k++)
            in_pcm[k] = (opus_int16)(((k * 1031) & 0x3FFF) - 0x2000);
         opus_encoder_ctl(enc, OPUS_SET_BITRATE(256000));
         opus_encoder_ctl(enc, OPUS_SET_QEXT(0));
         len_f0 = opus_encode(enc, in_pcm, 960, f0_pkt, sizeof(f0_pkt));
         opus_encoder_ctl(enc, OPUS_SET_QEXT(1));
         len_f1 = opus_encode(enc, in_pcm + 960, 960, f1_pkt, sizeof(f1_pkt));
         expect_true(len_f0 > 0 && len_f1 > 0, "expected QEXT encode success");
         opus_repacketizer_init(&rp);
         expect_true(opus_repacketizer_cat(&rp, f0_pkt, len_f0) == OPUS_OK
          && opus_repacketizer_cat(&rp, f1_pkt, len_f1) == OPUS_OK, "cat qext");
         len_cat = opus_repacketizer_out(&rp, cat_pkt, sizeof(cat_pkt));
         expect_true(len_cat > 0, "expected repacketized QEXT packet");
         expect_true(opus_decode(dec1, f0_pkt, len_f0, pcm1, 960, 0) == 960
          && opus_decode(dec1, f1_pkt, len_f1, pcm1 + 960, 960, 0) == 960,
          "expected single-frame QEXT decode to succeed");
         opus_decoder_ctl(dec1, OPUS_GET_FINAL_RANGE(&rng1));
         expect_true(opus_decode(dec2, cat_pkt, len_cat, pcm2, 1920, 0) == 1920,
          "expected multi-frame QEXT decode to succeed");
         opus_decoder_ctl(dec2, OPUS_GET_FINAL_RANGE(&rng2));
         opus_decoder_ctl(dec3, OPUS_SET_IGNORE_EXTENSIONS(1));
         expect_true(opus_decode(dec3, cat_pkt, len_cat, pcm3, 1920, 0) == 1920,
          "expected ignore-extensions decode to succeed");
         expect_true(memcmp(pcm1, pcm2, sizeof(pcm1)) == 0 && rng1 == rng2,
          "expected multi-frame late QEXT decode to match single-frame decode");
#ifndef FUZZING
         expect_true((f1_pkt[0] & 0x03) == 3 && (f1_pkt[1] & 0x40) != 0,
          "expected QEXT extension in frame 1");
#endif
         expect_true(memcmp(pcm1, pcm3, 960 * sizeof(opus_int16)) == 0
          && (((f1_pkt[0] & 0x03) != 3 || (f1_pkt[1] & 0x40) == 0)
          || memcmp(pcm1 + 960, pcm3 + 960, 960 * sizeof(opus_int16)) != 0),
          "expected late QEXT payload to affect only frame 1");
         opus_encoder_destroy(enc);
         opus_decoder_destroy(dec1);
         opus_decoder_destroy(dec2);
         opus_decoder_destroy(dec3);
      }
#endif
      free(pcm);
   }

   free(ext_buf);
   free(pkt1);
   free(out_small);
   free(out_large);
}

int main(int argc, char **argv)
{
   int env_used;
   char *env_seed;
   env_used=0;
   env_seed=getenv("SEED");
   if(argc>1)iseed=atoi(argv[1]);
   else if(env_seed)
   {
      iseed=atoi(env_seed);
      env_used=1;
   }
   else iseed=(opus_uint32)time(NULL)^(((opus_uint32)getpid()&65535)<<16);
   Rw=Rz=iseed;

   fprintf(stderr,"Testing extensions. Random seed: %u (%.4X)\n", iseed, fast_rand() % 65535);
   if(env_used)fprintf(stderr,"  Random seed set from the environment (SEED=%s).\n", env_seed);

   test_extensions_generate_success();
   test_extensions_generate_zero();
   test_extensions_generate_no_padding();
   test_extensions_generate_fail();
   test_extensions_parse_success();
   test_extensions_parse_zero();
   test_extensions_parse_fail();
   test_extensions_repeating();
   test_random_extensions_parse();
   test_opus_repacketizer_out_range_impl();
   test_repacketizer_out_range_slice();
   test_extension_iterator_frame_max();
   test_extensions_rep_and_pathological();
   fprintf(stderr,"Tests completed successfully.\n");
   return 0;
}
