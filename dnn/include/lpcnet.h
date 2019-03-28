/* Copyright (c) 2018 Mozilla */
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

#ifndef _LPCNET_H_
#define _LPCNET_H_

#ifndef LPCNET_EXPORT
# if defined(WIN32)
#  if defined(LPCNET_BUILD) && defined(DLL_EXPORT)
#   define LPCNET_EXPORT __declspec(dllexport)
#  else
#   define LPCNET_EXPORT
#  endif
# elif defined(__GNUC__) && defined(LPCNET_BUILD)
#  define LPCNET_EXPORT __attribute__ ((visibility ("default")))
# else
#  define LPCNET_EXPORT
# endif
#endif


#define NB_FEATURES 38
#define NB_TOTAL_FEATURES 55

/** Number of bytes in a compressed packet. */
#define LPCNET_COMPRESSED_SIZE 8
/** Number of audio samples in a packet. */
#define LPCNET_PACKET_SAMPLES (4*160)
/** Number of audio samples in a feature frame (not for encoding/decoding). */
#define LPCNET_FRAME_SIZE (160)

typedef struct LPCNetState LPCNetState;

typedef struct LPCNetDecState LPCNetDecState;

typedef struct LPCNetEncState LPCNetEncState;


/** Gets the size of an <code>LPCNetDecState</code> structure.
  * @returns The size in bytes.
  */
LPCNET_EXPORT int lpcnet_decoder_get_size();

/** Initializes a previously allocated decoder state
  * The memory pointed to by st must be at least the size returned by lpcnet_decoder_get_size().
  * This is intended for applications which use their own allocator instead of malloc.
  * @see lpcnet_decoder_create(),lpcnet_decoder_get_size()
  * @param [in] st <tt>LPCNetDecState*</tt>: Decoder state
  * @retval 0 Success
  */
LPCNET_EXPORT int lpcnet_decoder_init(LPCNetDecState *st);

/** Allocates and initializes a decoder state.
  *  @returns The newly created state
  */
LPCNET_EXPORT LPCNetDecState *lpcnet_decoder_create();

/** Frees an <code>LPCNetDecState</code> allocated by lpcnet_decoder_create().
  * @param[in] st <tt>LPCNetDecState*</tt>: State to be freed.
  */
LPCNET_EXPORT void lpcnet_decoder_destroy(LPCNetDecState *st);

/** Decodes a packet of LPCNET_COMPRESSED_SIZE bytes (currently 8) into LPCNET_PACKET_SAMPLES samples (currently 640).
  * @param [in] st <tt>LPCNetDecState*</tt>: Decoder state
  * @param [in] buf <tt>const unsigned char *</tt>: Compressed packet
  * @param [out] pcm <tt>short **</tt>: Decoded audio
  * @retval 0 Success
  */
LPCNET_EXPORT int lpcnet_decode(LPCNetDecState *st, const unsigned char *buf, short *pcm);



/** Gets the size of an <code>LPCNetEncState</code> structure.
  * @returns The size in bytes.
  */
LPCNET_EXPORT int lpcnet_encoder_get_size();

/** Initializes a previously allocated encoder state
  * The memory pointed to by st must be at least the size returned by lpcnet_encoder_get_size().
  * This is intended for applications which use their own allocator instead of malloc.
  * @see lpcnet_encoder_create(),lpcnet_encoder_get_size()
  * @param [in] st <tt>LPCNetEncState*</tt>: Encoder state
  * @retval 0 Success
  */
LPCNET_EXPORT int lpcnet_encoder_init(LPCNetEncState *st);

/** Allocates and initializes an encoder state.
  *  @returns The newly created state
  */
LPCNET_EXPORT LPCNetEncState *lpcnet_encoder_create();

/** Frees an <code>LPCNetEncState</code> allocated by lpcnet_encoder_create().
  * @param[in] st <tt>LPCNetEncState*</tt>: State to be freed.
  */
LPCNET_EXPORT void lpcnet_encoder_destroy(LPCNetEncState *st);

/** Encodes LPCNET_PACKET_SAMPLES speech samples (currently 640) into a packet of LPCNET_COMPRESSED_SIZE bytes (currently 8).
  * @param [in] st <tt>LPCNetDecState*</tt>: Encoder state
  * @param [in] pcm <tt>short **</tt>: Input speech to be encoded
  * @param [out] buf <tt>const unsigned char *</tt>: Compressed packet
  * @retval 0 Success
  */
LPCNET_EXPORT int lpcnet_encode(LPCNetEncState *st, const short *pcm, unsigned char *buf);

/** Compute features on LPCNET_PACKET_SAMPLES speech samples (currently 640) and output features for 4 10-ms frames at once.
  * @param [in] st <tt>LPCNetDecState*</tt>: Encoder state
  * @param [in] pcm <tt>short **</tt>: Input speech to be analyzed
  * @param [out] features <tt>float[4][NB_TOTAL_FEATURES]</tt>: Four feature vectors
  * @retval 0 Success
  */
LPCNET_EXPORT int lpcnet_compute_features(LPCNetEncState *st, const short *pcm, float features[4][NB_TOTAL_FEATURES]);


/** Gets the size of an <code>LPCNetState</code> structure.
  * @returns The size in bytes.
  */
LPCNET_EXPORT int lpcnet_get_size();

/** Initializes a previously allocated synthesis state
  * The memory pointed to by st must be at least the size returned by lpcnet_get_size().
  * This is intended for applications which use their own allocator instead of malloc.
  * @see lpcnet_create(),lpcnet_get_size()
  * @param [in] st <tt>LPCNetState*</tt>: Synthesis state
  * @retval 0 Success
  */
LPCNET_EXPORT int lpcnet_init(LPCNetState *st);

/** Allocates and initializes a synthesis state.
  *  @returns The newly created state
  */
LPCNET_EXPORT LPCNetState *lpcnet_create();

/** Frees an <code>LPCNetState</code> allocated by lpcnet_create().
  * @param[in] st <tt>LPCNetState*</tt>: State to be freed.
  */
LPCNET_EXPORT void lpcnet_destroy(LPCNetState *st);

/** Synthesizes speech from an LPCNet feature vector.
  * @param [in] st <tt>LPCNetState*</tt>: Synthesis state
  * @param [in] features <tt>const float *</tt>: Compressed packet
  * @param [out] output <tt>short **</tt>: Synthesized speech
  * @param [in] N <tt>int</tt>: Number of samples to generate
  * @retval 0 Success
  */
LPCNET_EXPORT void lpcnet_synthesize(LPCNetState *st, const float *features, short *output, int N);

#endif
