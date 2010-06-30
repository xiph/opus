/***********************************************************************
Copyright (c) 2006-2010, Skype Limited. All rights reserved. 
Redistribution and use in source and binary forms, with or without 
modification, (subject to the limitations in the disclaimer below) 
are permitted provided that the following conditions are met:
- Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright 
notice, this list of conditions and the following disclaimer in the 
documentation and/or other materials provided with the distribution.
- Neither the name of Skype Limited, nor the names of specific 
contributors, may be used to endorse or promote products derived from 
this software without specific prior written permission.
NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED 
BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND 
CONTRIBUTORS ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND 
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF 
USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON 
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
***********************************************************************/

#include "SKP_Silk_main.h"

#define MAX_SIZE 10000

/* Range encoder for one symbol */
void SKP_Silk_range_encoder(
    SKP_Silk_range_coder_state      *psRC,              /* I/O  compressor data structure                   */
    const SKP_int                   data,               /* I    uncompressed data                           */
    const SKP_uint16                prob[]              /* I    cumulative density functions                */
)
{
    SKP_uint32 low_Q16, high_Q16;

    if( psRC->error ) {
        return;
    }
    low_Q16  = prob[ data ];
    high_Q16 = prob[ data + 1 ];
    
#ifdef SAVE_ALL_INTERNAL_DATA
    DEBUG_STORE_DATA( enc_l.dat, &low_Q16,  sizeof(SKP_uint32) );
    DEBUG_STORE_DATA( enc_h.dat, &high_Q16, sizeof(SKP_uint32) );
    DEBUG_STORE_DATA( enc.dat,   &data,     sizeof(SKP_int) );
#endif

    if( prob[ 2 ] == 65535 ) {
        /* Instead of detection, we could add a separate function and call when we know that input is a bit */
        ec_enc_bit_prob( &psRC->range_enc_celt_state, data, 65536 - prob[ 1 ] );
    } else {
        ec_encode_bin( &psRC->range_enc_celt_state, low_Q16, high_Q16, 16 );
    }
}

/* Range encoder for one symbol, with uniform PDF*/
void SKP_Silk_range_encode_uniform(
    SKP_Silk_range_coder_state      *psRC,              /* I/O  compressor data structure                   */
    const SKP_int                   data,               /* I    uncompressed data                           */
    const SKP_int                   N                   /* I    number of possible outcomes                 */
)
{
    SKP_int i;
    SKP_uint16 delta, prob[ MAX_SIZE + 1 ];

    SKP_assert( N < MAX_SIZE );

    delta = ( SKP_uint16 )SKP_DIV32_16( 65535, N );
    prob[ 0 ] = 0;
    for( i = 0; i < N - 1; i++ ) {
        prob[ i + 1 ] = prob[ i ] + delta;
    }
    prob[ N ] = 65535;

    SKP_Silk_range_encoder( psRC, data, prob );
}

/* Range encoder for multiple symbols */
void SKP_Silk_range_encoder_multi(
    SKP_Silk_range_coder_state      *psRC,              /* I/O  compressor data structure                   */
    const SKP_int                   data[],             /* I    uncompressed data    [nSymbols]             */
    const SKP_uint16 * const        prob[],             /* I    cumulative density functions                */
    const SKP_int                   nSymbols            /* I    number of data symbols                      */
)
{
    SKP_int k;
    for( k = 0; k < nSymbols; k++ ) {
        SKP_Silk_range_encoder( psRC, data[ k ], prob[ k ] );
    }
}

/* Range decoder for one symbol */
void SKP_Silk_range_decoder(
    SKP_int                         data[],             /* O    uncompressed data                           */
    SKP_Silk_range_coder_state      *psRC,              /* I/O  compressor data structure                   */
    const SKP_uint16                prob[],             /* I    cumulative density function                 */
    SKP_int                         probIx              /* I    initial (middle) entry of cdf               */
)
{
    SKP_uint32 low_Q16, high_Q16;

    SKP_uint32 low_Q16_returned;
    SKP_int    temp;

    if( psRC->error ) {
        /* Set output to zero */
        *data = 0;
        return;
    }

    if( prob[ 2 ] == 65535 ) {
        /* Instead of detection, we could add a separate function and call when we know that output is a bit */
        *data = ec_dec_bit_prob( &psRC->range_dec_celt_state, 65536 - prob[ 1 ] );
    } else {
        low_Q16_returned = ec_decode_bin( &psRC->range_dec_celt_state, 16 );
    }

    /* OPTIMIZE ME WITH BI-SECTION */
    if( prob[ 2 ] != 65535 ) {
#if 1
        temp = 0;
        while( low_Q16_returned >= prob[ ++temp ] ) {}
        *data = temp - 1;
#else
        temp = probIx;
        if( low_Q16_returned >= prob[ temp ] ){
            while( low_Q16_returned >= prob[ temp ] ) {
                temp++;
            }
            temp = temp - 1;
        } else {
            /* search down */
            while( low_Q16_returned < prob[ temp ] ) {
                temp--;
            }
        }
        *data = temp;
#endif

        low_Q16  = prob[ *data ];
        high_Q16 = prob[ *data + 1 ];

#ifdef SAVE_ALL_INTERNAL_DATA
        DEBUG_STORE_DATA( dec_lr.dat, &low_Q16_returned,  sizeof( SKP_uint32 ) );
        DEBUG_STORE_DATA( dec_l.dat,  &low_Q16,           sizeof( SKP_uint32 ) );
        DEBUG_STORE_DATA( dec_h.dat,  &high_Q16,          sizeof( SKP_uint32 ) );
#endif  
        ec_dec_update( &psRC->range_dec_celt_state, low_Q16, high_Q16,( 1 << 16 ) );
    }
#ifdef SAVE_ALL_INTERNAL_DATA
    DEBUG_STORE_DATA( dec.dat, data, sizeof( SKP_int ) );
#endif
}

/* Range decoder for one symbol, with uniform PDF*/
void SKP_Silk_range_decode_uniform(
    SKP_int                         data[],             /* O    uncompressed data                           */
    SKP_Silk_range_coder_state      *psRC,              /* I/O  compressor data structure                   */
    const SKP_int                   N                   /* I    number of possible outcomes                 */
)
{
    SKP_int i;
    SKP_uint16 delta, prob[ MAX_SIZE + 1 ];

    SKP_assert( N < MAX_SIZE );

    delta = ( SKP_uint16 )SKP_DIV32_16( 65535, N );
    prob[ 0 ] = 0;
    for( i = 0; i < N - 1; i++ ) {
        prob[ i + 1 ] = prob[ i ] + delta;
    }
    prob[ N ] = 65535;

    SKP_Silk_range_decoder( data, psRC, prob, ( N >> 1 ) );
}

/* Range decoder for multiple symbols */
void SKP_Silk_range_decoder_multi(
    SKP_int                         data[],             /* O    uncompressed data                [nSymbols] */
    SKP_Silk_range_coder_state      *psRC,              /* I/O  compressor data structure                   */
    const SKP_uint16 * const        prob[],             /* I    cumulative density functions                */
    const SKP_int                   probStartIx[],      /* I    initial (middle) entries of cdfs [nSymbols] */
    const SKP_int                   nSymbols            /* I    number of data symbols                      */
)
{
    SKP_int k;
    for( k = 0; k < nSymbols; k++ ) {
        SKP_Silk_range_decoder( &data[ k ], psRC, prob[ k ], probStartIx[ k ] );
    }
}

/* Initialize range encoder */
void SKP_Silk_range_enc_init(
    SKP_Silk_range_coder_state      *psRC               /* O    compressor data structure                   */
)
{
    psRC->error        = 0;
}

/* Initialize range decoder */
void SKP_Silk_range_dec_init(
    SKP_Silk_range_coder_state      *psRC,              /* O    compressor data structure                   */
    const SKP_uint8                 buffer[],           /* I    buffer for compressed data [bufferLength]   */
    const SKP_int32                 bufferLength        /* I    buffer length (in bytes)                    */
)
{
    /* check input */
    if( bufferLength > MAX_ARITHM_BYTES ) {
        psRC->error = RANGE_CODER_DEC_PAYLOAD_TOO_LONG;
        return;
    }
    /* Initialize structure */
    /* Copy to internal buffer */
    SKP_memcpy( psRC->buffer, buffer, bufferLength * sizeof( SKP_uint8 ) ); 
    psRC->bufferLength = bufferLength;
    psRC->bufferIx = 0;
    psRC->base_Q32 = 
        SKP_LSHIFT_uint( ( SKP_uint32 )buffer[ 0 ], 24 ) | 
        SKP_LSHIFT_uint( ( SKP_uint32 )buffer[ 1 ], 16 ) | 
        SKP_LSHIFT_uint( ( SKP_uint32 )buffer[ 2 ],  8 ) | 
                         ( SKP_uint32 )buffer[ 3 ];
    psRC->range_Q16 = 0x0000FFFF;
    psRC->error     = 0;
}

/* Determine length of bitstream */
SKP_int SKP_Silk_range_encoder_get_length(              /* O    returns number of BITS in stream            */
    SKP_Silk_range_coder_state          *psRC,          /* I    compressed data structure                   */
    SKP_int                             *nBytes         /* O    number of BYTES in stream                   */
)
{
    SKP_int nBits;

    /* Get number of bits in bitstream */
    nBits = ec_enc_tell( &psRC->range_enc_celt_state, 0 );

    /* Round up to an integer number of bytes */
    *nBytes = SKP_RSHIFT( nBits + 7, 3 );

    /* Return number of bits in bitstream */
    return nBits;
}

/* Determine length of bitstream */
SKP_int SKP_Silk_range_decoder_get_length(              /* O    returns number of BITS in stream            */
    SKP_Silk_range_coder_state          *psRC,          /* I    compressed data structure                   */
    SKP_int                             *nBytes         /* O    number of BYTES in stream                   */
)
{
    SKP_int nBits;

    /* Get number of bits in bitstream */
    nBits = ec_dec_tell( &psRC->range_dec_celt_state, 0 );

    /* Round up to an integer number of bytes */
    *nBytes = SKP_RSHIFT( nBits + 7, 3 );

    /* Return number of bits in bitstream */
    return nBits;
}

/* Check that any remaining bits in the last byte are set to 1 */
void SKP_Silk_range_coder_check_after_decoding(
    SKP_Silk_range_coder_state      *psRC               /* I/O  compressed data structure                   */
)
{
    SKP_int bits_in_stream, nBytes, mask;

    bits_in_stream = SKP_Silk_range_decoder_get_length( psRC, &nBytes );

    /* Make sure not to read beyond buffer */
    if( nBytes - 1 >= psRC->range_dec_celt_state.buf->storage ) {
        psRC->error = RANGE_CODER_DECODER_CHECK_FAILED;
        return;
    }

    /* Test any remaining bits in last byte */
    if( bits_in_stream & 7 ) {
        mask = SKP_RSHIFT( 0xFF, bits_in_stream & 7 );
        if( ( psRC->range_dec_celt_state.buf->buf[ nBytes - 1 ] & mask ) != mask ) {
            psRC->error = RANGE_CODER_DECODER_CHECK_FAILED;
            return;
        }
    }
}
