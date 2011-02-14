/***********************************************************************
Copyright (c) 2006-2011, Skype Limited. All rights reserved. 
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

#ifndef _SIGPROCFIX_API_DEBUG_H_
#define _SIGPROCFIX_API_DEBUG_H_

// Redefine macro functions with extensive assertion in Win32_DEBUG mode. 
// As function can't be undefined, this file can't work with SigProcFIX_MacroCount.h

#if 0 && defined (_WIN32) && defined (_DEBUG) && !defined (SKP_MACRO_COUNT)

#undef	SKP_ADD16
SKP_INLINE SKP_int16 SKP_ADD16(SKP_int16 a, SKP_int16 b){
	SKP_int16 ret;

	ret = a + b;
	SKP_assert( ret == SKP_ADD_SAT16( a, b ));
	return ret;
}

#undef	SKP_ADD32
SKP_INLINE SKP_int32 SKP_ADD32(SKP_int32 a, SKP_int32 b){
	SKP_int32 ret;

	ret = a + b;
	SKP_assert( ret == SKP_ADD_SAT32( a, b ));
	return ret;
}

#undef	SKP_ADD64
SKP_INLINE SKP_int64 SKP_ADD64(SKP_int64 a, SKP_int64 b){
	SKP_int64 ret;

	ret = a + b;
	SKP_assert( ret == SKP_ADD_SAT64( a, b ));
	return ret;
}

#undef	SKP_SUB16
SKP_INLINE SKP_int16 SKP_SUB16(SKP_int16 a, SKP_int16 b){
	SKP_int16 ret;

	ret = a - b;
	SKP_assert( ret == SKP_SUB_SAT16( a, b ));
	return ret;
}

#undef	SKP_SUB32
SKP_INLINE SKP_int32 SKP_SUB32(SKP_int32 a, SKP_int32 b){
	SKP_int32 ret;

	ret = a - b;
	SKP_assert( ret == SKP_SUB_SAT32( a, b ));
	return ret;
}

#undef	SKP_SUB64
SKP_INLINE SKP_int64 SKP_SUB64(SKP_int64 a, SKP_int64 b){
	SKP_int64 ret;

	ret = a - b;
	SKP_assert( ret == SKP_SUB_SAT64( a, b ));
	return ret;
}

#undef SKP_ADD_SAT16
SKP_INLINE SKP_int16 SKP_ADD_SAT16( SKP_int16 a16, SKP_int16 b16 ) {
	SKP_int16 res;
	res = (SKP_int16)SKP_SAT16( SKP_ADD32( (SKP_int32)(a16), (b16) ) );
	SKP_assert( res == SKP_SAT16( ( SKP_int32 )a16 + ( SKP_int32 )b16 ) );
	return res;
}

#undef SKP_ADD_SAT32
SKP_INLINE SKP_int32 SKP_ADD_SAT32(SKP_int32 a32, SKP_int32 b32){
	SKP_int32 res;
	res =	((((a32) + (b32)) & 0x80000000) == 0 ?									\
			((((a32) & (b32)) & 0x80000000) != 0 ? SKP_int32_MIN : (a32)+(b32)) :	\
			((((a32) | (b32)) & 0x80000000) == 0 ? SKP_int32_MAX : (a32)+(b32)) );
	SKP_assert( res == SKP_SAT32( ( SKP_int64 )a32 + ( SKP_int64 )b32 ) );
	return res;
}

#undef SKP_ADD_SAT64
SKP_INLINE SKP_int64 SKP_ADD_SAT64( SKP_int64 a64, SKP_int64 b64 ) {
	SKP_int64 res;
	res =	((((a64) + (b64)) & 0x8000000000000000LL) == 0 ?								\
			((((a64) & (b64)) & 0x8000000000000000LL) != 0 ? SKP_int64_MIN : (a64)+(b64)) :	\
			((((a64) | (b64)) & 0x8000000000000000LL) == 0 ? SKP_int64_MAX : (a64)+(b64)) );
	if( res != a64 + b64 ) {
		// Check that we saturated to the correct extreme value
		SKP_assert( ( res == SKP_int64_MAX && ( ( a64 >> 1 ) + ( b64 >> 1 ) > ( SKP_int64_MAX >> 3 ) ) ) ||
					( res == SKP_int64_MIN && ( ( a64 >> 1 ) + ( b64 >> 1 ) < ( SKP_int64_MIN >> 3 ) ) ) );
	} else {
		// Saturation not necessary
		SKP_assert( res == a64 + b64 );
	}
	return res;
}

#undef SKP_SUB_SAT16
SKP_INLINE SKP_int16 SKP_SUB_SAT16( SKP_int16 a16, SKP_int16 b16 ) {
	SKP_int16 res;
	res = (SKP_int16)SKP_SAT16( SKP_SUB32( (SKP_int32)(a16), (b16) ) );
	SKP_assert( res == SKP_SAT16( ( SKP_int32 )a16 - ( SKP_int32 )b16 ) );
	return res;
}

#undef SKP_SUB_SAT32
SKP_INLINE SKP_int32 SKP_SUB_SAT32( SKP_int32 a32, SKP_int32 b32 ) {
	SKP_int32 res;
	res = 	((((a32)-(b32)) & 0x80000000) == 0 ?											\
			(( (a32) & ((b32)^0x80000000) & 0x80000000) ? SKP_int32_MIN : (a32)-(b32)) :	\
			((((a32)^0x80000000) & (b32)  & 0x80000000) ? SKP_int32_MAX : (a32)-(b32)) );
	SKP_assert( res == SKP_SAT32( ( SKP_int64 )a32 - ( SKP_int64 )b32 ) );
	return res;
}

#undef SKP_SUB_SAT64
SKP_INLINE SKP_int64 SKP_SUB_SAT64( SKP_int64 a64, SKP_int64 b64 ) {
	SKP_int64 res;
	res =	((((a64)-(b64)) & 0x8000000000000000LL) == 0 ?														\
			(( (a64) & ((b64)^0x8000000000000000LL) & 0x8000000000000000LL) ? SKP_int64_MIN : (a64)-(b64)) :	\
			((((a64)^0x8000000000000000LL) & (b64)  & 0x8000000000000000LL) ? SKP_int64_MAX : (a64)-(b64)) );

	if( res != a64 - b64 ) {
		// Check that we saturated to the correct extreme value
		SKP_assert( ( res == SKP_int64_MAX && ( ( a64 >> 1 ) + ( b64 >> 1 ) > ( SKP_int64_MAX >> 3 ) ) ) ||
					( res == SKP_int64_MIN && ( ( a64 >> 1 ) + ( b64 >> 1 ) < ( SKP_int64_MIN >> 3 ) ) ) );
	} else {
		// Saturation not necessary
		SKP_assert( res == a64 - b64 );
	}
	return res;
}

#undef SKP_MUL
SKP_INLINE SKP_int32 SKP_MUL(SKP_int32 a32, SKP_int32 b32){
	SKP_int32 ret;
	SKP_int64 ret64; // Will easily show how many bits that are needed
	ret = a32 * b32;
	ret64 = (SKP_int64)a32 * (SKP_int64)b32; 
	SKP_assert((SKP_int64)ret == ret64 );		//Check output overflow
	return ret;
}

#undef SKP_MUL_uint
SKP_INLINE SKP_uint32 SKP_MUL_uint(SKP_uint32 a32, SKP_uint32 b32){
	SKP_uint32 ret;
	ret = a32 * b32;
	SKP_assert((SKP_uint64)ret == (SKP_uint64)a32 * (SKP_uint64)b32);		//Check output overflow
	return ret;
}
#undef SKP_MLA
SKP_INLINE SKP_int32 SKP_MLA(SKP_int32 a32, SKP_int32 b32, SKP_int32 c32){
	SKP_int32 ret;
	ret = a32 + b32 * c32;
	SKP_assert((SKP_int64)ret == (SKP_int64)a32 + (SKP_int64)b32 * (SKP_int64)c32);	//Check output overflow
	return ret;
}

#undef SKP_MLA_uint
SKP_INLINE SKP_int32 SKP_MLA_uint(SKP_uint32 a32, SKP_uint32 b32, SKP_uint32 c32){
	SKP_uint32 ret;
	ret = a32 + b32 * c32;
	SKP_assert((SKP_int64)ret == (SKP_int64)a32 + (SKP_int64)b32 * (SKP_int64)c32);	//Check output overflow
	return ret;
}

#undef	SKP_SMULWB
SKP_INLINE SKP_int32 SKP_SMULWB(SKP_int32 a32, SKP_int32 b32){	
	SKP_int32 ret;
	ret = (a32 >> 16) * (SKP_int32)((SKP_int16)b32) + (((a32 & 0x0000FFFF) * (SKP_int32)((SKP_int16)b32)) >> 16);
	SKP_assert((SKP_int64)ret == ((SKP_int64)a32 * (SKP_int16)b32) >> 16);
	return ret;
}
#undef	SKP_SMLAWB
SKP_INLINE SKP_int32 SKP_SMLAWB(SKP_int32 a32, SKP_int32 b32, SKP_int32 c32){	
	SKP_int32 ret;
	ret = SKP_ADD32( a32, SKP_SMULWB( b32, c32 ) );
	SKP_assert(SKP_ADD32( a32, SKP_SMULWB( b32, c32 ) ) == SKP_ADD_SAT32( a32, SKP_SMULWB( b32, c32 ) ));
	return ret;
}

#undef SKP_SMULWT
SKP_INLINE SKP_int32 SKP_SMULWT(SKP_int32 a32, SKP_int32 b32){
	SKP_int32 ret;
	ret = (a32 >> 16) * (b32 >> 16) + (((a32 & 0x0000FFFF) * (b32 >> 16)) >> 16);
	SKP_assert((SKP_int64)ret == ((SKP_int64)a32 * (b32 >> 16)) >> 16);
	return ret;
}
#undef SKP_SMLAWT
SKP_INLINE SKP_int32 SKP_SMLAWT(SKP_int32 a32, SKP_int32 b32, SKP_int32 c32){
	SKP_int32 ret;
	ret = a32 + ((b32 >> 16) * (c32 >> 16)) + (((b32 & 0x0000FFFF) * ((c32 >> 16)) >> 16));
	SKP_assert((SKP_int64)ret == (SKP_int64)a32 + (((SKP_int64)b32 * (c32 >> 16)) >> 16));
	return ret;
}

#undef SKP_SMULL
SKP_INLINE SKP_int64 SKP_SMULL(SKP_int64 a64, SKP_int64 b64){
	SKP_int64 ret64;
	ret64 = a64 * b64;
	if( b64 != 0 ) {
		SKP_assert( a64 == (ret64 / b64) );
	} else if( a64 != 0 ) {
		SKP_assert( b64 == (ret64 / a64) );
	}
	return ret64;
}

// no checking needed for SKP_SMULBB
#undef	SKP_SMLABB
SKP_INLINE SKP_int32 SKP_SMLABB(SKP_int32 a32, SKP_int32 b32, SKP_int32 c32){
	SKP_int32 ret;
	ret = a32 + (SKP_int32)((SKP_int16)b32) * (SKP_int32)((SKP_int16)c32);
	SKP_assert((SKP_int64)ret == (SKP_int64)a32 + (SKP_int64)b32 * (SKP_int16)c32);
	return ret;
}

// no checking needed for SKP_SMULBT
#undef	SKP_SMLABT
SKP_INLINE SKP_int32 SKP_SMLABT(SKP_int32 a32, SKP_int32 b32, SKP_int32 c32){
	SKP_int32 ret;
	ret = a32 + ((SKP_int32)((SKP_int16)b32)) * (c32 >> 16);
	SKP_assert((SKP_int64)ret == (SKP_int64)a32 + (SKP_int64)b32 * (c32 >> 16));
	return ret;
}

// no checking needed for SKP_SMULTT
#undef	SKP_SMLATT
SKP_INLINE SKP_int32 SKP_SMLATT(SKP_int32 a32, SKP_int32 b32, SKP_int32 c32){
	SKP_int32 ret;
	ret = a32 + (b32 >> 16) * (c32 >> 16);
	SKP_assert((SKP_int64)ret == (SKP_int64)a32 + (b32 >> 16) * (c32 >> 16));
	return ret;
}

#undef	SKP_SMULWW
SKP_INLINE SKP_int32 SKP_SMULWW(SKP_int32 a32, SKP_int32 b32){	
	SKP_int32 ret, tmp1, tmp2;
	SKP_int64 ret64;

	ret  = SKP_SMULWB( a32, b32 );
	tmp1 = SKP_RSHIFT_ROUND( b32, 16 );
	tmp2 = SKP_MUL( a32, tmp1 );
	
	SKP_assert( (SKP_int64)tmp2 == (SKP_int64) a32 * (SKP_int64) tmp1 );
	
	tmp1 = ret;
	ret  = SKP_ADD32( tmp1, tmp2 );
	SKP_assert( SKP_ADD32( tmp1, tmp2 ) == SKP_ADD_SAT32( tmp1, tmp2 ) );
	
	ret64 = SKP_RSHIFT64( SKP_SMULL( a32, b32 ), 16 );
	SKP_assert( (SKP_int64)ret == ret64 );

	return ret;
}

#undef	SKP_SMLAWW
SKP_INLINE SKP_int32 SKP_SMLAWW(SKP_int32 a32, SKP_int32 b32, SKP_int32 c32){	
	SKP_int32 ret, tmp;

	tmp = SKP_SMULWW( b32, c32 );
	ret = SKP_ADD32( a32, tmp );
	SKP_assert( ret == SKP_ADD_SAT32( a32, tmp ) );	
	return ret;
}

// multiply-accumulate macros that allow overflow in the addition (ie, no asserts in debug mode)
#undef	SKP_MLA_ovflw
#define SKP_MLA_ovflw(a32, b32, c32)	((a32) + ((b32) * (c32)))
#undef	SKP_SMLABB_ovflw
#define SKP_SMLABB_ovflw(a32, b32, c32)	((a32) + ((SKP_int32)((SKP_int16)(b32))) * (SKP_int32)((SKP_int16)(c32)))
#undef	SKP_SMLABT_ovflw
#define SKP_SMLABT_ovflw(a32, b32, c32)	((a32) + ((SKP_int32)((SKP_int16)(b32))) * ((c32) >> 16))
#undef	SKP_SMLATT_ovflw
#define SKP_SMLATT_ovflw(a32, b32, c32)	((a32) + ((b32) >> 16) * ((c32) >> 16))
#undef	SKP_SMLAWB_ovflw
#define SKP_SMLAWB_ovflw(a32, b32, c32)	((a32) + ((((b32) >> 16) * (SKP_int32)((SKP_int16)(c32))) + ((((b32) & 0x0000FFFF) * (SKP_int32)((SKP_int16)(c32))) >> 16)))
#undef	SKP_SMLAWT_ovflw
#define SKP_SMLAWT_ovflw(a32, b32, c32)	((a32) + (((b32) >> 16) * ((c32) >> 16)) + ((((b32) & 0x0000FFFF) * ((c32) >> 16)) >> 16))

// no checking needed for SKP_SMULL
// no checking needed for SKP_SMLAL
// no checking needed for SKP_SMLALBB
// no checking needed for SigProcFIX_CLZ16
// no checking needed for SigProcFIX_CLZ32

#undef SKP_DIV32
SKP_INLINE SKP_int32 SKP_DIV32(SKP_int32 a32, SKP_int32 b32){
	SKP_assert( b32 != 0 );
	return a32 / b32;
}

#undef SKP_DIV32_16
SKP_INLINE SKP_int32 SKP_DIV32_16(SKP_int32 a32, SKP_int32 b32){
	SKP_assert( b32 != 0 );
	SKP_assert( b32 <= SKP_int16_MAX );
	SKP_assert( b32 >= SKP_int16_MIN );
	return a32 / b32;
}

// no checking needed for SKP_SAT8
// no checking needed for SKP_SAT16
// no checking needed for SKP_SAT32
// no checking needed for SKP_POS_SAT32
// no checking needed for SKP_ADD_POS_SAT8
// no checking needed for SKP_ADD_POS_SAT16
// no checking needed for SKP_ADD_POS_SAT32
// no checking needed for SKP_ADD_POS_SAT64
#undef	SKP_LSHIFT8
SKP_INLINE SKP_int8 SKP_LSHIFT8(SKP_int8 a, SKP_int32 shift){
	SKP_int8 ret;
	ret = a << shift;
	SKP_assert(shift >= 0);
	SKP_assert(shift < 8);
	SKP_assert((SKP_int64)ret == ((SKP_int64)a) << shift);
	return ret;
}
#undef	SKP_LSHIFT16
SKP_INLINE SKP_int16 SKP_LSHIFT16(SKP_int16 a, SKP_int32 shift){
	SKP_int16 ret;
	ret = a << shift;
	SKP_assert(shift >= 0);
	SKP_assert(shift < 16);
	SKP_assert((SKP_int64)ret == ((SKP_int64)a) << shift);
	return ret;
}
#undef	SKP_LSHIFT32
SKP_INLINE SKP_int32 SKP_LSHIFT32(SKP_int32 a, SKP_int32 shift){
	SKP_int32 ret;
	ret = a << shift;
	SKP_assert(shift >= 0);
	SKP_assert(shift < 32);
	SKP_assert((SKP_int64)ret == ((SKP_int64)a) << shift);
	return ret;
}
#undef	SKP_LSHIFT64
SKP_INLINE SKP_int64 SKP_LSHIFT64(SKP_int64 a, SKP_int shift){
	SKP_assert(shift >= 0);
	SKP_assert(shift < 64);
	return a << shift;
}

#undef	SKP_LSHIFT_ovflw
SKP_INLINE SKP_int32 SKP_LSHIFT_ovflw(SKP_int32 a, SKP_int32 shift){
	SKP_assert(shift >= 0);			/* no check for overflow */
	return a << shift;
}

#undef	SKP_LSHIFT_uint
SKP_INLINE SKP_uint32 SKP_LSHIFT_uint(SKP_uint32 a, SKP_int32 shift){
	SKP_uint32 ret;
	ret = a << shift;
	SKP_assert(shift >= 0);
	SKP_assert((SKP_int64)ret == ((SKP_int64)a) << shift);
	return ret;
}

#undef	SKP_RSHIFT8
SKP_INLINE SKP_int8 SKP_RSHIFT8(SKP_int8 a, SKP_int32 shift){
	SKP_assert(shift >=  0);
	SKP_assert(shift < 8);
	return a >> shift;
}
#undef	SKP_RSHIFT16
SKP_INLINE SKP_int16 SKP_RSHIFT16(SKP_int16 a, SKP_int32 shift){
	SKP_assert(shift >=  0);
	SKP_assert(shift < 16);
	return a >> shift;
}
#undef	SKP_RSHIFT32
SKP_INLINE SKP_int32 SKP_RSHIFT32(SKP_int32 a, SKP_int32 shift){
	SKP_assert(shift >=  0);
	SKP_assert(shift < 32);
	return a >> shift;
}
#undef	SKP_RSHIFT64
SKP_INLINE SKP_int64 SKP_RSHIFT64(SKP_int64 a, SKP_int64 shift){
	SKP_assert(shift >=  0);
	SKP_assert(shift <= 63);
	return a >> shift;
}

#undef	SKP_RSHIFT_uint
SKP_INLINE SKP_uint32 SKP_RSHIFT_uint(SKP_uint32 a, SKP_int32 shift){
	SKP_assert(shift >=  0);
	SKP_assert(shift <= 32);
	return a >> shift;
}

#undef	SKP_ADD_LSHIFT
SKP_INLINE SKP_int32 SKP_ADD_LSHIFT(SKP_int32 a, SKP_int32 b, SKP_int32 shift){
	SKP_int32 ret;
	SKP_assert(shift >= 0);
	SKP_assert(shift <= 31);
	ret = a + (b << shift);
	SKP_assert((SKP_int64)ret == (SKP_int64)a + (((SKP_int64)b) << shift));
	return ret;				// shift >= 0
}
#undef	SKP_ADD_LSHIFT32
SKP_INLINE SKP_int32 SKP_ADD_LSHIFT32(SKP_int32 a, SKP_int32 b, SKP_int32 shift){
	SKP_int32 ret;
	SKP_assert(shift >= 0);
	SKP_assert(shift <= 31);
	ret = a + (b << shift);
	SKP_assert((SKP_int64)ret == (SKP_int64)a + (((SKP_int64)b) << shift));
	return ret;				// shift >= 0
}
#undef	SKP_ADD_LSHIFT_uint
SKP_INLINE SKP_uint32 SKP_ADD_LSHIFT_uint(SKP_uint32 a, SKP_uint32 b, SKP_int32 shift){
	SKP_uint32 ret;
	SKP_assert(shift >= 0);
	SKP_assert(shift <= 32);
	ret = a + (b << shift);
	SKP_assert((SKP_int64)ret == (SKP_int64)a + (((SKP_int64)b) << shift));
	return ret;				// shift >= 0
}
#undef	SKP_ADD_RSHIFT
SKP_INLINE SKP_int32 SKP_ADD_RSHIFT(SKP_int32 a, SKP_int32 b, SKP_int32 shift){		
	SKP_int32 ret;
	SKP_assert(shift >= 0);
	SKP_assert(shift <= 31);
	ret = a + (b >> shift);
	SKP_assert((SKP_int64)ret == (SKP_int64)a + (((SKP_int64)b) >> shift));
	return ret;				// shift  > 0
}
#undef	SKP_ADD_RSHIFT32
SKP_INLINE SKP_int32 SKP_ADD_RSHIFT32(SKP_int32 a, SKP_int32 b, SKP_int32 shift){		
	SKP_int32 ret;
	SKP_assert(shift >= 0);
	SKP_assert(shift <= 31);
	ret = a + (b >> shift);
	SKP_assert((SKP_int64)ret == (SKP_int64)a + (((SKP_int64)b) >> shift));
	return ret;				// shift  > 0
}
#undef	SKP_ADD_RSHIFT_uint
SKP_INLINE SKP_uint32 SKP_ADD_RSHIFT_uint(SKP_uint32 a, SKP_uint32 b, SKP_int32 shift){		
	SKP_uint32 ret;
	SKP_assert(shift >= 0);
	SKP_assert(shift <= 32);
	ret = a + (b >> shift);
	SKP_assert((SKP_int64)ret == (SKP_int64)a + (((SKP_int64)b) >> shift));
	return ret;				// shift  > 0
}
#undef	SKP_SUB_LSHIFT32
SKP_INLINE SKP_int32 SKP_SUB_LSHIFT32(SKP_int32 a, SKP_int32 b, SKP_int32 shift){
	SKP_int32 ret;
	SKP_assert(shift >= 0);
	SKP_assert(shift <= 31);
	ret = a - (b << shift);
	SKP_assert((SKP_int64)ret == (SKP_int64)a - (((SKP_int64)b) << shift));
	return ret;				// shift >= 0
}
#undef	SKP_SUB_RSHIFT32
SKP_INLINE SKP_int32 SKP_SUB_RSHIFT32(SKP_int32 a, SKP_int32 b, SKP_int32 shift){		
	SKP_int32 ret;
	SKP_assert(shift >= 0);
	SKP_assert(shift <= 31);
	ret = a - (b >> shift);
	SKP_assert((SKP_int64)ret == (SKP_int64)a - (((SKP_int64)b) >> shift));
	return ret;				// shift  > 0
}

#undef	SKP_RSHIFT_ROUND
SKP_INLINE SKP_int32 SKP_RSHIFT_ROUND(SKP_int32 a, SKP_int32 shift){
	SKP_int32 ret;
	SKP_assert(shift > 0);		/* the marco definition can't handle a shift of zero */
	SKP_assert(shift < 32);
	ret = shift == 1 ? (a >> 1) + (a & 1) : ((a >> (shift - 1)) + 1) >> 1;
	SKP_assert((SKP_int64)ret == ((SKP_int64)a + ((SKP_int64)1 << (shift - 1))) >> shift);
	return ret;
}

#undef	SKP_RSHIFT_ROUND64
SKP_INLINE SKP_int64 SKP_RSHIFT_ROUND64(SKP_int64 a, SKP_int32 shift){
	SKP_int64 ret;
	SKP_assert(shift > 0);		/* the marco definition can't handle a shift of zero */
	SKP_assert(shift < 64);
	ret = shift == 1 ? (a >> 1) + (a & 1) : ((a >> (shift - 1)) + 1) >> 1;
	return ret;
}

// SKP_abs is used on floats also, so doesn't work...
//#undef	SKP_abs
//SKP_INLINE SKP_int32 SKP_abs(SKP_int32 a){
//	SKP_assert(a != 0x80000000);
//	return (((a) >  0)  ? (a) : -(a));			// Be careful, SKP_abs returns wrong when input equals to SKP_intXX_MIN
//}

#undef	SKP_abs_int64
SKP_INLINE SKP_int64 SKP_abs_int64(SKP_int64 a){
	SKP_assert(a != 0x8000000000000000);
	return (((a) >  0)  ? (a) : -(a));			// Be careful, SKP_abs returns wrong when input equals to SKP_intXX_MIN
}

#undef	SKP_abs_int32
SKP_INLINE SKP_int32 SKP_abs_int32(SKP_int32 a){
	SKP_assert(a != 0x80000000);
	return abs(a);
}

#undef	SKP_CHECK_FIT8
SKP_INLINE SKP_int8 SKP_CHECK_FIT8( SKP_int64 a ){
	SKP_int8 ret;
	ret = (SKP_int8)a;
	SKP_assert( (SKP_int64)ret == a );
	return( ret ); 
}

#undef	SKP_CHECK_FIT16
SKP_INLINE SKP_int16 SKP_CHECK_FIT16( SKP_int64 a ){
	SKP_int16 ret;
	ret = (SKP_int16)a;
	SKP_assert( (SKP_int64)ret == a );
	return( ret ); 
}

#undef	SKP_CHECK_FIT32
SKP_INLINE SKP_int32 SKP_CHECK_FIT32( SKP_int64 a ){
	SKP_int32 ret;
	ret = (SKP_int32)a;
	SKP_assert( (SKP_int64)ret == a );
	return( ret ); 
}

// no checking for SKP_NSHIFT_MUL_32_32
// no checking for SKP_NSHIFT_MUL_16_16	
// no checking needed for SKP_min
// no checking needed for SKP_max
// no checking needed for SKP_sign

#endif
#endif
