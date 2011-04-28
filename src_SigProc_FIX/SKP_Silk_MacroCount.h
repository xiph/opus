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

#ifndef _SIGPROCFIX_API_MACROCOUNT_H_
#define _SIGPROCFIX_API_MACROCOUNT_H_
#include <stdio.h>

#ifdef	SKP_MACRO_COUNT
#define varDefine SKP_int64 ops_count = 0;

extern SKP_int64 ops_count;

SKP_INLINE SKP_int64 SKP_SaveCount(){
	return(ops_count);
}

SKP_INLINE SKP_int64 SKP_SaveResetCount(){
	SKP_int64 ret;

	ret = ops_count;
	ops_count = 0;
	return(ret);
}

SKP_INLINE SKP_PrintCount(){
	printf("ops_count = %d \n ", (SKP_int32)ops_count);
}

#undef SKP_MUL
SKP_INLINE SKP_int32 SKP_MUL(SKP_int32 a32, SKP_int32 b32){
	SKP_int32 ret;
	ops_count += 4;
	ret = a32 * b32;
	return ret;
}

#undef SKP_MUL_uint
SKP_INLINE SKP_uint32 SKP_MUL_uint(SKP_uint32 a32, SKP_uint32 b32){
	SKP_uint32 ret;
	ops_count += 4;
	ret = a32 * b32;
	return ret;
}
#undef SKP_MLA
SKP_INLINE SKP_int32 SKP_MLA(SKP_int32 a32, SKP_int32 b32, SKP_int32 c32){
	SKP_int32 ret;
	ops_count += 4;
	ret = a32 + b32 * c32;
	return ret;
}

#undef SKP_MLA_uint
SKP_INLINE SKP_int32 SKP_MLA_uint(SKP_uint32 a32, SKP_uint32 b32, SKP_uint32 c32){
	SKP_uint32 ret;
	ops_count += 4;
	ret = a32 + b32 * c32;
	return ret;
}

#undef SKP_SMULWB
SKP_INLINE SKP_int32 SKP_SMULWB(SKP_int32 a32, SKP_int32 b32){	
	SKP_int32 ret;
	ops_count += 5;
	ret = (a32 >> 16) * (SKP_int32)((SKP_int16)b32) + (((a32 & 0x0000FFFF) * (SKP_int32)((SKP_int16)b32)) >> 16);
	return ret;
}
#undef	SKP_SMLAWB
SKP_INLINE SKP_int32 SKP_SMLAWB(SKP_int32 a32, SKP_int32 b32, SKP_int32 c32){	
	SKP_int32 ret;
	ops_count += 5;
	ret = ((a32) + ((((b32) >> 16) * (SKP_int32)((SKP_int16)(c32))) + ((((b32) & 0x0000FFFF) * (SKP_int32)((SKP_int16)(c32))) >> 16)));
	return ret;
}

#undef SKP_SMULWT
SKP_INLINE SKP_int32 SKP_SMULWT(SKP_int32 a32, SKP_int32 b32){
	SKP_int32 ret;
	ops_count += 4;
	ret = (a32 >> 16) * (b32 >> 16) + (((a32 & 0x0000FFFF) * (b32 >> 16)) >> 16);
	return ret;
}
#undef SKP_SMLAWT
SKP_INLINE SKP_int32 SKP_SMLAWT(SKP_int32 a32, SKP_int32 b32, SKP_int32 c32){
	SKP_int32 ret;
	ops_count += 4;
	ret = a32 + ((b32 >> 16) * (c32 >> 16)) + (((b32 & 0x0000FFFF) * ((c32 >> 16)) >> 16));
	return ret;
}

#undef SKP_SMULBB
SKP_INLINE SKP_int32 SKP_SMULBB(SKP_int32 a32, SKP_int32 b32){
	SKP_int32 ret;
	ops_count += 1;
	ret = (SKP_int32)((SKP_int16)a32) * (SKP_int32)((SKP_int16)b32);
	return ret;
}
#undef SKP_SMLABB
SKP_INLINE SKP_int32 SKP_SMLABB(SKP_int32 a32, SKP_int32 b32, SKP_int32 c32){
	SKP_int32 ret;
	ops_count += 1;
	ret = a32 + (SKP_int32)((SKP_int16)b32) * (SKP_int32)((SKP_int16)c32);
	return ret;
}

#undef SKP_SMULBT
SKP_INLINE SKP_int32 SKP_SMULBT(SKP_int32 a32, SKP_int32 b32 ){
	SKP_int32 ret;
	ops_count += 4;
	ret = ((SKP_int32)((SKP_int16)a32)) * (b32 >> 16);
	return ret;
}

#undef SKP_SMLABT
SKP_INLINE SKP_int32 SKP_SMLABT(SKP_int32 a32, SKP_int32 b32, SKP_int32 c32){
	SKP_int32 ret;
	ops_count += 1;
	ret = a32 + ((SKP_int32)((SKP_int16)b32)) * (c32 >> 16);
	return ret;
}

#undef SKP_SMULTT
SKP_INLINE SKP_int32 SKP_SMULTT(SKP_int32 a32, SKP_int32 b32){
	SKP_int32 ret;
	ops_count += 1;
	ret = (a32 >> 16) * (b32 >> 16);
	return ret;
}

#undef	SKP_SMLATT
SKP_INLINE SKP_int32 SKP_SMLATT(SKP_int32 a32, SKP_int32 b32, SKP_int32 c32){
	SKP_int32 ret;
	ops_count += 1;
	ret = a32 + (b32 >> 16) * (c32 >> 16);
	return ret;
}


// multiply-accumulate macros that allow overflow in the addition (ie, no asserts in debug mode)
#undef	SKP_MLA_ovflw
#define SKP_MLA_ovflw SKP_MLA

#undef SKP_SMLABB_ovflw
#define SKP_SMLABB_ovflw SKP_SMLABB

#undef SKP_SMLABT_ovflw
#define SKP_SMLABT_ovflw SKP_SMLABT

#undef SKP_SMLATT_ovflw
#define SKP_SMLATT_ovflw SKP_SMLATT

#undef SKP_SMLAWB_ovflw
#define SKP_SMLAWB_ovflw SKP_SMLAWB

#undef SKP_SMLAWT_ovflw
#define SKP_SMLAWT_ovflw SKP_SMLAWT

#undef SKP_SMULL
SKP_INLINE SKP_int64 SKP_SMULL(SKP_int32 a32, SKP_int32 b32){	
	SKP_int64 ret;
	ops_count += 8;
	ret = ((SKP_int64)(a32) * /*(SKP_int64)*/(b32));
	return ret;
}

#undef	SKP_SMLAL
SKP_INLINE SKP_int64 SKP_SMLAL(SKP_int64 a64, SKP_int32 b32, SKP_int32 c32){	
	SKP_int64 ret;
	ops_count += 8;
	ret = a64 + ((SKP_int64)(b32) * /*(SKP_int64)*/(c32));
	return ret;
}
#undef	SKP_SMLALBB
SKP_INLINE SKP_int64 SKP_SMLALBB(SKP_int64 a64, SKP_int16 b16, SKP_int16 c16){	
	SKP_int64 ret;
	ops_count += 4;
	ret = a64 + ((SKP_int64)(b16) * /*(SKP_int64)*/(c16));
	return ret;
}

#undef	SigProcFIX_CLZ16
SKP_INLINE SKP_int32 SigProcFIX_CLZ16(SKP_int16 in16)
{
    SKP_int32 out32 = 0;
	ops_count += 10;
    if( in16 == 0 ) {
        return 16;
    }
    /* test nibbles */
    if( in16 & 0xFF00 ) {
        if( in16 & 0xF000 ) {
            in16 >>= 12;
        } else {
            out32 += 4;
            in16 >>= 8;
        }
    } else {
        if( in16 & 0xFFF0 ) {
            out32 += 8;
            in16 >>= 4;
        } else {
            out32 += 12;
        }
    }
    /* test bits and return */
    if( in16 & 0xC ) {
        if( in16 & 0x8 )
            return out32 + 0;
        else
            return out32 + 1;
    } else {
        if( in16 & 0xE )
            return out32 + 2;
        else
            return out32 + 3;
    }
}

#undef SigProcFIX_CLZ32
SKP_INLINE SKP_int32 SigProcFIX_CLZ32(SKP_int32 in32)
{
    /* test highest 16 bits and convert to SKP_int16 */
	ops_count += 2;
    if( in32 & 0xFFFF0000 ) {
        return SigProcFIX_CLZ16((SKP_int16)(in32 >> 16));
    } else {
        return SigProcFIX_CLZ16((SKP_int16)in32) + 16;
    }
}

#undef SKP_DIV32
SKP_INLINE SKP_int32 SKP_DIV32(SKP_int32 a32, SKP_int32 b32){
	ops_count += 64;
	return a32 / b32;
}

#undef SKP_DIV32_16
SKP_INLINE SKP_int32 SKP_DIV32_16(SKP_int32 a32, SKP_int32 b32){
	ops_count += 32;
	return a32 / b32;
}

#undef SKP_SAT8
SKP_INLINE SKP_int8 SKP_SAT8(SKP_int64 a){
	SKP_int8 tmp;
	ops_count += 1;
	tmp = (SKP_int8)((a) > SKP_int8_MAX ? SKP_int8_MAX  : \
                    ((a) < SKP_int8_MIN ? SKP_int8_MIN  : (a)));
	return(tmp);
}

#undef SKP_SAT16
SKP_INLINE SKP_int16 SKP_SAT16(SKP_int64 a){
	SKP_int16 tmp;
	ops_count += 1;
	tmp = (SKP_int16)((a) > SKP_int16_MAX ? SKP_int16_MAX  : \
                     ((a) < SKP_int16_MIN ? SKP_int16_MIN  : (a)));
	return(tmp);
}
#undef SKP_SAT32
SKP_INLINE SKP_int32 SKP_SAT32(SKP_int64 a){
	SKP_int32 tmp;
	ops_count += 1;
	tmp = (SKP_int32)((a) > SKP_int32_MAX ? SKP_int32_MAX  : \
                     ((a) < SKP_int32_MIN ? SKP_int32_MIN  : (a)));
	return(tmp);
}
#undef SKP_POS_SAT32
SKP_INLINE SKP_int32 SKP_POS_SAT32(SKP_int64 a){
	SKP_int32 tmp;
	ops_count += 1;
	tmp = (SKP_int32)((a) > SKP_int32_MAX ? SKP_int32_MAX : (a));
	return(tmp);
}

#undef SKP_ADD_POS_SAT8
SKP_INLINE SKP_int8 SKP_ADD_POS_SAT8(SKP_int64 a, SKP_int64 b){
	SKP_int8 tmp;
	ops_count += 1;
	tmp = (SKP_int8)((((a)+(b)) & 0x80) ? SKP_int8_MAX  : ((a)+(b)));
	return(tmp);
}
#undef SKP_ADD_POS_SAT16
SKP_INLINE SKP_int16 SKP_ADD_POS_SAT16(SKP_int64 a, SKP_int64 b){
	SKP_int16 tmp;
	ops_count += 1;
	tmp = (SKP_int16)((((a)+(b)) & 0x8000) ? SKP_int16_MAX : ((a)+(b)));
	return(tmp);
}

#undef SKP_ADD_POS_SAT32
SKP_INLINE SKP_int32 SKP_ADD_POS_SAT32(SKP_int64 a, SKP_int64 b){
	SKP_int32 tmp;
	ops_count += 1;
	tmp = (SKP_int32)((((a)+(b)) & 0x80000000) ? SKP_int32_MAX : ((a)+(b)));
	return(tmp);
}

#undef SKP_ADD_POS_SAT64
SKP_INLINE SKP_int64 SKP_ADD_POS_SAT64(SKP_int64 a, SKP_int64 b){
	SKP_int64 tmp;
	ops_count += 1;
	tmp = ((((a)+(b)) & 0x8000000000000000LL) ? SKP_int64_MAX : ((a)+(b)));
	return(tmp);
}

#undef	SKP_LSHIFT8
SKP_INLINE SKP_int8 SKP_LSHIFT8(SKP_int8 a, SKP_int32 shift){
	SKP_int8 ret;
	ops_count += 1;
	ret = a << shift;
	return ret;
}
#undef	SKP_LSHIFT16
SKP_INLINE SKP_int16 SKP_LSHIFT16(SKP_int16 a, SKP_int32 shift){
	SKP_int16 ret;
	ops_count += 1;
	ret = a << shift;
	return ret;
}
#undef	SKP_LSHIFT32
SKP_INLINE SKP_int32 SKP_LSHIFT32(SKP_int32 a, SKP_int32 shift){
	SKP_int32 ret;
	ops_count += 1;
	ret = a << shift;
	return ret;
}
#undef	SKP_LSHIFT64
SKP_INLINE SKP_int64 SKP_LSHIFT64(SKP_int64 a, SKP_int shift){
	ops_count += 1;
	return a << shift;
}

#undef	SKP_LSHIFT_ovflw
SKP_INLINE SKP_int32 SKP_LSHIFT_ovflw(SKP_int32 a, SKP_int32 shift){
	ops_count += 1;
	return a << shift;
}

#undef	SKP_LSHIFT_uint
SKP_INLINE SKP_uint32 SKP_LSHIFT_uint(SKP_uint32 a, SKP_int32 shift){
	SKP_uint32 ret;
	ops_count += 1;
	ret = a << shift;
	return ret;
}

#undef	SKP_RSHIFT8
SKP_INLINE SKP_int8 SKP_RSHIFT8(SKP_int8 a, SKP_int32 shift){
	ops_count += 1;
	return a >> shift;
}
#undef	SKP_RSHIFT16
SKP_INLINE SKP_int16 SKP_RSHIFT16(SKP_int16 a, SKP_int32 shift){
	ops_count += 1;
	return a >> shift;
}
#undef	SKP_RSHIFT32
SKP_INLINE SKP_int32 SKP_RSHIFT32(SKP_int32 a, SKP_int32 shift){
	ops_count += 1;
	return a >> shift;
}
#undef	SKP_RSHIFT64
SKP_INLINE SKP_int64 SKP_RSHIFT64(SKP_int64 a, SKP_int64 shift){
	ops_count += 1;
	return a >> shift;
}

#undef	SKP_RSHIFT_uint
SKP_INLINE SKP_uint32 SKP_RSHIFT_uint(SKP_uint32 a, SKP_int32 shift){
	ops_count += 1;
	return a >> shift;
}

#undef	SKP_ADD_LSHIFT
SKP_INLINE SKP_int32 SKP_ADD_LSHIFT(SKP_int32 a, SKP_int32 b, SKP_int32 shift){
	SKP_int32 ret;
	ops_count += 1;
	ret = a + (b << shift);
	return ret;				// shift >= 0
}
#undef	SKP_ADD_LSHIFT32
SKP_INLINE SKP_int32 SKP_ADD_LSHIFT32(SKP_int32 a, SKP_int32 b, SKP_int32 shift){
	SKP_int32 ret;
	ops_count += 1;
	ret = a + (b << shift);
	return ret;				// shift >= 0
}
#undef	SKP_ADD_LSHIFT_uint
SKP_INLINE SKP_uint32 SKP_ADD_LSHIFT_uint(SKP_uint32 a, SKP_uint32 b, SKP_int32 shift){
	SKP_uint32 ret;
	ops_count += 1;
	ret = a + (b << shift);
	return ret;				// shift >= 0
}
#undef	SKP_ADD_RSHIFT
SKP_INLINE SKP_int32 SKP_ADD_RSHIFT(SKP_int32 a, SKP_int32 b, SKP_int32 shift){		
	SKP_int32 ret;
	ops_count += 1;
	ret = a + (b >> shift);
	return ret;				// shift  > 0
}
#undef	SKP_ADD_RSHIFT32
SKP_INLINE SKP_int32 SKP_ADD_RSHIFT32(SKP_int32 a, SKP_int32 b, SKP_int32 shift){		
	SKP_int32 ret;
	ops_count += 1;
	ret = a + (b >> shift);
	return ret;				// shift  > 0
}
#undef	SKP_ADD_RSHIFT_uint
SKP_INLINE SKP_uint32 SKP_ADD_RSHIFT_uint(SKP_uint32 a, SKP_uint32 b, SKP_int32 shift){		
	SKP_uint32 ret;
	ops_count += 1;
	ret = a + (b >> shift);
	return ret;				// shift  > 0
}
#undef	SKP_SUB_LSHIFT32
SKP_INLINE SKP_int32 SKP_SUB_LSHIFT32(SKP_int32 a, SKP_int32 b, SKP_int32 shift){
	SKP_int32 ret;
	ops_count += 1;
	ret = a - (b << shift);
	return ret;				// shift >= 0
}
#undef	SKP_SUB_RSHIFT32
SKP_INLINE SKP_int32 SKP_SUB_RSHIFT32(SKP_int32 a, SKP_int32 b, SKP_int32 shift){		
	SKP_int32 ret;
	ops_count += 1;
	ret = a - (b >> shift);
	return ret;				// shift  > 0
}

#undef	SKP_RSHIFT_ROUND
SKP_INLINE SKP_int32 SKP_RSHIFT_ROUND(SKP_int32 a, SKP_int32 shift){
	SKP_int32 ret;
	ops_count += 3;
	ret = shift == 1 ? (a >> 1) + (a & 1) : ((a >> (shift - 1)) + 1) >> 1;
	return ret;
}

#undef	SKP_RSHIFT_ROUND64
SKP_INLINE SKP_int64 SKP_RSHIFT_ROUND64(SKP_int64 a, SKP_int32 shift){
	SKP_int64 ret;
	ops_count += 6;
	ret = shift == 1 ? (a >> 1) + (a & 1) : ((a >> (shift - 1)) + 1) >> 1;
	return ret;
}

#undef	SKP_abs_int64
SKP_INLINE SKP_int64 SKP_abs_int64(SKP_int64 a){
	ops_count += 1;
	return (((a) >  0)  ? (a) : -(a));			// Be careful, SKP_abs returns wrong when input equals to SKP_intXX_MIN
}

#undef	SKP_abs_int32
SKP_INLINE SKP_int32 SKP_abs_int32(SKP_int32 a){
	ops_count += 1;
	return abs(a);
}


#undef SKP_min
static SKP_min(a, b){
	ops_count += 1;
	return (((a) < (b)) ? (a) :  (b));
}
#undef SKP_max
static SKP_max(a, b){
	ops_count += 1;
	return (((a) > (b)) ? (a) :  (b));
}
#undef SKP_sign
static SKP_sign(a){
	ops_count += 1;
	return ((a) > 0 ? 1 : ( (a) < 0 ? -1 : 0 ));
}

#undef	SKP_ADD16
SKP_INLINE SKP_int16 SKP_ADD16(SKP_int16 a, SKP_int16 b){
	SKP_int16 ret;
	ops_count += 1;
	ret = a + b;
	return ret;
}

#undef	SKP_ADD32
SKP_INLINE SKP_int32 SKP_ADD32(SKP_int32 a, SKP_int32 b){
	SKP_int32 ret;
	ops_count += 1;
	ret = a + b;
	return ret;
}

#undef	SKP_ADD64
SKP_INLINE SKP_int64 SKP_ADD64(SKP_int64 a, SKP_int64 b){
	SKP_int64 ret;
	ops_count += 2;
	ret = a + b;
	return ret;
}

#undef	SKP_SUB16
SKP_INLINE SKP_int16 SKP_SUB16(SKP_int16 a, SKP_int16 b){
	SKP_int16 ret;
	ops_count += 1;
	ret = a - b;
	return ret;
}

#undef	SKP_SUB32
SKP_INLINE SKP_int32 SKP_SUB32(SKP_int32 a, SKP_int32 b){
	SKP_int32 ret;
	ops_count += 1;
	ret = a - b;
	return ret;
}

#undef	SKP_SUB64
SKP_INLINE SKP_int64 SKP_SUB64(SKP_int64 a, SKP_int64 b){
	SKP_int64 ret;
	ops_count += 2;
	ret = a - b;
	return ret;
}

#undef SKP_ADD_SAT16
SKP_INLINE SKP_int16 SKP_ADD_SAT16( SKP_int16 a16, SKP_int16 b16 ) {
	SKP_int16 res;
	// Nb will be counted in AKP_add32 and SKP_SAT16
	res = (SKP_int16)SKP_SAT16( SKP_ADD32( (SKP_int32)(a16), (b16) ) );
	return res;
}

#undef SKP_ADD_SAT32
SKP_INLINE SKP_int32 SKP_ADD_SAT32(SKP_int32 a32, SKP_int32 b32){
	SKP_int32 res;
	ops_count += 1;
	res =	((((a32) + (b32)) & 0x80000000) == 0 ?									\
			((((a32) & (b32)) & 0x80000000) != 0 ? SKP_int32_MIN : (a32)+(b32)) :	\
			((((a32) | (b32)) & 0x80000000) == 0 ? SKP_int32_MAX : (a32)+(b32)) );
	return res;
}

#undef SKP_ADD_SAT64
SKP_INLINE SKP_int64 SKP_ADD_SAT64( SKP_int64 a64, SKP_int64 b64 ) {
	SKP_int64 res;
	ops_count += 1;
	res =	((((a64) + (b64)) & 0x8000000000000000LL) == 0 ?								\
			((((a64) & (b64)) & 0x8000000000000000LL) != 0 ? SKP_int64_MIN : (a64)+(b64)) :	\
			((((a64) | (b64)) & 0x8000000000000000LL) == 0 ? SKP_int64_MAX : (a64)+(b64)) );
	return res;
}

#undef SKP_SUB_SAT16
SKP_INLINE SKP_int16 SKP_SUB_SAT16( SKP_int16 a16, SKP_int16 b16 ) {
	SKP_int16 res;
	SKP_assert(0);
	// Nb will be counted in sub-macros
	res = (SKP_int16)SKP_SAT16( SKP_SUB32( (SKP_int32)(a16), (b16) ) );
	return res;
}

#undef SKP_SUB_SAT32
SKP_INLINE SKP_int32 SKP_SUB_SAT32( SKP_int32 a32, SKP_int32 b32 ) {
	SKP_int32 res;
	ops_count += 1;
	res = 	((((a32)-(b32)) & 0x80000000) == 0 ?											\
			(( (a32) & ((b32)^0x80000000) & 0x80000000) ? SKP_int32_MIN : (a32)-(b32)) :	\
			((((a32)^0x80000000) & (b32)  & 0x80000000) ? SKP_int32_MAX : (a32)-(b32)) );
	return res;
}

#undef SKP_SUB_SAT64
SKP_INLINE SKP_int64 SKP_SUB_SAT64( SKP_int64 a64, SKP_int64 b64 ) {
	SKP_int64 res;
	ops_count += 1;
	res =	((((a64)-(b64)) & 0x8000000000000000LL) == 0 ?														\
			(( (a64) & ((b64)^0x8000000000000000LL) & 0x8000000000000000LL) ? SKP_int64_MIN : (a64)-(b64)) :	\
			((((a64)^0x8000000000000000LL) & (b64)  & 0x8000000000000000LL) ? SKP_int64_MAX : (a64)-(b64)) );

	return res;
}

#undef	SKP_SMULWW
SKP_INLINE SKP_int32 SKP_SMULWW(SKP_int32 a32, SKP_int32 b32){	
	SKP_int32 ret;
	// Nb will be counted in sub-macros
	ret = SKP_MLA(SKP_SMULWB((a32), (b32)), (a32), SKP_RSHIFT_ROUND((b32), 16));
	return ret;
}

#undef	SKP_SMLAWW
SKP_INLINE SKP_int32 SKP_SMLAWW(SKP_int32 a32, SKP_int32 b32, SKP_int32 c32){	
	SKP_int32 ret;
	// Nb will be counted in sub-macros
	ret = SKP_MLA(SKP_SMLAWB((a32), (b32), (c32)), (b32), SKP_RSHIFT_ROUND((c32), 16));
	return ret;
}

#undef	SKP_min_int
SKP_INLINE SKP_int SKP_min_int(SKP_int a, SKP_int b)
{
	ops_count += 1;
	return (((a) < (b)) ? (a) : (b));
}

#undef	SKP_min_16
SKP_INLINE SKP_int16 SKP_min_16(SKP_int16 a, SKP_int16 b)
{
	ops_count += 1;
	return (((a) < (b)) ? (a) : (b));
}
#undef	SKP_min_32
SKP_INLINE SKP_int32 SKP_min_32(SKP_int32 a, SKP_int32 b)
{
	ops_count += 1;
	return (((a) < (b)) ? (a) : (b));
}
#undef	SKP_min_64
SKP_INLINE SKP_int64 SKP_min_64(SKP_int64 a, SKP_int64 b)
{
	ops_count += 1;
	return (((a) < (b)) ? (a) : (b));
}

/* SKP_min() versions with typecast in the function call */
#undef	SKP_max_int
SKP_INLINE SKP_int SKP_max_int(SKP_int a, SKP_int b)
{
	ops_count += 1;
	return (((a) > (b)) ? (a) : (b));
}
#undef	SKP_max_16
SKP_INLINE SKP_int16 SKP_max_16(SKP_int16 a, SKP_int16 b)
{
	ops_count += 1;
	return (((a) > (b)) ? (a) : (b));
}
#undef	SKP_max_32
SKP_INLINE SKP_int32 SKP_max_32(SKP_int32 a, SKP_int32 b)
{
    ops_count += 1;
    return (((a) > (b)) ? (a) : (b));
}

#undef	SKP_max_64
SKP_INLINE SKP_int64 SKP_max_64(SKP_int64 a, SKP_int64 b)
{
    ops_count += 1;
    return (((a) > (b)) ? (a) : (b));
}


#undef SKP_LIMIT_int
SKP_INLINE SKP_int SKP_LIMIT_int(SKP_int a, SKP_int limit1, SKP_int limit2)
{
    SKP_int ret;
    ops_count += 6;

    ret = ((limit1) > (limit2) ? ((a) > (limit1) ? (limit1) : ((a) < (limit2) ? (limit2) : (a))) \
        : ((a) > (limit2) ? (limit2) : ((a) < (limit1) ? (limit1) : (a))));

    return(ret);
}

#undef SKP_LIMIT_16
SKP_INLINE SKP_int16 SKP_LIMIT_16(SKP_int16 a, SKP_int16 limit1, SKP_int16 limit2)
{
    SKP_int16 ret;
    ops_count += 6;

    ret = ((limit1) > (limit2) ? ((a) > (limit1) ? (limit1) : ((a) < (limit2) ? (limit2) : (a))) \
        : ((a) > (limit2) ? (limit2) : ((a) < (limit1) ? (limit1) : (a))));

return(ret);
}


#undef SKP_LIMIT_32
SKP_INLINE SKP_int SKP_LIMIT_32(SKP_int32 a, SKP_int32 limit1, SKP_int32 limit2)
{
    SKP_int32 ret;
    ops_count += 6;

    ret = ((limit1) > (limit2) ? ((a) > (limit1) ? (limit1) : ((a) < (limit2) ? (limit2) : (a))) \
        : ((a) > (limit2) ? (limit2) : ((a) < (limit1) ? (limit1) : (a))));
    return(ret);
}

#else
#define exVarDefine
#define varDefine
#define SKP_SaveCount()

#endif
#endif

