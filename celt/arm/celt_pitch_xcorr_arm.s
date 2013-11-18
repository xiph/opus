; Copyright (c) 2007-2008 CSIRO
; Copyright (c) 2007-2009 Xiph.Org Foundation
; Copyright (c) 2013      Parrot
; Written by Aurélien Zanelli
;
; Redistribution and use in source and binary forms, with or without
; modification, are permitted provided that the following conditions
; are met:
;
; - Redistributions of source code must retain the above copyright
; notice, this list of conditions and the following disclaimer.
;
; - Redistributions in binary form must reproduce the above copyright
; notice, this list of conditions and the following disclaimer in the
; documentation and/or other materials provided with the distribution.
;
; THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
; ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
; LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
; A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
; OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
; EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
; PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
; PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
; LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
; NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
; SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  AREA  |.text|, CODE, READONLY

  GET    celt/arm/armopts.s

IF OPUS_ARM_MAY_HAVE_EDSP
  EXPORT celt_pitch_xcorr_edsp
ENDIF

IF OPUS_ARM_MAY_HAVE_NEON
  EXPORT celt_pitch_xcorr_neon
ENDIF

IF OPUS_ARM_MAY_HAVE_NEON

;; Compute sum[k]=sum(x[j]*y[j+k],j=0...len-1), k=0...3
;xcorr_kernel_neon PROC
;  ; input:
;  ;   r3     = int         len
;  ;   r4     = opus_val16 *x
;  ;   r5     = opus_val16 *y
;  ;   q0     = opus_val32  sum[4]
;  ; output:
;  ;   q0     = opus_val32  sum[4]
;  ; preserved: r0-r3, r6-r11, d2, q4-q7, q9-q15
;  ; internal usage:
;  ;   r12 = int j
;  ;   d3  = y_3|y_2|y_1|y_0
;  ;   q2  = y_B|y_A|y_9|y_8|y_7|y_6|y_5|y_4
;  ;   q3  = x_7|x_6|x_5|x_4|x_3|x_2|x_1|x_0
;  ;   q8  = scratch
;  ;
;  ; Load y[0...3]
;  ; This requires len>0 to always be valid (which we assert in the C code).
;  VLD1.16      {d5}, [r5]!
;  SUBS         r12, r3, #8
;  BLE xcorr_kernel_neon_process4
;; Process 8 samples at a time.
;; This loop loads one y value more than we actually need. Therefore we have to
;; stop as soon as there are 8 or fewer samples left (instead of 7), to avoid
;; reading past the end of the array.
;xcorr_kernel_neon_process8
;  ; This loop has 19 total instructions (10 cycles to issue, minimum), with
;  ; - 2 cycles of ARM insrtuctions,
;  ; - 10 cycles of load/store/byte permute instructions, and
;  ; - 9 cycles of data processing instructions.
;  ; On a Cortex A8, we dual-issue the maximum amount (9 cycles) between the
;  ; latter two categories, meaning the whole loop should run in 10 cycles per
;  ; iteration, barring cache misses.
;  ;
;  ; Load x[0...7]
;  VLD1.16      {d6, d7}, [r4]!
;  ; Unlike VMOV, VAND is a data processsing instruction (and doesn't get
;  ; assembled to VMOV, like VORR would), so it dual-issues with the prior VLD1.
;  VAND         d3, d5, d5
;  SUBS         r12, r12, #8
;  ; Load y[4...11]
;  VLD1.16      {d4, d5}, [r5]!
;  VMLAL.S16    q0, d3, d6[0]
;  VEXT.16      d16, d3, d4, #1
;  VMLAL.S16    q0, d4, d7[0]
;  VEXT.16      d17, d4, d5, #1
;  VMLAL.S16    q0, d16, d6[1]
;  VEXT.16      d16, d3, d4, #2
;  VMLAL.S16    q0, d17, d7[1]
;  VEXT.16      d17, d4, d5, #2
;  VMLAL.S16    q0, d16, d6[2]
;  VEXT.16      d16, d3, d4, #3
;  VMLAL.S16    q0, d17, d7[2]
;  VEXT.16      d17, d4, d5, #3
;  VMLAL.S16    q0, d16, d6[3]
;  VMLAL.S16    q0, d17, d7[3]
;  BGT xcorr_kernel_neon_process8
;; Process 4 samples here if we have > 4 left (still reading one extra y value).
;xcorr_kernel_neon_process4
;  ADDS         r12, r12, #4
;  BLE xcorr_kernel_neon_process2
;  ; Load x[0...3]
;  VLD1.16      d6, [r4]!
;  ; Use VAND since it's a data processing instruction again.
;  VAND         d4, d5, d5
;  SUB          r12, r12, #4
;  ; Load y[4...7]
;  VLD1.16      d5, [r5]!
;  VMLAL.S16    q0, d4, d6[0]
;  VEXT.16      d16, d4, d5, #1
;  VMLAL.S16    q0, d16, d6[1]
;  VEXT.16      d16, d4, d5, #2
;  VMLAL.S16    q0, d16, d6[2]
;  VEXT.16      d16, d4, d5, #3
;  VMLAL.S16    q0, d16, d6[3]
;; Process 2 samples here if we have > 2 left (still reading one extra y value).
;xcorr_kernel_neon_process2
;  ADDS         r12, r12, #2
;  BLE xcorr_kernel_neon_process1
;  ; Load x[0...1]
;  VLD2.16      {d6[],d7[]}, [r4]!
;  ; Use VAND since it's a data processing instruction again.
;  VAND         d4, d5, d5
;  SUB          r12, r12, #2
;  ; Load y[4...5]
;  VLD1.32      {d5[]}, [r5]!
;  VMLAL.S16    q0, d4, d6
;  VEXT.16      d16, d4, d5, #1
;  ; Replace bottom copy of {y5,y4} in d5 with {y3,y2} from d4, using VSRI
;  ; instead of VEXT, since it's a data-processing instruction.
;  VSRI.64      d5, d4, #32
;  VMLAL.S16    q0, d16, d7
;; Process 1 sample using the extra y value we loaded above.
;xcorr_kernel_neon_process1
;  ; Load next *x
;  VLD1.16      {d6[]}, [r4]!
;  ADDS         r12, r12, #1
;  ; y[0...3] are left in d5 from prior iteration(s) (if any)
;  VMLAL.S16    q0, d5, d6
;  MOVLE        pc, lr
;; Now process 1 last sample, not reading ahead.
;  ; Load last *y
;  VLD1.16      {d4[]}, [r5]!
;  VSRI.64      d4, d5, #16
;  ; Load last *x
;  VLD1.16      {d6[]}, [r4]!
;  VMLAL.S16    q0, d4, d6
;  MOV          pc, lr
;  ENDP

;; opus_val32 celt_pitch_xcorr_neon(opus_val16 *_x, opus_val16 *_y,
;;  opus_val32 *xcorr, int len, int max_pitch)
;celt_pitch_xcorr_neon PROC
;  ; input:
;  ;   r0  = opus_val16 *_x
;  ;   r1  = opus_val16 *_y
;  ;   r2  = opus_val32 *xcorr
;  ;   r3  = int         len
;  ; output:
;  ;   r0  = int         maxcorr
;  ; internal usage:
;  ;   r4  = opus_val16 *x (for xcorr_kernel_neon())
;  ;   r5  = opus_val16 *y (for xcorr_kernel_neon())
;  ;   r6  = int         max_pitch
;  ;   r12 = int         j
;  ;   q15 = int         maxcorr[4] (q15 is not used by xcorr_kernel_neon())
;  STMFD        sp!, {r4-r6, lr}
;  LDR          r6, [sp, #16]
;  VMOV.S32     q15, #1
;  ; if (max_pitch < 4) goto celt_pitch_xcorr_neon_process4_done
;  SUBS         r6, r6, #4
;  BLT celt_pitch_xcorr_neon_process4_done
;celt_pitch_xcorr_neon_process4
;  ; xcorr_kernel_neon parameters:
;  ; r3 = len, r4 = _x, r5 = _y, q0 = {0, 0, 0, 0}
;  MOV          r4, r0
;  MOV          r5, r1
;  VEOR         q0, q0, q0
;  ; xcorr_kernel_neon only modifies r4, r5, r12, and q0...q3.
;  ; So we don't save/restore any other registers.
;  BL xcorr_kernel_neon
;  SUBS         r6, r6, #4
;  VST1.32      {q0}, [r2]!
;  ; _y += 4
;  ADD          r1, r1, #8
;  VMAX.S32     q15, q15, q0
;  ; if (max_pitch < 4) goto celt_pitch_xcorr_neon_process4_done
;  BGE celt_pitch_xcorr_neon_process4
;; We have less than 4 sums left to compute.
;celt_pitch_xcorr_neon_process4_done
;  ADDS         r6, r6, #4
;  ; Reduce maxcorr to a single value
;  VMAX.S32     d30, d30, d31
;  VPMAX.S32    d30, d30, d30
;  ; if (max_pitch <= 0) goto celt_pitch_xcorr_neon_done
;  BLE celt_pitch_xcorr_neon_done
;; Now compute each remaining sum one at a time.
;celt_pitch_xcorr_neon_process_remaining
;  MOV          r4, r0
;  MOV          r5, r1
;  VMOV.I32     q0, #0
;  SUBS         r12, r3, #8
;  BLT celt_pitch_xcorr_neon_process_remaining4
;; Sum terms 8 at a time.
;celt_pitch_xcorr_neon_process_remaining_loop8
;  ; Load x[0...7]
;  VLD1.16      {q1}, [r4]!
;  ; Load y[0...7]
;  VLD1.16      {q2}, [r5]!
;  SUBS         r12, r12, #8
;  VMLAL.S16    q0, d4, d2
;  VMLAL.S16    q0, d5, d3
;  BGE celt_pitch_xcorr_neon_process_remaining_loop8
;; Sum terms 4 at a time.
;celt_pitch_xcorr_neon_process_remaining4
;  ADDS         r12, r12, #4
;  BLT celt_pitch_xcorr_neon_process_remaining4_done
;  ; Load x[0...3]
;  VLD1.16      {d2}, [r4]!
;  ; Load y[0...3]
;  VLD1.16      {d3}, [r5]!
;  SUB          r12, r12, #4
;  VMLAL.S16    q0, d3, d2
;  ; Reduce the sum to a single value.
;  VADD.S32     d0, d0, d1
;  VPADDL.S32   d0, d0
;celt_pitch_xcorr_neon_process_remaining4_done
;  ADDS         r12, r12, #4
;  BLE celt_pitch_xcorr_neon_process_remaining_loop_done
;; Sum terms 1 at a time.
;celt_pitch_xcorr_neon_process_remaining_loop1
;  VLD1.16      {d2[]}, [r4]!
;  VLD1.16      {d3[]}, [r5]!
;  SUBS         r12, r12, #1
;  VMLAL.S16    q0, d2, d3
;  BGT celt_pitch_xcorr_neon_process_remaining_loop1
;celt_pitch_xcorr_neon_process_remaining_loop_done
;  VST1.32      {d0[0]}, [r2]!
;  VMAX.S32     d30, d30, d0
;  SUBS         r6, r6, #1
;  ; _y++
;  ADD          r1, r1, #2
;  ; if (--max_pitch > 0) goto celt_pitch_xcorr_neon_process_remaining
;  BGT celt_pitch_xcorr_neon_process_remaining
;celt_pitch_xcorr_neon_done
;  VMOV.32      r0, d30[0]
;  LDMFD        sp!, {r4-r6, pc}
;  ENDP

xcorr_kernel_neon PROC
  ; input:
  ; r0 = opus_val16 *x
  ; r1 = opus_val16 *y
  ; r2 = int        len
  ; q0 = opus_val32 sum (sum[3] | sum[2] | sum[1] | sum[0])

  ; output:
  ; q0 = sum

  ; internal usage:
  ; r3 = j
  ; d2 = x_3|x_2|x_1|x_0  d3 = y_3|y_2|y_1|y_0
  ; d4 = y_7|y_6|y_5|y_4  d5 = y_4|y_3|y_2|y_1
  ; d6 = y_5|y_4|y_3|y_2  d7 = y_6|y_5|y_4|y_3
  ; We will build d5, d6 and d7 vector from d3 and d4


  VLD1.16   {d3}, [r1]!      ; Load y[3] downto y[0] to d3 lane (yy0)
  SUB       r3, r2, #1
  MOVS      r3, r3, lsr #2   ; j=(len-1)>>2
  BEQ       xcorr_kernel_neon_process4_done

  ; Process 4 x samples at a time
  ; For this, we will need 4 y vectors
xcorr_kernel_neon_process4
  SUBS      r3, r3, #1       ; j--
  VLD1.16   d4, [r1]!        ; Load y[7] downto y[4] to d4 lane
  VLD1.16   d2, [r0]!        ; Load x[3] downto x[0] to d2 lane
  VEXT.16   d5, d3, d4, #1   ; Build y[4] downto y[1] vector (yy1)
  VEXT.16   d6, d3, d4, #2   ; Build y[5] downto y[2] vector (yy2)
  VEXT.16   d7, d3, d4, #3   ; Build y[6] downto y[3] vector (yy3)

  VMLAL.S16 q0, d3, d2[0]    ; MAC16_16(sum, x[0], yy0)
  VMLAL.S16 q0, d5, d2[1]    ; MAC16_16(sum, x[1], yy1)
  VMLAL.S16 q0, d6, d2[2]    ; MAC16_16(sum, x[2], yy2)
  VMLAL.S16 q0, d7, d2[3]    ; MAC16_16(sum, x[3], yy3)

  VMOV.S16  d3, d4           ; Next y vector should be in d3 (yy0)

  BNE xcorr_kernel_neon_process4

xcorr_kernel_neon_process4_done
  ;Process len-1 to len
  VLD1.16   {d2[]}, [r0]!    ; Load *x and duplicate to d2 lane

  SUB       r3, r2, #1
  ANDS      r3, r3, #3       ; j=(len-1)&3
  VMLAL.S16 q0, d3, d2       ; MAC16_16(sum, *x, yy0)
  BEQ xcorr_kernel_neon_done

xcorr_kernel_neon_process_remaining
  SUBS      r3, r3, #1       ; j--
  VLD1.16   {d4[]}, [r1]!    ; Load y value and duplicate to d4 lane
  VLD1.16   {d2[]}, [r0]!    ; Load *x and duplicate to d2 lane
  VEXT.16   d3, d3, d4, #1   ; Build y vector from previous and d4
  VMLAL.S16 q0, d3, d2       ; MAC16_16(sum, *x, yy0)
  BNE xcorr_kernel_neon_process_remaining

xcorr_kernel_neon_done
  MOV       pc, lr
  ENDP

celt_pitch_xcorr_neon PROC
  ; input:
  ; r0 = opus_val16 *_x
  ; r1 = opus_val16 *_y
  ; r2 = opus_val32 *xcorr
  ; r3 = int        len

  ; output:
  ; r0 = maxcorr

  STMFD     sp!, {r4-r9, lr}

  LDR       r4, [sp, #28]        ; r4 = int max_pitch
  MOV       r5, r0               ; r5 = _x
  MOV       r6, r1               ; r6 = _y
  MOV       r7, r2               ; r7 = xcorr
  MOV       r2, r3               ; r2 = len

  VMOV.S32  d16, #1              ; d16 = {1, 1}  (not used by xcorr_kernel_neon)
  MOV       r8, #0               ; r8 = i = 0
  CMP       r4, #3               ; max_pitch-3 <= 0  ---> pitch_xcorr_neon_process4_done
  BLE       celt_pitch_xcorr_neon_process4_done

  SUB       r9, r4, #3           ; r9 = max_pitch-3

celt_pitch_xcorr_neon_process4
  MOV       r0, r5               ; r0 = _x
  ADD       r1, r6 ,r8, LSL #1   ; r1 = _y + i
  VMOV.I32  q0, #0               ; q0 = opus_val32 sum[4] = {0, 0, 0, 0}

                                 ; xcorr_kernel_neon don't touch r2 (len)
                                 ; So we don't store it
  BL xcorr_kernel_neon           ; xcorr_kernel_neon(_x, _y+i, sum, len)

  VST1.32   {q0}, [r7]!          ; Store sum to xcorr
  VPMAX.S32 d0, d0, d1           ; d0 = max(sum[3], sum[2]) | max(sum[1], sum[0])
  ADD       r8, r8, #4           ; i+=4
  VPMAX.S32 d0, d0, d0           ; d0 = max(sum[3], sum[2], sum[1], sum[0])
  CMP       r8, r9               ; i < max_pitch-3 ----> pitch_xcorr_neon_process4
  VMAX.S32  d16, d16, d0         ; d16 = maxcorr = max(maxcorr, sum)

  BLT       celt_pitch_xcorr_neon_process4

celt_pitch_xcorr_neon_process4_done
  CMP       r8, r4;
  BGE       celt_pitch_xcorr_neon_done

celt_pitch_xcorr_neon_process_remaining
  MOV       r0, r5               ; r0 = _x
  ADD       r1, r6, r8, LSL #1   ; r1 = _y + i
  VMOV.I32  q0, #0
  MOVS      r3, r2, LSR #2       ; r3 = j = len
  BEQ       inner_loop_neon_process4_done

inner_loop_neon_process4
  VLD1.16   {d2}, [r0]!          ; Load x
  VLD1.16   {d3}, [r1]!          ; Load y
  SUBS      r3, r3, #1
  VMLAL.S16 q0, d2, d3
  BNE       inner_loop_neon_process4

  VPADD.S32 d0, d0, d1          ; Reduce sum
  VPADD.S32 d0, d0, d0

inner_loop_neon_process4_done
  ANDS      r3, r2, #3
  BEQ       inner_loop_neon_done

inner_loop_neon_process_remaining
  VLD1.16   {d2[]}, [r0]!
  VLD1.16   {d3[]}, [r1]!
  SUBS      r3, r3, #1
  VMLAL.S16 q0, d2, d3
  BNE       inner_loop_neon_process_remaining

inner_loop_neon_done
  VST1.32   {d0[0]}, [r7]!
  VMAX.S32  d16, d16, d0

  ADD       r8, r8, #1
  CMP       r8, r4
  BCC       celt_pitch_xcorr_neon_process_remaining

celt_pitch_xcorr_neon_done
  VMOV      d0, d16
  VMOV.32   r0, d0[0]
  LDMFD     sp!, {r4-r9, pc}
  ENDP


ENDIF

IF OPUS_ARM_MAY_HAVE_EDSP

; This will get used on ARMv7 devices without NEON, so it has been optimized
; to take advantage of dual-issuing where possible.
xcorr_kernel_edsp PROC
  ; input:
  ;   r3      = int         len
  ;   r4      = opus_val16 *_x
  ;   r5      = opus_val16 *_y
  ;   r6...r9 = opus_val32  sum[4]
  ; output:
  ;   r6...r9 = opus_val32  sum[4]
  ; preserved: r0-r5
  ; internal usage
  ;   r2      = int         j
  ;   r12,r14 = opus_val16  x[4]
  ;   r10,r11 = opus_val16  y[4]
  STMFD        sp!, {r2,r4,r5,lr}
  SUBS         r2, r3, #4         ; j = len-4
  LDRD         r10, r11, [r5], #8 ; Load y[0...3]
  BLE xcorr_kernel_edsp_process4_done
  LDR          r12, [r4], #4      ; Load x[0...1]
  ; Stall
xcorr_kernel_edsp_process4
  ; The multiplies must issue from pipeline 0, and can't dual-issue with each
  ; other. Every other instruction here dual-issues with a multiply, and is
  ; thus "free". There should be no stalls in the body of the loop.
  SMLABB       r6, r12, r10, r6   ; sum[0] = MAC16_16(sum[0],x_0,y_0)
  LDR          r14, [r4], #4      ; Load x[2...3]
  SMLABT       r7, r12, r10, r7   ; sum[1] = MAC16_16(sum[1],x_0,y_1)
  SUBS         r2, r2, #4         ; j-=4
  SMLABB       r8, r12, r11, r8   ; sum[2] = MAC16_16(sum[2],x_0,y_2)
  SMLABT       r9, r12, r11, r9   ; sum[3] = MAC16_16(sum[3],x_0,y_3)
  SMLATT       r6, r12, r10, r6   ; sum[0] = MAC16_16(sum[0],x_1,y_1)
  LDR          r10, [r5], #4      ; Load y[4...5]
  SMLATB       r7, r12, r11, r7   ; sum[1] = MAC16_16(sum[1],x_1,y_2)
  SMLATT       r8, r12, r11, r8   ; sum[2] = MAC16_16(sum[2],x_1,y_3)
  SMLATB       r9, r12, r10, r9   ; sum[3] = MAC16_16(sum[3],x_1,y_4)
  LDRGT        r12, [r4], #4      ; Load x[0...1]
  SMLABB       r6, r14, r11, r6   ; sum[0] = MAC16_16(sum[0],x_2,y_2)
  SMLABT       r7, r14, r11, r7   ; sum[1] = MAC16_16(sum[1],x_2,y_3)
  SMLABB       r8, r14, r10, r8   ; sum[2] = MAC16_16(sum[2],x_2,y_4)
  SMLABT       r9, r14, r10, r9   ; sum[3] = MAC16_16(sum[3],x_2,y_5)
  SMLATT       r6, r14, r11, r6   ; sum[0] = MAC16_16(sum[0],x_3,y_3)
  LDR          r11, [r5], #4      ; Load y[6...7]
  SMLATB       r7, r14, r10, r7   ; sum[1] = MAC16_16(sum[1],x_3,y_4)
  SMLATT       r8, r14, r10, r8   ; sum[2] = MAC16_16(sum[2],x_3,y_5)
  SMLATB       r9, r14, r11, r9   ; sum[3] = MAC16_16(sum[3],x_3,y_6)
  BGT xcorr_kernel_edsp_process4
xcorr_kernel_edsp_process4_done
  ADDS         r2, r2, #4
  BLE xcorr_kernel_edsp_done
  LDRH         r12, [r4], #2      ; r12 = *x++
  SUBS         r2, r2, #1         ; j--
  ; Stall
  SMLABB       r6, r12, r10, r6   ; sum[0] = MAC16_16(sum[0],x,y_0)
  LDRGTH       r14, [r4], #2      ; r14 = *x++
  SMLABT       r7, r12, r10, r7   ; sum[1] = MAC16_16(sum[1],x,y_1)
  SMLABB       r8, r12, r11, r8   ; sum[2] = MAC16_16(sum[2],x,y_2)
  SMLABT       r9, r12, r11, r9   ; sum[3] = MAC16_16(sum[3],x,y_3)
  BLE xcorr_kernel_edsp_done
  SMLABT       r6, r14, r10, r6   ; sum[0] = MAC16_16(sum[0],x,y_1)
  SUBS         r2, r2, #1         ; j--
  SMLABB       r7, r14, r11, r7   ; sum[1] = MAC16_16(sum[1],x,y_2)
  LDRH         r10, [r5], #2      ; r10 = y_4 = *y++
  SMLABT       r8, r14, r11, r8   ; sum[2] = MAC16_16(sum[2],x,y_3)
  LDRGTH       r12, [r4], #2      ; r12 = *x++
  SMLABB       r9, r14, r10, r9   ; sum[3] = MAC16_16(sum[3],x,y_4)
  BLE xcorr_kernel_edsp_done
  SMLABB       r6, r12, r11, r6   ; sum[0] = MAC16_16(sum[0],tmp,y_2)
  CMP          r2, #1             ; j--
  SMLABT       r7, r12, r11, r7   ; sum[1] = MAC16_16(sum[1],tmp,y_3)
  LDRH         r2, [r5], #2       ; r2 = y_5 = *y++
  SMLABB       r8, r12, r10, r8   ; sum[2] = MAC16_16(sum[2],tmp,y_4)
  LDRGTH       r14, [r4]          ; r14 = *x
  SMLABB       r9, r12, r2, r9    ; sum[3] = MAC16_16(sum[3],tmp,y_5)
  BLE xcorr_kernel_edsp_done
  SMLABT       r6, r14, r11, r6   ; sum[0] = MAC16_16(sum[0],tmp,y_3)
  LDRH         r11, [r5]          ; r11 = y_6 = *y
  SMLABB       r7, r14, r10, r7   ; sum[1] = MAC16_16(sum[1],tmp,y_4)
  SMLABB       r8, r14, r2, r8    ; sum[2] = MAC16_16(sum[2],tmp,y_5)
  SMLABB       r9, r14, r11, r9   ; sum[3] = MAC16_16(sum[3],tmp,y_6)
xcorr_kernel_edsp_done
  LDMFD        sp!, {r2,r4,r5,pc}
  ENDP

celt_pitch_xcorr_edsp PROC
  ; input:
  ;   r0  = opus_val16 *_x
  ;   r1  = opus_val16 *_y
  ;   r2  = opus_val32 *xcorr
  ;   r3  = int         len
  ; output:
  ;   r0  = maxcorr
  ; internal usage
  ;   r4  = opus_val16 *x
  ;   r5  = opus_val16 *y
  ;   r6  = opus_val32  sum0
  ;   r7  = opus_val32  sum1
  ;   r8  = opus_val32  sum2
  ;   r9  = opus_val32  sum3
  ;   r1  = int         max_pitch
  ;   r12 = int         j
  STMFD        sp!, {r4-r11, lr}
  MOV          r5, r1
  LDR          r1, [sp, #36]
  MOV          r4, r0
  ; maxcorr = 1
  MOV          r0, #1
  ; if (max_pitch < 4) goto celt_pitch_xcorr_edsp_process4_done
  SUBS         r1, r1, #4
  BLT celt_pitch_xcorr_edsp_process4_done
celt_pitch_xcorr_edsp_process4
  ; xcorr_kernel_edsp parameters:
  ; r3 = len, r4 = _x, r5 = _y, r6...r9 = sum[4] = {0, 0, 0, 0}
  MOV          r6, #0
  MOV          r7, #0
  MOV          r8, #0
  MOV          r9, #0
  BL xcorr_kernel_edsp  ; xcorr_kernel_edsp(_x, _y+i, xcorr+i, len)
  ; maxcorr = max(maxcorr, sum0, sum1, sum2, sum3)
  CMP          r0, r6
  ; _y+=4
  ADD          r5, r5, #8
  MOVLT        r0, r6
  CMP          r0, r7
  STRD         r6, r7, [r2], #8
  MOVLT        r0, r7
  CMP          r0, r8
  STRD         r8, r9, [r2], #8
  MOVLT        r0, r8
  CMP          r0, r9
  MOVLT        r0, r9
  SUBS         r1, r1, #4
  BGE celt_pitch_xcorr_edsp_process4
celt_pitch_xcorr_edsp_process4_done
  ADDS         r1, r1, #4
  BLE celt_pitch_xcorr_edsp_done
; Now compute each remaining sum one at a time.
celt_pitch_xcorr_edsp_process_remaining
  SUBS         r12, r3, #4
  ; r14 = sum = 0
  MOV          r14, #0
  BLT celt_pitch_xcorr_edsp_process_remaining_loop_done
  LDRD         r6, r7, [r4], #8
  LDRD         r8, r9, [r5], #8
  ; Stall
celt_pitch_xcorr_edsp_process_remaining_loop4
  SMLABB       r14, r6, r8, r14     ; sum = MAC16_16(sum, x_0, y_0)
  SUBS         r12, r12, #4         ; j--
  SMLATT       r14, r6, r8, r14     ; sum = MAC16_16(sum, x_1, y_1)
  LDRGE        r6, [r4], #4
  SMLABB       r14, r7, r9, r14     ; sum = MAC16_16(sum, x_2, y_2)
  LDRGE        r8, [r5], #4
  SMLATT       r14, r7, r9, r14     ; sum = MAC16_16(sum, x_3, y_3)
  LDRGE        r7, [r4], #4
  LDRGE        r9, [r5], #4
  BGE celt_pitch_xcorr_edsp_process_remaining_loop4
celt_pitch_xcorr_edsp_process_remaining_loop_done
  ADDS         r12, r12, #2
  LDRGE        r6, [r4], #4
  LDRGE        r8, [r5], #4
  ; Stall
  SMLABBGE     r14, r6, r8, r14     ; sum = MAC16_16(sum, x_0, y_0)
  SUBGE        r12, r12, #2
  SMLATTGE     r14, r6, r8, r14     ; sum = MAC16_16(sum, x_1, y_1)
  ADDS         r12, r12, #1
  LDRGEH       r6, [r4], #2
  LDRGEH       r8, [r5], #2
  ; Restore _x
  SUB          r4, r4, r3, LSL #1
  ; Stall
  SMLABBGE     r14, r6, r8, r14     ; sum = MAC16_16(sum, *x, *y)
  ; Restore and advance _y
  SUB          r5, r5, r3, LSL #1
  ; maxcorr = max(maxcorr, sum)
  ; Stall
  CMP          r0, r14
  ADD          r5, r5, #2
  MOVLT        r0, r14
  SUBS         r1, r1, #1
  ; xcorr[i] = sum
  STR          r14, [r2], #4
  BGT celt_pitch_xcorr_edsp_process_remaining
celt_pitch_xcorr_edsp_done
  LDMFD        sp!, {r4-r11, pc}
  ENDP

ENDIF

END
