/* Copyright (c) 2007-2008 CSIRO
   Copyright (c) 2007-2009 Xiph.Org Foundation
   Copyright (c) 2008-2009 Gregory Maxwell 
   Written by Jean-Marc Valin and Gregory Maxwell */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:
   
   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
   
   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
   
   - Neither the name of the Xiph.org Foundation nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.
   
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

#ifndef BANDS_H
#define BANDS_H

#include "arch.h"
#include "modes.h"
#include "entenc.h"
#include "entdec.h"
#include "rate.h"

/** Compute the amplitude (sqrt energy) in each of the bands 
 * @param m Mode data 
 * @param X Spectrum
 * @param bands Square root of the energy for each band (returned)
 */
void compute_band_energies(const CELTMode *m, const celt_sig *X, celt_ener *bands, int _C);

/*void compute_noise_energies(const CELTMode *m, const celt_sig *X, const celt_word16 *tonality, celt_ener *bank);*/

/** Normalise each band of X such that the energy in each band is 
    equal to 1
 * @param m Mode data 
 * @param X Spectrum (returned normalised)
 * @param bands Square root of the energy for each band
 */
void normalise_bands(const CELTMode *m, const celt_sig * restrict freq, celt_norm * restrict X, const celt_ener *bands, int _C);

void renormalise_bands(const CELTMode *m, celt_norm * restrict X, int _C);

/** Denormalise each band of X to restore full amplitude
 * @param m Mode data 
 * @param X Spectrum (returned de-normalised)
 * @param bands Square root of the energy for each band
 */
void denormalise_bands(const CELTMode *m, const celt_norm * restrict X, celt_sig * restrict freq, const celt_ener *bands, int _C);

/** Compute the pitch predictor gain for each pitch band
 * @param m Mode data 
 * @param X Spectrum to predict
 * @param P Pitch vector (normalised)
 * @param gains Gain computed for each pitch band (returned)
 * @param bank Square root of the energy for each band
 */
int compute_pitch_gain(const CELTMode *m, const celt_sig *X, const celt_sig *P, int norm_rate, int *gain_id, int _C, celt_word16 *gain_prod);

void apply_pitch(const CELTMode *m, celt_sig *X, const celt_sig *P, int gain_id, int pred, int _C);

int folding_decision(const CELTMode *m, celt_norm *X, celt_word16 *average, int *last_decision, int _C);

/** Quantisation/encoding of the residual spectrum
 * @param m Mode data 
 * @param X Residual (normalised)
 * @param total_bits Total number of bits that can be used for the frame (including the ones already spent)
 * @param enc Entropy encoder
 */
void quant_bands(const CELTMode *m, celt_norm * restrict X, const celt_ener *bandE, int *pulses, int time_domain, int fold, int total_bits, int encode, void *enc_dec);

void quant_bands_stereo(const CELTMode *m, celt_norm * restrict X, const celt_ener *bandE, int *pulses, int time_domain, int fold, int total_bits, ec_enc *enc);

/** Decoding of the residual spectrum
 * @param m Mode data 
 * @param X Residual (normalised)
 * @param total_bits Total number of bits that can be used for the frame (including the ones already spent)
 * @param dec Entropy decoder
*/
void unquant_bands(const CELTMode *m, celt_norm * restrict X, const celt_ener *bandE, int *pulses, int time_domain, int fold, int total_bits, ec_dec *dec);

void unquant_bands_stereo(const CELTMode *m, celt_norm * restrict X, const celt_ener *bandE, int *pulses, int time_domain, int fold, int total_bits, ec_dec *dec);

void stereo_decision(const CELTMode *m, celt_norm * restrict X, int *stereo_mode, int len);

#endif /* BANDS_H */
