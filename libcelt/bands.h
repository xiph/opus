/* (C) 2007 Jean-Marc Valin, CSIRO
*/
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

/** Applies a series of rotations so that pulses are spread like a two-sided
exponential. The effect of this is to reduce the tonal noise created by the
sparse spectrum resulting from the pulse codebook */
void exp_rotation(celt_norm_t *X, int len, int dir, int stride, int iter);

/** Compute the amplitude (sqrt energy) in each of the bands 
 * @param m Mode data 
 * @param X Spectrum
 * @param bands Square root of the energy for each band (returned)
 */
void compute_band_energies(const CELTMode *m, const celt_sig_t *X, celt_ener_t *bands);

void compute_noise_energies(const CELTMode *m, const celt_sig_t *X, const celt_word16_t *tonality, celt_ener_t *bank);

/** Normalise each band of X such that the energy in each band is 
    equal to 1
 * @param m Mode data 
 * @param X Spectrum (returned normalised)
 * @param bands Square root of the energy for each band
 */
void normalise_bands(const CELTMode *m, const celt_sig_t * restrict freq, celt_norm_t * restrict X, const celt_ener_t *bands);

void renormalise_bands(const CELTMode *m, celt_norm_t * restrict X);

/** Denormalise each band of X to restore full amplitude
 * @param m Mode data 
 * @param X Spectrum (returned de-normalised)
 * @param bands Square root of the energy for each band
 */
void denormalise_bands(const CELTMode *m, const celt_norm_t * restrict X, celt_sig_t * restrict freq, const celt_ener_t *bands);

/** Compute the pitch predictor gain for each pitch band
 * @param m Mode data 
 * @param X Spectrum to predict
 * @param P Pitch vector (normalised)
 * @param gains Gain computed for each pitch band (returned)
 * @param bank Square root of the energy for each band
 */
void compute_pitch_gain(const CELTMode *m, const celt_norm_t *X, const celt_norm_t *P, celt_pgain_t *gains);

void pitch_quant_bands(const CELTMode *m, celt_norm_t * restrict P, const celt_pgain_t * restrict gains);

/** Quantisation/encoding of the residual spectrum
 * @param m Mode data 
 * @param X Residual (normalised)
 * @param P Pitch vector (normalised)
 * @param W Perceptual weighting
 * @param total_bits Total number of bits that can be used for the frame (including the ones already spent)
 * @param enc Entropy encoder
 */
void quant_bands(const CELTMode *m, celt_norm_t * restrict X, celt_norm_t *P, celt_mask_t *W, const celt_ener_t *bandE, const int *stereo_mode, int *pulses, int time_domain, int fold, int total_bits, ec_enc *enc);

/** Decoding of the residual spectrum
 * @param m Mode data 
 * @param X Residual (normalised)
 * @param P Pitch vector (normalised)
 * @param total_bits Total number of bits that can be used for the frame (including the ones already spent)
 * @param dec Entropy decoder
*/
void unquant_bands(const CELTMode *m, celt_norm_t * restrict X, celt_norm_t *P, const celt_ener_t *bandE, const int *stereo_mode, int *pulses, int time_domain, int fold, int total_bits, ec_dec *dec);

void stereo_decision(const CELTMode *m, celt_norm_t * restrict X, int *stereo_mode, int len);

#endif /* BANDS_H */
