"""
/* Copyright (c) 2023 Amazon
   Written by Jan Buethe */
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
"""

import torch

def calculate_phase_features(x, periods, frame_size=160, ncoeffs=10, constant_term=False):
    """ calculates fourier expansion coefficients

    Args:
        x (torch.tensor): signal of shape (batch_size, *, N*frame_size)
        periods (torch.tensor): periods of shape (batch_size, num_frames)
        frame_size (int, optional): _description_. Defaults to 160.
        ncoeffs (int, optional): number of coefficients. Defaults to 10.

    Returns:
        torch.tensor: absolute values in [...,:ncoeffs], real part of normalized
        coefficients in [...,ncoeffs : 2*ncoeffs] and imaginary part of normalized
        coefficients in [...,2*ncoeffs : 3*ncoeffs]
    """

    batch_size = x.size(0)
    num_frames = x.size(-1) // frame_size

    x_frames = x.reshape(batch_size, num_frames, frame_size)
    x_past_frames = torch.cat(
        (torch.zeros_like(x_frames[..., 0:1, :]),
         x_frames[..., :-1, :]),
        dim=1
    )

    frames = torch.cat((x_past_frames, x_frames), dim=-1)

    # mask creation
    num_periods = ((2 * frame_size) / periods).long()
    mask = torch.ones(frames.shape, device=frames.device)
    index = torch.arange(2 * frame_size, device=x.device, dtype=x.dtype).view(1, 1, 2 * frame_size)
    index = torch.repeat_interleave(index, batch_size, dim=0)
    index = torch.repeat_interleave(index, num_frames, dim=1)
    max_index = frame_size + torch.div(num_periods * periods + 1, 2, rounding_mode='floor') - 1
    min_index = frame_size - torch.div(num_periods * periods, 2, rounding_mode='floor')
    mask[index > max_index.unsqueeze(-1)] = 0
    mask[index < min_index.unsqueeze(-1)] = 0

    # kernel creation
    f = 2 * torch.pi / periods
    index = index - frame_size
    f0 = (- f.unsqueeze(-1) * index).unsqueeze(-1)
    f1 = (torch.arange(ncoeffs, device=x.device) + (0 if constant_term else 1)).view(1, 1, 1, ncoeffs)
    f = f0 * f1
    kernel = torch.exp(1j*f)

    # masking and dot product computation
    frames = frames * mask
    dot_prod = torch.sum(kernel * frames.unsqueeze(-1), dim=-2)/ (num_periods * periods).unsqueeze(-1)

    # features
    norm_dot_prod = dot_prod / (torch.abs(dot_prod) + 1e-6)
    re_phase = torch.real(norm_dot_prod)
    im_phase = torch.imag(norm_dot_prod)
    abs_coeff = torch.abs(dot_prod)

    phase_features = torch.cat((abs_coeff, re_phase, im_phase), dim=-1)

    return phase_features


def create_phase_signal(periods, frame_size, pulses=False, sign=1, phase0=None):
    batch_size = periods.size(0)
    sign = -1 if sign < 0 else 1
    progression = torch.arange(1, frame_size + 1, dtype=periods.dtype, device=periods.device).view((1, -1))
    if sign < 0: progression = progression - 1
    progression = torch.repeat_interleave(progression, batch_size, 0)

    if phase0 is None:
        phase0 = torch.zeros(batch_size, dtype=periods.dtype, device=periods.device).unsqueeze(-1)

    chunks = []
    for sframe in range(periods.size(1)):
        f = (2.0 * torch.pi / periods[:, sframe]).unsqueeze(-1)

        if pulses:
            chunk_sin = torch.sin(sign * f  * progression + phase0).view(batch_size, 1, frame_size)
            chunk_sin = chunk_sin * (chunk_sin.abs()**10)
            pulse_a = torch.relu(chunk_sin)
            pulse_b = torch.relu(-chunk_sin)

            chunk = torch.cat((pulse_a, pulse_b), dim = 1)
        else:
            chunk_sin = torch.sin(sign * f  * progression + phase0).view(batch_size, 1, frame_size)
            chunk_cos = torch.cos(sign * f  * progression + phase0).view(batch_size, 1, frame_size)

            chunk = torch.cat((chunk_sin, chunk_cos), dim = 1)

        phase0 = phase0 + sign * frame_size * f

        chunks.append(chunk)

    phase_signal = torch.cat(chunks, dim=-1)

    return phase_signal