/* Copyright (c) 2026 Xiph.Org Foundation
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

/* osce_compare: reference C implementation of the OSCE qualification metric.
 *
 * This is the self-contained C counterpart of
 *   dnn/torch/osce/stndrd/qualification/moc.py
 * and serves as the normative metric definition for the Opus speech coding
 * enhancement qualification tests. It provides two metrics:
 *
 *   - lowband  (default): a perceptually weighted degradation score on the
 *     0-8 kHz band, comparing a reference and a test signal (which may be
 *     sampled at different integer multiples of 16 kHz; the higher-rate signal
 *     is reduced to the 16 kHz lowband in the STFT domain). Prints the score.
 *
 *   - highband (-highband): a bandwidth-extension pass/fail on the 8-20 kHz
 *     band. A clip passes if, in every one of the four highbands, the decoded
 *     test signal beats a lowpass "no extension" anchor by at least tau. Both
 *     signals must be at 48 kHz. Prints the four per-band margins and PASS/FAIL.
 *
 * Input files are headerless 16-bit little-endian mono PCM (.s16).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.141592653589793
#endif

/* ------------------------------------------------------------------ */
/* lowband analysis grid (16 kHz equivalent): 10 ms window, 2.5 ms hop */
#define BASE_RATE   16000
#define BASE_WINDOW 160
#define BASE_HOP    40
#define LOWBAND_BINS 81          /* 0..8 kHz on a 100 Hz grid */
#define LOWBAND_NUM_BANDS 17
/* bark-like band limits (in bins), cutoff at 7.5 kHz */
static const int lowband_limits[LOWBAND_NUM_BANDS + 1] =
    {0, 2, 4, 6, 7, 9, 11, 13, 15, 18, 22, 26, 31, 36, 43, 51, 60, 75};

/* highband analysis: 48 kHz, 480-point STFT (100 Hz/bin) */
#define HB_NPERSEG   480
#define HB_HOP       240
#define HB_BINS      241
#define HB_LP_BIN    80          /* bins >= 80 (>= 8 kHz) form the highband */
#define HB_NUM_EBANDS 21
#define HB_EVAL_BANDS 4          /* last 4 ebands: 8.0-9.6, 9.6-12, 12-15.6, 15.6-20 kHz */
/* opus eband limits on the 10 ms (100 Hz) grid = 2 * eband5ms */
static const int hb_ebands[HB_NUM_EBANDS + 1] =
    {0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 68, 80, 96, 120, 156, 200};

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "out of memory\n"); exit(EXIT_FAILURE); }
    return p;
}

/* read a headerless 16-bit little-endian mono PCM file, scaled to [-1, 1] */
static double *read_s16(const char *path, long *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "could not open %s\n", path); exit(EXIT_FAILURE); }
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    long n = bytes / 2;
    unsigned char *raw = (unsigned char *)xmalloc((size_t)bytes);
    if (fread(raw, 1, (size_t)bytes, f) != (size_t)bytes) { fprintf(stderr, "read error %s\n", path); exit(EXIT_FAILURE); }
    fclose(f);
    double *x = (double *)xmalloc((size_t)n * sizeof(double));
    for (long i = 0; i < n; i++) {
        short s = (short)(raw[2*i] | (raw[2*i+1] << 8));
        x[i] = s / 32768.0;
    }
    free(raw);
    *out_len = n;
    return x;
}

/* -------------------- self-contained FFT (radix-2 + Bluestein) ----------- */
/* Window sizes here (160, 480, ...) are not powers of two, so we use
 * Bluestein's algorithm to turn an arbitrary-length DFT into power-of-two FFTs.
 * A plan (chirp + b-sequence FFT) depends only on the length and is reused
 * across all frames, which makes this O(N log N) per frame. */

static void fft_pow2(double *re, double *im, int n, int inv) {
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { double t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2.0 * M_PI / len * (inv ? 1.0 : -1.0);
        double wr = cos(ang), wi = sin(ang);
        for (int i = 0; i < n; i += len) {
            double cwr = 1.0, cwi = 0.0;
            for (int k = 0; k < len / 2; k++) {
                double xr = re[i + k + len / 2], xi = im[i + k + len / 2];
                double vr = xr * cwr - xi * cwi;
                double vi = xr * cwi + xi * cwr;
                double ur = re[i + k], ui = im[i + k];
                re[i + k] = ur + vr; im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
                double ncwr = cwr * wr - cwi * wi; cwi = cwr * wi + cwi * wr; cwr = ncwr;
            }
        }
    }
    if (inv) for (int i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

typedef struct {
    int N, m;
    double *wr, *wi;   /* chirp exp(-i*pi*k^2/N), length N */
    double *Br, *Bi;   /* FFT of the b-sequence, length m */
    double *ar, *ai;   /* scratch, length m */
} fft_plan;

static fft_plan *plan_create(int N) {
    fft_plan *p = (fft_plan *)xmalloc(sizeof(fft_plan));
    p->N = N;
    int m = 1; while (m < 2 * N - 1) m <<= 1;
    p->m = m;
    p->wr = (double *)xmalloc(N * sizeof(double));
    p->wi = (double *)xmalloc(N * sizeof(double));
    p->Br = (double *)xmalloc(m * sizeof(double));
    p->Bi = (double *)xmalloc(m * sizeof(double));
    p->ar = (double *)xmalloc(m * sizeof(double));
    p->ai = (double *)xmalloc(m * sizeof(double));
    for (int k = 0; k < N; k++) {
        double ang = M_PI * (double)(((long)k * k) % (2 * N)) / N;
        p->wr[k] = cos(ang); p->wi[k] = -sin(ang);
    }
    for (int i = 0; i < m; i++) { p->Br[i] = 0.0; p->Bi[i] = 0.0; }
    p->Br[0] = p->wr[0]; p->Bi[0] = -p->wi[0];        /* b[0] = conj(w[0]) */
    for (int k = 1; k < N; k++) {
        p->Br[k] = p->wr[k]; p->Bi[k] = -p->wi[k];    /* b[k] = conj(w[k]) */
        p->Br[m - k] = p->wr[k]; p->Bi[m - k] = -p->wi[k];
    }
    fft_pow2(p->Br, p->Bi, m, 0);
    return p;
}

static void plan_destroy(fft_plan *p) {
    free(p->wr); free(p->wi); free(p->Br); free(p->Bi); free(p->ar); free(p->ai); free(p);
}

/* |DFT(frame)|^2 for bins 0..kmax, where frame is a real windowed signal. */
static void plan_power(fft_plan *p, const double *frame, int kmax, double *power) {
    int N = p->N, m = p->m;
    for (int n = 0; n < N; n++) {
        p->ar[n] = frame[n] * p->wr[n];
        p->ai[n] = frame[n] * p->wi[n];
    }
    for (int i = N; i < m; i++) { p->ar[i] = 0.0; p->ai[i] = 0.0; }
    fft_pow2(p->ar, p->ai, m, 0);
    for (int i = 0; i < m; i++) {
        double xr = p->ar[i] * p->Br[i] - p->ai[i] * p->Bi[i];
        double xi = p->ar[i] * p->Bi[i] + p->ai[i] * p->Br[i];
        p->ar[i] = xr; p->ai[i] = xi;
    }
    fft_pow2(p->ar, p->ai, m, 1);
    for (int k = 0; k <= kmax; k++) {
        double xr = p->wr[k] * p->ar[k] - p->wi[k] * p->ai[k];
        double xi = p->wr[k] * p->ai[k] + p->wi[k] * p->ar[k];
        power[k] = xr * xr + xi * xi;
    }
}

/* ---------------------- lowband metric (moc.compare) ---------------------- */

/* Short-time lowband power spectrum: keep the first LOWBAND_BINS bins of a
 * fixed 10 ms window / 2.5 ms hop STFT (scaled by the rate factor), normalised
 * by factor^2 and offset by 100000, matching moc.py lowband_psd + _compare. */
static double *lowband_psd(const double *x, long n, int fs, int *num_frames_out) {
    int factor = fs / BASE_RATE;
    int ws = BASE_WINDOW * factor;
    int hop = BASE_HOP * factor;
    long nf_l = ((long)n - ws - hop) / hop;
    int num_frames = nf_l > 0 ? (int)nf_l : 0;
    *num_frames_out = num_frames;
    if (num_frames <= 0) return NULL;

    double *win = (double *)xmalloc(ws * sizeof(double));
    for (int i = 0; i < ws; i++) win[i] = 0.54 - 0.46 * cos(2.0 * M_PI * i / ws); /* periodic hamming */
    fft_plan *plan = plan_create(ws);

    double *frame = (double *)xmalloc(ws * sizeof(double));
    double *power = (double *)xmalloc((LOWBAND_BINS) * sizeof(double));
    double *psd = (double *)xmalloc((size_t)num_frames * LOWBAND_BINS * sizeof(double));
    double invf2 = 1.0 / ((double)factor * factor);

    for (int fr = 0; fr < num_frames; fr++) {
        const double *seg = x + (long)fr * hop;
        /* moc.py scales the [-1,1] signal back to int16 range (x * 2**15) before
           the power spectrum; the +100000 floor and 0.1 mask gain are calibrated
           for that scale. */
        for (int i = 0; i < ws; i++) frame[i] = seg[i] * 32768.0 * win[i];
        plan_power(plan, frame, LOWBAND_BINS - 1, power);
        for (int b = 0; b < LOWBAND_BINS; b++)
            psd[(size_t)fr * LOWBAND_BINS + b] = power[b] * invf2 + 100000.0;
    }
    free(win); plan_destroy(plan); free(frame); free(power);
    return psd;
}

static double lowband_score(const double *x, long nx, int fs_x,
                            const double *y, long ny, int fs_y) {
    /* trim to common duration */
    double duration = fmin((double)nx / fs_x, (double)ny / fs_y);
    long nxt = (long)(duration * fs_x);
    long nyt = (long)(duration * fs_y);

    int nfx, nfy;
    double *psd_x = lowband_psd(x, nxt, fs_x, &nfx);
    double *psd_y = lowband_psd(y, nyt, fs_y, &nfy);
    int NF = nfx < nfy ? nfx : nfy;
    if (NF <= 1) { free(psd_x); free(psd_y); return 0.0; }

    /* bin -> band and band widths */
    int band_of_bin[LOWBAND_BINS];
    for (int b = 0; b < LOWBAND_BINS; b++) band_of_bin[b] = -1;
    double bandwidth[LOWBAND_NUM_BANDS];
    for (int bnd = 0; bnd < LOWBAND_NUM_BANDS; bnd++) {
        bandwidth[bnd] = lowband_limits[bnd + 1] - lowband_limits[bnd];
        for (int bin = lowband_limits[bnd]; bin < lowband_limits[bnd + 1]; bin++)
            band_of_bin[bin] = bnd;
    }

    /* frequency masking matrix fmask = down_mask @ up_mask (up=0.1, down=0.03) */
    const double up_factor = 0.1, down_factor = 0.03;
    double up[LOWBAND_NUM_BANDS][LOWBAND_NUM_BANDS];
    double dn[LOWBAND_NUM_BANDS][LOWBAND_NUM_BANDS];
    double fmask[LOWBAND_NUM_BANDS][LOWBAND_NUM_BANDS];
    for (int i = 0; i < LOWBAND_NUM_BANDS; i++)
        for (int j = 0; j < LOWBAND_NUM_BANDS; j++) {
            up[i][j] = (j <= i) ? pow(up_factor, i - j) : 0.0;
            dn[i][j] = (j >= i) ? pow(down_factor, j - i) : 0.0;
        }
    for (int i = 0; i < LOWBAND_NUM_BANDS; i++)
        for (int j = 0; j < LOWBAND_NUM_BANDS; j++) {
            double s = 0.0;
            for (int k = 0; k < LOWBAND_NUM_BANDS; k++) s += dn[i][k] * up[k][j];
            fmask[i][j] = s;
        }

    /* band energies of the reference and frequency+temporal masking pattern */
    double *maskx = (double *)xmalloc((size_t)NF * LOWBAND_NUM_BANDS * sizeof(double));
    double be[LOWBAND_NUM_BANDS];
    for (int fr = 0; fr < NF; fr++) {
        for (int bnd = 0; bnd < LOWBAND_NUM_BANDS; bnd++) be[bnd] = 0.0;
        for (int bin = 0; bin < LOWBAND_BINS; bin++) {
            int bnd = band_of_bin[bin];
            if (bnd >= 0) be[bnd] += psd_x[(size_t)fr * LOWBAND_BINS + bin];
        }
        for (int bnd = 0; bnd < LOWBAND_NUM_BANDS; bnd++) be[bnd] /= bandwidth[bnd];
        for (int bnd = 0; bnd < LOWBAND_NUM_BANDS; bnd++) {
            double s = 0.0;
            for (int j = 0; j < LOWBAND_NUM_BANDS; j++) s += be[j] * fmask[bnd][j];
            maskx[(size_t)fr * LOWBAND_NUM_BANDS + bnd] = s;
        }
    }
    /* temporal masking (0.5 decay per 2.5 ms frame) */
    for (int fr = 1; fr < NF; fr++)
        for (int bnd = 0; bnd < LOWBAND_NUM_BANDS; bnd++)
            maskx[(size_t)fr * LOWBAND_NUM_BANDS + bnd] += 0.5 * maskx[(size_t)(fr - 1) * LOWBAND_NUM_BANDS + bnd];

    /* masked spectra, 2-frame average, per-band log-ratio distortion, per-frame mean */
    int M = NF - 1;
    double *Ef = (double *)xmalloc((size_t)M * sizeof(double));
    double mpx[LOWBAND_BINS], mpy[LOWBAND_BINS], eb[LOWBAND_NUM_BANDS];
    for (int fr = 0; fr < M; fr++) {
        for (int bin = 0; bin < LOWBAND_BINS; bin++) {
            int bnd = band_of_bin[bin];
            double addx0 = (bnd >= 0) ? 0.1 * maskx[(size_t)fr * LOWBAND_NUM_BANDS + bnd] : 0.0;
            double addx1 = (bnd >= 0) ? 0.1 * maskx[(size_t)(fr + 1) * LOWBAND_NUM_BANDS + bnd] : 0.0;
            /* 2-frame average of masked psd (frames fr and fr+1) */
            mpx[bin] = (psd_x[(size_t)(fr + 1) * LOWBAND_BINS + bin] + addx1)
                     + (psd_x[(size_t)fr * LOWBAND_BINS + bin] + addx0);
            mpy[bin] = (psd_y[(size_t)(fr + 1) * LOWBAND_BINS + bin] + addx1)
                     + (psd_y[(size_t)fr * LOWBAND_BINS + bin] + addx0);
        }
        for (int bnd = 0; bnd < LOWBAND_NUM_BANDS; bnd++) eb[bnd] = 0.0;
        for (int bin = 0; bin < LOWBAND_BINS; bin++) {
            int bnd = band_of_bin[bin];
            if (bnd >= 0) {
                double lr = log(mpy[bin] / mpx[bin]);
                eb[bnd] += lr * lr;
            }
        }
        double ef = 0.0;
        for (int bnd = 0; bnd < LOWBAND_NUM_BANDS; bnd++) ef += eb[bnd] / bandwidth[bnd];
        Ef[fr] = ef / LOWBAND_NUM_BANDS;
    }

    /* tmp = |Ef|^3, centred 400-tap moving average (numpy 'same'), max, ^(1/6) */
    double *tmp = (double *)xmalloc((size_t)M * sizeof(double));
    for (int i = 0; i < M; i++) tmp[i] = fabs(Ef[i]) * fabs(Ef[i]) * fabs(Ef[i]);
    double best = 0.0;
    for (int i = 0; i < M; i++) {
        double acc = 0.0;
        for (int j = i - 200; j <= i + 199; j++)
            if (j >= 0 && j < M) acc += tmp[j];
        acc /= 400.0;
        if (acc > best) best = acc;
    }
    double err = pow(best, 1.0 / 6.0);

    free(psd_x); free(psd_y); free(maskx); free(Ef); free(tmp);
    return err;
}

/* ---------------------- highband metric (moc.highband_compare) ------------ */

/* full 48 kHz STFT power (num_frames x HB_BINS): periodic hann, boundary zeros
 * (HB_HOP each side), hop HB_HOP. Consistent framing for ref/test/anchor makes
 * the pass/fail decision invariant to the constant STFT scaling. */
static double *highband_psd(const double *x, long n, int *num_frames_out) {
    long ext = n + 2 * HB_HOP;                 /* boundary='zeros' padding */
    int num_frames = (int)((ext - HB_NPERSEG) / HB_HOP) + 1;
    if (num_frames < 1) { *num_frames_out = 0; return NULL; }
    *num_frames_out = num_frames;

    double *win = (double *)xmalloc(HB_NPERSEG * sizeof(double));
    double winsum = 0.0;
    for (int i = 0; i < HB_NPERSEG; i++) { win[i] = 0.5 - 0.5 * cos(2.0 * M_PI * i / HB_NPERSEG); winsum += win[i]; }
    fft_plan *plan = plan_create(HB_NPERSEG);

    double *frame = (double *)xmalloc(HB_NPERSEG * sizeof(double));
    double *power = (double *)xmalloc(HB_BINS * sizeof(double));
    double *psd = (double *)xmalloc((size_t)num_frames * HB_BINS * sizeof(double));
    double inv = 1.0 / winsum;

    for (int fr = 0; fr < num_frames; fr++) {
        long start = (long)fr * HB_HOP - HB_HOP;   /* index into original x (boundary offset) */
        for (int i = 0; i < HB_NPERSEG; i++) {
            long idx = start + i;
            double v = (idx >= 0 && idx < n) ? x[idx] : 0.0;
            frame[i] = v * win[i] * inv;
        }
        plan_power(plan, frame, HB_BINS - 1, power);
        memcpy(psd + (size_t)fr * HB_BINS, power, HB_BINS * sizeof(double));
    }
    free(win); plan_destroy(plan); free(frame); free(power);
    return psd;
}

/* per-eband distortion (last HB_EVAL_BANDS bands) between two power spectra,
 * using a -30 dB floor derived from the reference. */
static void highband_band_distortion(const double *psd_ref, const double *psd_cmp,
                                     int num_frames, double *dist_out /* [HB_EVAL_BANDS] */) {
    int first = HB_NUM_EBANDS - HB_EVAL_BANDS;
    for (int e = first; e < HB_NUM_EBANDS; e++) {
        int lo = hb_ebands[e], hi = hb_ebands[e + 1];
        /* reference band energy per frame + peak for noise floor */
        double peak = 0.0;
        double *yref = (double *)xmalloc((size_t)num_frames * sizeof(double));
        double *ycmp = (double *)xmalloc((size_t)num_frames * sizeof(double));
        for (int fr = 0; fr < num_frames; fr++) {
            double sr = 0.0, sc = 0.0;
            for (int bin = lo; bin < hi; bin++) {
                sr += psd_ref[(size_t)fr * HB_BINS + bin];
                sc += psd_cmp[(size_t)fr * HB_BINS + bin];
            }
            yref[fr] = sr; ycmp[fr] = sc;
            if (sr > peak) peak = sr;
        }
        double nf = peak * pow(10.0, -30.0 / 10.0);
        double acc = 0.0;
        for (int fr = 0; fr < num_frames; fr++) {
            double a = pow(yref[fr] > nf ? yref[fr] : nf, 0.25);
            double b = pow(ycmp[fr] > nf ? ycmp[fr] : nf, 0.25);
            double d = fabs(a - b);
            acc += d * d;
        }
        /* norm(delta / num_frames, ord=2) = sqrt(sum (delta/num_frames)^2) */
        double dist = sqrt(acc) / num_frames;
        dist_out[e - first] = 1000.0 * dist;
        free(yref); free(ycmp);
    }
}

static int highband_compare(const double *x, long nx, const double *y, long ny,
                            double tau, double *margins /* [HB_EVAL_BANDS] */) {
    long n = nx < ny ? nx : ny;
    /* normalise both by the reference peak */
    double m = 0.0;
    for (long i = 0; i < n; i++) { double a = fabs(x[i]); if (a > m) m = a; }
    if (m == 0.0) { for (int e = 0; e < HB_EVAL_BANDS; e++) margins[e] = 0.0; return 0; }
    double *xn = (double *)xmalloc((size_t)n * sizeof(double));
    double *yn = (double *)xmalloc((size_t)n * sizeof(double));
    for (long i = 0; i < n; i++) { xn[i] = x[i] / m; yn[i] = y[i] / m; }

    int nfr, nfc;
    double *psd_ref = highband_psd(xn, n, &nfr);
    double *psd_test = highband_psd(yn, n, &nfc);
    int NF = nfr < nfc ? nfr : nfc;

    /* lowpass anchor: reference with bins >= HB_LP_BIN zeroed */
    double *psd_lp = (double *)xmalloc((size_t)NF * HB_BINS * sizeof(double));
    for (int fr = 0; fr < NF; fr++)
        for (int bin = 0; bin < HB_BINS; bin++)
            psd_lp[(size_t)fr * HB_BINS + bin] = (bin < HB_LP_BIN) ? psd_ref[(size_t)fr * HB_BINS + bin] : 0.0;

    double dist_test[HB_EVAL_BANDS], dist_lp[HB_EVAL_BANDS];
    highband_band_distortion(psd_ref, psd_test, NF, dist_test);
    highband_band_distortion(psd_ref, psd_lp, NF, dist_lp);

    int passed = 1;
    for (int e = 0; e < HB_EVAL_BANDS; e++) {
        margins[e] = dist_lp[e] - dist_test[e];
        if (!(margins[e] >= tau)) passed = 0;
    }
    free(xn); free(yn); free(psd_ref); free(psd_test); free(psd_lp);
    return passed;
}

/* ------------------------------------------------------------------------- */

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s [-highband] [-fs_ref RATE] [-fs_test RATE] [-tau X] [-delay D] ref.s16 test.s16\n"
        "  default: lowband degradation score (prints the score)\n"
        "  -highband : highband bandwidth-extension margins + PASS/FAIL (both signals 48 kHz)\n"
        "  -fs_ref/-fs_test : sampling rates in Hz (multiples of 16000; default 16000)\n"
        "  -tau      : highband pass margin (default 0)\n"
        "  -delay    : samples trimmed from the start of the test signal (test rate; default 0)\n",
        prog);
}

int main(int argc, char **argv) {
    int highband = 0, fs_ref = BASE_RATE, fs_test = BASE_RATE, delay = 0;
    double tau = 0.0;
    const char *ref_path = NULL, *test_path = NULL;

    int a = 1;
    for (; a < argc; a++) {
        if (!strcmp(argv[a], "-highband")) highband = 1;
        else if (!strcmp(argv[a], "-fs_ref") && a + 1 < argc) fs_ref = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-fs_test") && a + 1 < argc) fs_test = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-tau") && a + 1 < argc) tau = atof(argv[++a]);
        else if (!strcmp(argv[a], "-delay") && a + 1 < argc) delay = atoi(argv[++a]);
        else if (argv[a][0] == '-') { usage(argv[0]); return 2; }
        else if (!ref_path) ref_path = argv[a];
        else if (!test_path) test_path = argv[a];
        else { usage(argv[0]); return 2; }
    }
    if (!ref_path || !test_path) { usage(argv[0]); return 2; }

    if (highband) { fs_ref = 48000; fs_test = 48000; }
    if (fs_ref % BASE_RATE || fs_test % BASE_RATE) {
        fprintf(stderr, "error: sampling rates must be multiples of %d\n", BASE_RATE);
        return 2;
    }

    long nref, ntest;
    double *xref = read_s16(ref_path, &nref);
    double *xtest = read_s16(test_path, &ntest);

    const double *tsig = xtest;
    long tn = ntest;
    if (delay > 0 && delay < ntest) { tsig = xtest + delay; tn = ntest - delay; }

    if (highband) {
        double margins[HB_EVAL_BANDS];
        int passed = highband_compare(xref, nref, tsig, tn, tau, margins);
        printf("%.8f %.8f %.8f %.8f %s\n",
               margins[0], margins[1], margins[2], margins[3], passed ? "PASS" : "FAIL");
    } else {
        double score = lowband_score(xref, nref, fs_ref, tsig, tn, fs_test);
        printf("%.8f\n", score);
    }
    free(xref); free(xtest);
    return 0;
}
