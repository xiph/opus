%% Tests bit-stream compliance for the Opus codec
%% x: Signal from the Opus reference implementation (float or fixed)
%% y: Signal from the decoder under test
%% stereo: 0 for mono, 1 for stereo
function [err, NMR] = opus_compare(x, y, stereo)

% Bands on which we compute the pseudo-NMR (Bark-derived CELT bands)
b = 2*[0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100];
d = diff(b);

% Per-band SNR threshold
T = 50-.7*[1:21];

% Noise floor
N = 10 .^ ((10-0.6*[1:21])/10);

% Error signal
e=x-y;

%Add a +/- 1 dead zone on the error
e = e - min(1, max(-1, e));

% Compute spectrum of original and error
if (stereo)
  X=(abs(specgram(x(1:2:end),480))+abs(specgram(x(2:2:end),480)))/2;
  E=(abs(specgram(e(1:2:end),480))+abs(specgram(e(2:2:end),480)))/2;
else
  X=abs(specgram(x,480));
  E=abs(specgram(e,480));
endif

% Group energy per band
for k=1:21
   Xb(k,:) = sum(X(b(k)+1:b(k+1),:).^2)/d(k)+1;
   Eb(k,:) = sum(E(b(k)+1:b(k+1),:).^2)/d(k)+1;
end

% Frequency masking (low to high) with 10 dB/Bark slope
Xb = filter(1, [1, -.1], Xb);
% Frequency masking (high to low) with 15 dB/Bark slope
Xb(end:-1:1,:) = filter(1, [1, -.03], Xb(end:-1:1,:));

% Temporal masking with 5 dB/5 ms slope
Xb = filter(1, [1, -.3], Xb')';

% NMR threshold
T0 = ones(length(Eb), 1)*(10.^((T)/10));

% Time-frequency SNR
NMR = (Xb./Eb)';

%Picking only errors in the 90th percentile
tmp = Eb(:);
thresh = sort(tmp)(round(.90*length(tmp)));
weight = Eb'>thresh;

printf("Average pseudo-NMR: %3.2f dB\n", mean(mean(10*log10(NMR))));

if (sum(sum(weight))<1)
   printf("Mismatch level: below noise floor\n");
   err = -100;
else
   M = (T0./NMR) .* weight;

   err = 10*log10(sum(sum(M)) / sum(sum(weight)));

   printf("Weighted mismatch: %3.2f dB\n", err);
endif

printf("\n");

if (err < 0)
   printf("**Decoder PASSES test (mismatch < 0 dB)\n");
else
   printf("**Decoder FAILS test (mismatch >= 0 dB)\n");
endif



