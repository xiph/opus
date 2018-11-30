#!/bin/sh

gcc -DTRAINING=1 -Wall -W -O3 -g -I../include denoise.c kiss_fft.c pitch.c celt_lpc.c -o dump_data -lm
gcc -o test_lpcnet -mavx2 -mfma -g -O3 -Wall -W -Wextra lpcnet.c nnet.c nnet_data.c -lm
