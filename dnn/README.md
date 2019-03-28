# LPCNet

Low complexity implementation of the WaveRNN-based LPCNet algorithm, as described in:

- J.-M. Valin, J. Skoglund, [A Real-Time Wideband Neural Vocoder at 1.6 kb/s Using LPCNet](https://jmvalin.ca/papers/lpcnet_codec.pdf), *Submitted for INTERSPEECH 2019*.
- J.-M. Valin, J. Skoglund, [LPCNet: Improving Neural Speech Synthesis Through Linear Prediction](https://jmvalin.ca/papers/lpcnet_icassp2019.pdf), *Proc. International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, arXiv:1810.11846, 2019.

# Introduction

Work in progress software for researching low CPU complexity algorithms for speech synthesis and compression by applying Linear Prediction techniques to WaveRNN. High quality speech can be synthesised on regular CPUs (around 3 GFLOP) with SIMD support (AVX, AVX2/FMA, NEON currently supported). The code also supports very low bitrate compression at 1.6 kb/s.

The BSD licensed software is written in C and Python/Keras. For training, a GTX 1080 Ti or better is recommended.

This software is an open source starting point for LPCNet/WaveRNN-based speech synthesis and coding.

# Using the existing software

You can build the code using:

```
./autogen.sh
./configure
make
```
Note that the autogen.sh script is used when building from Git and will automatically download the latest model
(models are too large to put in Git).

It is highly recommended to set the CFLAGS environment variable to enable AVX or NEON *prior* to running configure, otherwise
no vectorization will take place and the code will be very slow. On a recent x86 CPU, something like
```
export CFLAGS='-O3 -g -mavx2 -mfma'
```
should work. On ARM, you can enable Neon with:
```
export CFLAGS='-O3 -g -mfpu=neon'
```

You can test the capabilities of LPCNet using the lpcnet_demo application. To encode a file:
```
./lpcnet_demo -encode input.pcm compressed.bin
```
where input.pcm is a 16-bit (machine endian) PCM file sampled at 16 kHz. The raw compressed data (no header)
is written to compressed.bin and consists of 8 bytes per 40-ms packet.

To decode:
```
./lpcnet_demo -decode compressed.bin output.pcm
```
where output.pcm is also 16-bit, 16 kHz PCM.

The same functionality is available in the form of a library. See include/lpcnet.h for the API.

# Training a new model

This codebase is also meant for research and it is possible to train new models. These are the steps to do that:

1. Set up a Keras system with GPU.

1. Generate training data:
   ```
   ./dump_data -train input.s16 features.f32 data.u8
   ```
   where the first file contains 16 kHz 16-bit raw PCM audio (no header) and the other files are output files. This program makes several passes over the data with different filters to generate a large amount of training data.

1. Now that you have your files, train with:
   ```
   ./src/train_lpcnet.py features.f32 data.u8
   ```
   and it will generate an lpcnet*.h5 file for each iteration. If it stops with a
   "Failed to allocate RNN reserve space" message try reducing the *batch\_size* variable in train_lpcnet.py.

1. You can synthesise speech with Python and your GPU card:
   ```
   ./dump_data -test test_input.s16 test_features.f32
   ./src/test_lpcnet.py test_features.f32 test.s16
   ```
   Note the .h5 is hard coded in test_lpcnet.py, modify for your .h5 file.

1. Or with C on a CPU:
   First extract the model files nnet_data.h and nnet_data.c
   ```
   ./dump_lpcnet.py lpcnet15_384_10_G16_64.h5
   ```
   and move the generated nnet_data.* files to the src/ directory.
   Then you just need to rebuild the software and use lpcnet_demo as explained above.

# Speech Material for Training 

Suitable training material can be obtained from the [McGill University Telecommunications & Signal Processing Laboratory](http://www-mmsp.ece.mcgill.ca/Documents/Data/).  Download the ISO and extract the 16k-LP7 directory, the src/concat.sh script can be used to generate a headerless file of training samples.
```
cd 16k-LP7
sh /path/to/concat.sh
```

# Reading Further

1. [LPCNet: DSP-Boosted Neural Speech Synthesis](https://people.xiph.org/~jm/demo/lpcnet/)
1. Sample model files:
https://jmvalin.ca/misc_stuff/lpcnet_models/

