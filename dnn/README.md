# CELPNet

Low complexity WaveRNN-based speech coding by [Jean-Marc Valin](https://jmvalin.ca/)

# Introduction

Work in progress software for researching low CPU complexity algorithms for speech compression by applying Linear Prediction techniques to WaveRNN. The goal is to reduce the CPU complexity such that high quality speech can be synthesised on regular CPUs (around 1 GFLOP).

The BSD licensed software is written in C and Keras and currently requires a GPU (e.g. GT1060) to run.
For training models, a GTX 1080 Ti or better is recommended.

This software is also a useful resource as an open source starting point for WaveRNN-based speech coding.

# Quickstart

1. Set up a Keras system with GPU.

1. In the src/ directory, run ./compile.sh to compile the data processing program.

1. Then, run the resulting executable:
   ```
   ./dump_data input.s16 exc.s8 features.f32 pred.s16 pcm.s16
   ```

   where the first file contains 16 kHz 16-bit raw PCM audio (no header)
and the other files are output files. The input file currently used 
is 6 hours long, but you may be able to get away with less (and you can
always use ±5% or 10% resampling to augment your data).

1. Now that you have your files, you can do the training with:
   ```
   ./train_lpcnet.py exc.s8 features.f32 pred.s16 pcm.s16
   ```
   and it will generate a wavenet*.h5 file for each iteration. If it stops with a 
   "Failed to allocate RNN reserve space" message try reducing the *batch\_size* variable in train_wavenet_audio.py.

1. You can synthesise speech with:
  ```
   ./test_lpcnet.py features.f32 > pcm.txt
  ```
  The output file pcm.txt contains ASCII PCM samples that need to be converted to WAV for playback
  
# Speech Material for Training 

Suitable training material can be obtained from the [McGill University Telecommunications & Signal Processing Laboratory](http://www-mmsp.ece.mcgill.ca/Documents/Data/).  Download the ISO and extract the 16k-LP7 directory, the src/concat.sh script can be used to generate a headerless file of training samples.
```
cd 16k-LP7
sh ~/CELP/src/concat.sh
```

# Reading Further

1. If you're lucky, you may be able to get the current model at:
https://jmvalin.ca/misc_stuff/lpcnet_models/

1. [WaveNet and Codec 2](https://www.rowetel.com/?p=5966)
