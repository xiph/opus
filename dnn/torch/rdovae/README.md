# Rate-Distortion-Optimized Variational Auto-Encoder

## Setup
The python code requires python >= 3.6 and has been tested with python 3.6 and python 3.10. To install requirements run
```
python -m pip install -r requirements.txt
```

## Training
To generate training data use dump date from the main LPCNet repo
```
./dump_data -train 16khz_speech_input.s16 features.f32 data.s16
```

To train the model, simply run
```
python train_rdovae.py features.f32 output_folder
```

To train on CUDA device add `--cuda-visible-devices idx`.


## ToDo
- Upload checkpoints and add URLs
