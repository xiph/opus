#!/usr/bin/python3

import lpcnet
import sys
import numpy as np
from keras.optimizers import Adam
from ulaw import ulaw2lin, lin2ulaw

nb_epochs = 10
batch_size = 32

model = lpcnet.new_wavernn_model()
model.compile(optimizer=Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.summary()

pcmfile = sys.argv[1]
chunk_size = int(sys.argv[2])

data = np.fromfile(pcmfile, dtype='int16')
#data = data[:100000000]
data = data/32768
nb_frames = (len(data)-1)//chunk_size

in_data = data[:nb_frames*chunk_size]
#out_data = data[1:1+nb_frames*chunk_size]//256 + 128
out_data = lin2ulaw(data[1:1+nb_frames*chunk_size]) + 128

in_data = np.reshape(in_data, (nb_frames, chunk_size, 1))
out_data = np.reshape(out_data, (nb_frames, chunk_size, 1))

model.fit(in_data, out_data, batch_size=batch_size, epochs=nb_epochs, validation_split=0.2)
