import numpy as np
from tensorflow.keras.utils import Sequence

class LPCNetLoader(Sequence):
    def __init__(self, data, features, periods, batch_size):
        self.batch_size = batch_size
        self.nb_batches = np.minimum(np.minimum(data.shape[0], features.shape[0]), periods.shape[0])//self.batch_size
        self.data = data[:self.nb_batches*self.batch_size, :]
        self.features = features[:self.nb_batches*self.batch_size, :]
        self.periods = periods[:self.nb_batches*self.batch_size, :]
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indices = np.arange(self.nb_batches*self.batch_size)
        np.random.shuffle(self.indices)

    def __getitem__(self, index):
        data = self.data[self.indices[index*self.batch_size:(index+1)*self.batch_size], :, :]
        in_data = data[: , :, :3]
        out_data = data[: , :, 3:4]
        features = self.features[self.indices[index*self.batch_size:(index+1)*self.batch_size], :, :]
        periods = self.periods[self.indices[index*self.batch_size:(index+1)*self.batch_size], :, :]
        return ([in_data, features, periods], out_data)

    def __len__(self):
        return self.nb_batches
