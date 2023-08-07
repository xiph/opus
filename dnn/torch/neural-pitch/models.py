"""
Pitch Estimation Models and dataloaders
    - Classification Based (Input features, output logits)
"""

import torch
import numpy as np

class large_if_ccode(torch.nn.Module):

    def __init__(self,input_dim = 90,gru_dim = 64,output_dim = 192):
        super(large_if_ccode,self).__init__()

        self.activation = torch.nn.Tanh()
        self.initial = torch.nn.Linear(input_dim,gru_dim)
        self.gru = torch.nn.GRU(input_size = gru_dim,hidden_size = gru_dim,batch_first = True)
        self.upsample = torch.nn.Linear(gru_dim,output_dim)

    def forward(self, x):

        x = self.initial(x)
        x = self.activation(x)
        x,_ = self.gru(x)
        x = self.upsample(x)
        x = self.activation(x)
        x = x.permute(0,2,1)

        return x


# Dataloaders
class loader(torch.utils.data.Dataset):
      def __init__(self, features_if, file_pitch,confidence_threshold = 0.4,dimension_if = 30,context = 100):
            self.if_feat = np.memmap(features_if, dtype=np.float32).reshape(-1,3*dimension_if)
            
            # Resolution of 20 cents
            self.cents = np.rint(np.load(file_pitch)[0,:]/20)
            self.cents = np.clip(self.cents,0,179)
            self.confidence = np.load(file_pitch)[1,:]

            # Filter confidence for CREPE
            self.confidence[self.confidence < confidence_threshold] = 0
            self.context = context
            # Clip both to same size
            size_common = min(self.if_feat.shape[0],self.cents.shape[0])
            self.if_feat = self.if_feat[:size_common,:]
            self.cents = self.cents[:size_common]
            self.confidence = self.confidence[:size_common]

            frame_max = self.if_feat.shape[0]//context
            self.if_feat = np.reshape(self.if_feat[:frame_max*context,:],(frame_max,context,3*dimension_if))
            self.cents = np.reshape(self.cents[:frame_max*context],(frame_max,context))
            self.confidence = np.reshape(self.confidence[:frame_max*context],(frame_max,context))

      def __len__(self):
            return self.if_feat.shape[0]

      def __getitem__(self, index):
            return torch.from_numpy(self.if_feat[index,:,:]),torch.from_numpy(self.cents[index]),torch.from_numpy(self.confidence[index])