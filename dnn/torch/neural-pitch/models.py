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
        self.hidden = torch.nn.Linear(gru_dim,gru_dim)
        self.gru = torch.nn.GRU(input_size = gru_dim,hidden_size = gru_dim,batch_first = True)
        self.upsample = torch.nn.Linear(gru_dim,output_dim)

    def forward(self, x):

        x = self.initial(x)
        x = self.activation(x)
        x = self.hidden(x)
        x = self.activation(x)
        x,_ = self.gru(x)
        x = self.upsample(x)
        x = self.activation(x)
        x = x.permute(0,2,1)

        return x

class large_xcorr(torch.nn.Module):

    def __init__(self,input_dim = 90,gru_dim = 64,output_dim = 192):
        super(large_xcorr,self).__init__()

        self.activation = torch.nn.Tanh()

        self.conv = torch.nn.Sequential(
            torch.nn.ZeroPad2d((2,0,1,1)),
            torch.nn.Conv2d(1, 8, 3, bias = True),
            self.activation,
            torch.nn.ZeroPad2d((2,0,1,1)),
            torch.nn.Conv2d(8, 8, 3, bias = True),
            self.activation,
            torch.nn.ZeroPad2d((2,0,1,1)),
            torch.nn.Conv2d(8, 1, 3, bias = True),
            self.activation,
        )

        # self.conv = torch.nn.Sequential(
        #     torch.nn.ConstantPad1d((2,0),0),
        #     torch.nn.Conv1d(64,10,3),
        #     self.activation,
        #     torch.nn.ConstantPad1d((2,0),0),
        #     torch.nn.Conv1d(10,64,3),
        #     self.activation,
        # )

        self.downsample = torch.nn.Sequential(
            torch.nn.Linear(input_dim,gru_dim),
            self.activation
        )
        self.GRU = torch.nn.GRU(input_size = gru_dim,hidden_size = gru_dim,num_layers = 1,batch_first = True)
        self.upsample = torch.nn.Sequential(
            torch.nn.Linear(gru_dim,output_dim),
            self.activation
        )

    def forward(self, x):
        # x = x[:,:,:257].unsqueeze(-1)
        x = self.conv(x.unsqueeze(-1).permute(0,3,2,1)).squeeze(1)
        # print(x.shape)
        # x = self.conv(x.permute(0,3,2,1)).squeeze(1)
        x,_ = self.GRU(self.downsample(x.permute(0,2,1)))
        x = self.upsample(x).permute(0,2,1)

        # x = self.downsample(x)
        # x = self.activation(x)
        # x = self.conv(x.permute(0,2,1)).permute(0,2,1)
        # x,_ = self.GRU(x)
        # x = self.upsample(x).permute(0,2,1)
        return x

class large_joint(torch.nn.Module):
    """
    Joint IF-xcorr
    1D CNN on IF, merge with xcorr, 2D CNN on merged + GRU
    """

    def __init__(self,input_IF_dim = 90,input_xcorr_dim = 257,gru_dim = 64,output_dim = 192):
        super(large_joint,self).__init__()

        self.activation = torch.nn.Tanh()

        self.if_upsample = torch.nn.Sequential(
            torch.nn.Linear(input_IF_dim,64),
            self.activation,
            torch.nn.Linear(64,input_xcorr_dim),
            self.activation,
        )

        # self.if_upsample = torch.nn.Sequential(
        #     torch.nn.ConstantPad1d((2,0),0),
        #     torch.nn.Conv1d(90,10,3),
        #     self.activation,
        #     torch.nn.ConstantPad1d((2,0),0),
        #     torch.nn.Conv1d(10,257,3),
        #     self.activation,
        # )

        self.conv = torch.nn.Sequential(
            torch.nn.ZeroPad2d((2,0,1,1)),
            torch.nn.Conv2d(2, 8, 3, bias = True),
            self.activation,
            torch.nn.ZeroPad2d((2,0,1,1)),
            torch.nn.Conv2d(8, 8, 3, bias = True),
            self.activation,
            torch.nn.ZeroPad2d((2,0,1,1)),
            torch.nn.Conv2d(8, 1, 3, bias = True),
            self.activation,
        )

        # self.conv = torch.nn.Sequential(
        #     torch.nn.ConstantPad1d((2,0),0),
        #     torch.nn.Conv1d(257,10,3),
        #     self.activation,
        #     torch.nn.ConstantPad1d((2,0),0),
        #     torch.nn.Conv1d(10,64,3),
        #     self.activation,
        # )

        self.downsample = torch.nn.Sequential(
            torch.nn.Linear(input_xcorr_dim,gru_dim),
            self.activation
        )
        self.GRU = torch.nn.GRU(input_size = gru_dim,hidden_size = gru_dim,num_layers = 1,batch_first = True)
        self.upsample = torch.nn.Sequential(
            torch.nn.Linear(gru_dim,output_dim),
            self.activation
        )

    def forward(self, x):
        xcorr_feat = x[:,:,:257]
        if_feat = x[:,:,257:]
        x = torch.cat([xcorr_feat.unsqueeze(-1),self.if_upsample(if_feat).unsqueeze(-1)],axis = -1)
        x = self.conv(x.permute(0,3,2,1)).squeeze(1)
        x,_ = self.GRU(self.downsample(x.permute(0,2,1)))
        x = self.upsample(x).permute(0,2,1)

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
      
class loader_joint(torch.utils.data.Dataset):
      def __init__(self, features_if, file_pitch, features_xcorr,confidence_threshold = 0.4,context = 100, choice_data = 'both'):
            self.if_feat = np.memmap(features_if, dtype=np.float32).reshape(-1,90)
            self.xcorr = np.memmap(features_xcorr, dtype=np.float32).reshape(-1,257)
            self.cents = np.rint(np.load(file_pitch)[0,:]/20)
            self.cents = np.clip(self.cents,0,179)
            self.confidence = np.load(file_pitch)[1,:]
            # Filter confidence for CREPE
            self.confidence[self.confidence < confidence_threshold] = 0
            self.context = context
            
            self.choice_data = choice_data

            frame_max = self.if_feat.shape[0]//context
            self.if_feat = np.reshape(self.if_feat[:frame_max*context,:],(frame_max,context,90))
            self.cents = np.reshape(self.cents[:frame_max*context],(frame_max,context))
            self.xcorr = np.reshape(self.xcorr[:frame_max*context,:],(frame_max,context,257))
            # self.cents = np.rint(60*np.log2(256/(self.periods + 1.0e-8))).astype('int')
            # self.cents = np.clip(self.cents,0,239)
            self.confidence = np.reshape(self.confidence[:frame_max*context],(frame_max,context))
            # print(self.if_feat.shape)
      def __len__(self):
            return self.if_feat.shape[0]

      def __getitem__(self, index):
            if self.choice_data == 'both':
                return torch.cat([torch.from_numpy(self.xcorr[index,:,:]),torch.from_numpy(self.if_feat[index,:,:])],dim=-1),torch.from_numpy(self.cents[index]),torch.from_numpy(self.confidence[index])
            elif self.choice_data == 'if':
                return torch.from_numpy(self.if_feat[index,:,:]),torch.from_numpy(self.cents[index]),torch.from_numpy(self.confidence[index])
            else:
                return torch.from_numpy(self.xcorr[index,:,:]),torch.from_numpy(self.cents[index]),torch.from_numpy(self.confidence[index])