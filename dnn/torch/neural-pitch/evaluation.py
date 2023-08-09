"""
Evaluation script to compute the Raw Pitch Accuracy
Procedure:
    - Look at all voiced frames in file
    - Compute number of pitches in those frames that lie within a 50 cent threshold
    RPA = (Total number of pitches within threshold summed across all files)/(Total number of voiced frames summed accross all files)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from prettytable import PrettyTable
import numpy as np
import glob
import random
import tqdm
import torch
import librosa
import json
from utils import stft, random_filter, feature_xform
import subprocess
import crepe

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rca(reference,input,voicing,thresh = 25):
    idx_voiced = np.where(voicing != 0)[0]
    acc = np.where(np.abs(reference - input)[idx_voiced] < thresh)[0]
    return acc.shape[0]

def sweep_rca(reference,input,voicing,thresh = 25,ind_arr = np.arange(-10,10)):
    l = []
    for i in ind_arr:
        l.append(rca(reference,np.roll(input,i),voicing,thresh))
    l = np.array(l)

    return np.max(l)

def rpa(model,device = 'cpu',data_format = 'if'):
    list_files = glob.glob('/home/ubuntu/Code/Datasets/SPEECH DATA/combined_mic_16k_raw/*.raw')
    dir_f0 = '/home/ubuntu/Code/Datasets/SPEECH DATA/combine_f0_ptdb/'
    # random_shuffle = list(np.random.permutation(len(list_files)))
    random.shuffle(list_files)
    list_files = list_files[:1000]

    # C_lp = 0
    # C_lp_m = 0
    # C_lp_f = 0
    # list_rca_model_lp = []
    # list_rca_male_lp = []
    # list_rca_female_lp = []

    # C_hp = 0
    # C_hp_m = 0
    # C_hp_f = 0
    # list_rca_model_hp = []
    # list_rca_male_hp = []
    # list_rca_female_hp = []

    C_all = 0
    C_all_m = 0
    C_all_f = 0
    list_rca_model_all = []
    list_rca_male_all = []
    list_rca_female_all = []

    thresh = 50
    N = 320
    H = 160
    freq_keep = 30

    for idx in tqdm.trange(len(list_files)):
        audio_file = list_files[idx]
        file_name = os.path.basename(list_files[idx])[:-4]

        audio = np.memmap(list_files[idx], dtype=np.int16)/(2**15 - 1)
        offset = 432
        audio = audio[offset:]
        rmse = np.squeeze(librosa.feature.rms(y = audio,frame_length = 320,hop_length = 160))

        spec = stft(x = np.concatenate([np.zeros(160),audio]), w = 'boxcar', N = N, H = H).T
        phase_diff = spec*np.conj(np.roll(spec,1,axis = -1))
        phase_diff = phase_diff/(np.abs(phase_diff) + 1.0e-8)
        idx_save = np.concatenate([np.arange(freq_keep),(N//2 + 1) + np.arange(freq_keep),2*(N//2 + 1) + np.arange(freq_keep)])
        feature = np.concatenate([np.log(np.abs(spec) + 1.0e-8),np.real(phase_diff),np.imag(phase_diff)],axis = 0).T
        feature_if = feature[:,idx_save]

        data_temp = np.memmap('./temp.raw', dtype=np.int16, shape=(audio.shape[0]), mode='w+')
        data_temp[:audio.shape[0]] = (audio/(np.max(np.abs(audio)))*(2**15 - 1)).astype(np.int16)

        subprocess.run(["../../../lpcnet_xcorr_extractor", './temp.raw', './temp_xcorr.f32'])
        feature_xcorr = np.flip(np.fromfile('./temp_xcorr.f32', dtype='float32').reshape((-1,256),order = 'C'),axis = 1)
        ones_zero_lag = np.expand_dims(np.ones(feature_xcorr.shape[0]),-1)
        feature_xcorr = np.concatenate([ones_zero_lag,feature_xcorr],axis = -1)
        # feature_xcorr = feature_xform(feature_xcorr)

        os.remove('./temp.raw')
        os.remove('./temp_xcorr.f32')

        if data_format == 'if':
            feature = feature_if
        elif data_format == 'xcorr':
            feature = feature_xcorr
        else:
            indmin = min(feature_if.shape[0],feature_xcorr.shape[0])
            feature = np.concatenate([feature_xcorr[:indmin,:],feature_if[:indmin,:]],-1)


        pitch_file_name = dir_f0 + "ref" + os.path.basename(list_files[idx])[3:-4] + ".f0"
        pitch = np.loadtxt(pitch_file_name)[:,0]
        voicing = np.loadtxt(pitch_file_name)[:,1]
        indmin = min(voicing.shape[0],rmse.shape[0],pitch.shape[0])
        pitch = pitch[:indmin]
        voicing = voicing[:indmin]
        rmse = rmse[:indmin]
        voicing = voicing*(rmse > 0.05*np.max(rmse))
        if "mic_F" in audio_file:
            idx_correct = np.where(pitch < 125)
            voicing[idx_correct] = 0

        cent = np.rint(1200*np.log2(np.divide(pitch, (16000/256), out=np.zeros_like(pitch), where=pitch!=0) + 1.0e-8)).astype('int')

        # if (model == 'penn'):
        # model_frequency, _ = penn.from_audio(
        # torch.from_numpy(audio).unsqueeze(0).float(),
        # 16000,
        # hopsize=0.01,
        # fmin=(16000.0/256),
        # fmax=500,
        # checkpoint=penn.DEFAULT_CHECKPOINT,
        # batch_size=32,
        # pad=True,
        # interp_unvoiced_at=0.065,
        # gpu=0)
        # model_frequency = model_frequency.cpu().detach().squeeze().numpy()
        # model_cents = 1200*np.log2(model_frequency/(16000/256))

        # elif (model == 'crepe'):
        # _, model_frequency, _, _ = crepe.predict(audio, 16000, viterbi=vflag,center=True,verbose=0)
        # lpcnet_file_name = '/home/ubuntu/Code/Datasets/SPEECH_DATA/lpcnet_f0_16k_residual/' + file_name + '_f0.f32'
        # period_lpcnet = np.fromfile(lpcnet_file_name, dtype='float32')
        # model_frequency = 16000/(period_lpcnet + 1.0e-6)
        # model_cents = 1200*np.log2(model_frequency/(16000/256))
        # else:
        model_cents = model(torch.from_numpy(np.copy(np.expand_dims(feature,0))).float().to(device))
        model_cents = 20*model_cents.argmax(dim=1).cpu().detach().squeeze().numpy()
        # model_cents = np.roll(model_cents,-1*3)

        num_frames = min(cent.shape[0],model_cents.shape[0])
        pitch = pitch[:num_frames]
        cent = cent[:num_frames]
        voicing = voicing[:num_frames]
        model_cents = model_cents[:num_frames]

        voicing_all = np.copy(voicing)
        # Forcefully make regions where pitch is <65 or greater than 500 unvoiced for relevant accurate pitch comparisons for our model
        force_out_of_pitch = np.where(np.logical_or(pitch < 65,pitch > 500)==True)
        voicing_all[force_out_of_pitch] = 0
        C_all = C_all + np.where(voicing_all != 0)[0].shape[0]

        # list_rca_model_all.append(sweep_rca(cent,model_cents,voicing_all,thresh,[0]))
        list_rca_model_all.append(rca(cent,model_cents,voicing_all,thresh))
        # list_rca_model_all.append(np.count_nonzero(np.where(np.abs(cent - model_cents))))

        if "mic_M" in audio_file:
            # list_rca_male_all.append(sweep_rca(cent,model_cents,voicing_all,thresh,[0]))
            list_rca_male_all.append(rca(cent,model_cents,voicing_all,thresh))
            C_all_m = C_all_m + np.where(voicing_all != 0)[0].shape[0]
        else:
            # list_rca_female_all.append(sweep_rca(cent,model_cents,voicing_all,thresh,[0]))
            list_rca_female_all.append(rca(cent,model_cents,voicing_all,thresh))
            C_all_f = C_all_f + np.where(voicing_all != 0)[0].shape[0]

        """
        # Low pitch estimation
        voicing_lp = np.copy(voicing)
        force_out_of_pitch = np.where(np.logical_or(pitch < 65,pitch > 125)==True)
        voicing_lp[force_out_of_pitch] = 0
        C_lp = C_lp + np.where(voicing_lp != 0)[0].shape[0]

        # list_rca_model_lp.append(sweep_rca(cent,model_cents,voicing_lp,thresh,[0]))
        list_rca_model_lp.append(rca(cent,model_cents,voicing_lp,thresh))

        if "mic_M" in audio_file:
            # list_rca_male_lp.append(sweep_rca(cent,model_cents,voicing_lp,thresh,[0]))
            list_rca_male_lp.append(rca(cent,model_cents,voicing_lp,thresh))
            C_lp_m = C_lp_m + np.where(voicing_lp != 0)[0].shape[0]
        else:
            # list_rca_female_lp.append(sweep_rca(cent,model_cents,voicing_lp,thresh,[0]))
            list_rca_female_lp.append(rca(cent,model_cents,voicing_lp,thresh))
            C_lp_f = C_lp_f + np.where(voicing_lp != 0)[0].shape[0]

        # High pitch estimation
        voicing_hp = np.copy(voicing)
        force_out_of_pitch = np.where(np.logical_or(pitch < 125,pitch > 500)==True)
        voicing_hp[force_out_of_pitch] = 0
        C_hp = C_hp + np.where(voicing_hp != 0)[0].shape[0]

        # list_rca_model_hp.append(sweep_rca(cent,model_cents,voicing_hp,thresh,[0]))
        list_rca_model_hp.append(rca(cent,model_cents,voicing_hp,thresh))

        if "mic_M" in audio_file:
            # list_rca_male_hp.append(sweep_rca(cent,model_cents,voicing_hp,thresh,[0]))
            list_rca_male_hp.append(rca(cent,model_cents,voicing_hp,thresh))
            C_hp_m = C_hp_m + np.where(voicing_hp != 0)[0].shape[0]
        else:
            # list_rca_female_hp.append(sweep_rca(cent,model_cents,voicing_hp,thresh,[0]))
            list_rca_female_hp.append(rca(cent,model_cents,voicing_hp,thresh))
            C_hp_f = C_hp_f + np.where(voicing_hp != 0)[0].shape[0]
        # list_rca_model.append(acc_model)
        # list_rca_crepe.append(acc_crepe)
        # list_rca_lpcnet.append(acc_lpcnet)
        # list_rca_penn.append(acc_penn)
        """

    # list_rca_crepe = np.array(list_rca_crepe)
    # list_rca_model_lp = np.array(list_rca_model_lp)
    # list_rca_male_lp = np.array(list_rca_male_lp)
    # list_rca_female_lp = np.array(list_rca_female_lp)

    # list_rca_model_hp = np.array(list_rca_model_hp)
    # list_rca_male_hp = np.array(list_rca_male_hp)
    # list_rca_female_hp = np.array(list_rca_female_hp)

    list_rca_model_all = np.array(list_rca_model_all)
    list_rca_male_all = np.array(list_rca_male_all)
    list_rca_female_all = np.array(list_rca_female_all)
    # list_rca_lpcnet = np.array(list_rca_lpcnet)
    # list_rca_penn = np.array(list_rca_penn)

    x = PrettyTable()

    x.field_names = ["Experiment", "Mean RPA"]
    x.add_row(["Both all pitches", np.sum(list_rca_model_all)/C_all])
    # x.add_row(["Both low pitches", np.sum(list_rca_model_lp)/C_lp])
    # x.add_row(["Both high pitches", np.sum(list_rca_model_hp)/C_hp])

    x.add_row(["Male all pitches", np.sum(list_rca_male_all)/C_all_m])
    # x.add_row(["Male low pitches", np.sum(list_rca_male_lp)/C_lp_m])
    # x.add_row(["Male high pitches", np.sum(list_rca_male_hp)/C_hp_m])

    x.add_row(["Female all pitches", np.sum(list_rca_female_all)/C_all_f])
    # x.add_row(["Female low pitches", np.sum(list_rca_female_lp)/C_lp_f])
    # x.add_row(["Female high pitches", np.sum(list_rca_female_hp)/C_hp_f])

    print(x)

    return None

def cycle_eval(list_files_pth, noise_type = 'synthetic', noise_dataset = None, list_snr = [-20,-15,-10,-5,0,5,10,15,20], ptdb_dataset_path = None,fraction = 0.1,thresh = 50):
    """
    Cycle through SNR evaluation for list of .pth files
    """
    # list_files = glob.glob('/home/ubuntu/Code/Datasets/SPEECH DATA/combined_mic_16k_raw/*.raw')
    # dir_f0 = '/home/ubuntu/Code/Datasets/SPEECH DATA/combine_f0_ptdb/'
    # random_shuffle = list(np.random.permutation(len(list_files)))
    list_files = glob.glob(ptdb_dataset_path + 'combined_mic_16k/*.raw')
    dir_f0 = ptdb_dataset_path + 'combined_reference_f0/'
    random.shuffle(list_files)
    list_files = list_files[:(int)(fraction*len(list_files))]

    # list_nfiles = ['DKITCHEN','NFIELD','OHALLWAY','PCAFETER','SPSQUARE','TCAR','DLIVING','NPARK','OMEETING','PRESTO','STRAFFIC','TMETRO','DWASHING','NRIVER','OOFFICE','PSTATION','TBUS']

    dict_models = {}
    list_snr.append(np.inf)
    # thresh = 50

    for f in list_files_pth:
        if (f!='crepe') and (f!='lpcnet'):
            fname = os.path.basename(f).split('_')[0]  + '_' + os.path.basename(f).split('_')[-1][:-4]
            config_path = os.path.dirname(f) + '/' + os.path.basename(f).split('_')[0] + '_' + 'config_' + os.path.basename(f).split('_')[-1][:-4] + '.json'
            with open(config_path) as json_file:
                dict_params = json.load(json_file)
            
            if dict_params['data_format'] == 'if':
                from models import large_if_ccode as model
                pitch_nn = model(dict_params['freq_keep']*3,dict_params['gru_dim'],dict_params['output_dim']).to(device)
            elif dict_params['data_format'] == 'xcorr':
                from models import large_xcorr as model
                pitch_nn = model(dict_params['xcorr_dim'],dict_params['gru_dim'],dict_params['output_dim']).to(device)
            else:
                from models import large_joint as model
                pitch_nn = model(dict_params['freq_keep']*3,dict_params['xcorr_dim'],dict_params['gru_dim'],dict_params['output_dim']).to(device)
            
            pitch_nn.load_state_dict(torch.load(f))

            N = dict_params['window_size']
            H = dict_params['hop_factor']
            freq_keep = dict_params['freq_keep']

            list_mean = []
            list_std = []
            for snr_dB in list_snr:
                C_all = 0
                C_correct = 0
                for idx in tqdm.trange(len(list_files)):
                    audio_file = list_files[idx]
                    file_name = os.path.basename(list_files[idx])[:-4]

                    audio = np.memmap(list_files[idx], dtype=np.int16)/(2**15 - 1)
                    offset = 432
                    audio = audio[offset:]
                    rmse = np.squeeze(librosa.feature.rms(y = audio,frame_length = N,hop_length = H))

                    if noise_type != 'synthetic':
                        list_noisefiles = noise_dataset + '*.wav'
                        noise_file = random.choice(glob.glob(list_noisefiles))
                        n = np.memmap(noise_file, dtype=np.int16,mode = 'r')/(2**15 - 1)
                        rand_range = np.random.randint(low = 0, high = (16000*60*5 - audio.shape[0])) # Last 1 minute of noise used for testing
                        n = n[rand_range:rand_range + audio.shape[0]]
                    else:
                        n = np.random.randn(audio.shape[0])
                        n = random_filter(n)

                    snr_multiplier = np.sqrt((np.sum(np.abs(audio)**2)/np.sum(np.abs(n)**2))*10**(-snr_dB/10))
                    audio = audio + snr_multiplier*n

                    spec = stft(x = np.concatenate([np.zeros(160),audio]), w = 'boxcar', N = N, H = H).T
                    phase_diff = spec*np.conj(np.roll(spec,1,axis = -1))
                    phase_diff = phase_diff/(np.abs(phase_diff) + 1.0e-8)
                    idx_save = np.concatenate([np.arange(freq_keep),(N//2 + 1) + np.arange(freq_keep),2*(N//2 + 1) + np.arange(freq_keep)])
                    feature = np.concatenate([np.log(np.abs(spec) + 1.0e-8),np.real(phase_diff),np.imag(phase_diff)],axis = 0).T
                    feature_if = feature[:,idx_save]

                    data_temp = np.memmap('./temp.raw', dtype=np.int16, shape=(audio.shape[0]), mode='w+')
                    # data_temp[:audio.shape[0]] = (audio/(np.max(np.abs(audio)))*(2**15 - 1)).astype(np.int16)
                    data_temp[:audio.shape[0]] = ((audio)*(2**15 - 1)).astype(np.int16)

                    subprocess.run(["../../../lpcnet_xcorr_extractor", './temp.raw', './temp_xcorr.f32'])
                    feature_xcorr = np.flip(np.fromfile('./temp_xcorr.f32', dtype='float32').reshape((-1,256),order = 'C'),axis = 1)
                    ones_zero_lag = np.expand_dims(np.ones(feature_xcorr.shape[0]),-1)
                    feature_xcorr = np.concatenate([ones_zero_lag,feature_xcorr],axis = -1)

                    os.remove('./temp.raw')
                    os.remove('./temp_xcorr.f32')

                    if dict_params['data_format'] == 'if':
                        feature = feature_if
                    elif dict_params['data_format'] == 'xcorr':
                        feature = feature_xcorr
                    else:
                        indmin = min(feature_if.shape[0],feature_xcorr.shape[0])
                        feature = np.concatenate([feature_xcorr[:indmin,:],feature_if[:indmin,:]],-1)

                    pitch_file_name = dir_f0 + "ref" + os.path.basename(list_files[idx])[3:-4] + ".f0"
                    pitch = np.loadtxt(pitch_file_name)[:,0]
                    voicing = np.loadtxt(pitch_file_name)[:,1]
                    indmin = min(voicing.shape[0],rmse.shape[0],pitch.shape[0])
                    pitch = pitch[:indmin]
                    voicing = voicing[:indmin]
                    rmse = rmse[:indmin]
                    voicing = voicing*(rmse > 0.05*np.max(rmse))
                    if "mic_F" in audio_file:
                        idx_correct = np.where(pitch < 125)
                        voicing[idx_correct] = 0

                    cent = np.rint(1200*np.log2(np.divide(pitch, (16000/256), out=np.zeros_like(pitch), where=pitch!=0) + 1.0e-8)).astype('int')

                    # if os.path.basename(f) == 'crepe':
                    # elif (model == 'crepe'):
                        # _, model_frequency, _, _ = crepe.predict(np.concatenate([np.zeros(80),audio]), 16000, viterbi=True,center=True,verbose=0)
                        # model_cents = 1200*np.log2(model_frequency/(16000/256))
                    # else:
                    # else:
                    model_cents = pitch_nn(torch.from_numpy(np.copy(np.expand_dims(feature,0))).float().to(device))
                    model_cents = 20*model_cents.argmax(dim=1).cpu().detach().squeeze().numpy()
                    # model_cents = np.roll(model_cents,-1*3)

                    num_frames = min(cent.shape[0],model_cents.shape[0])
                    pitch = pitch[:num_frames]
                    cent = cent[:num_frames]
                    voicing = voicing[:num_frames]
                    model_cents = model_cents[:num_frames]

                    voicing_all = np.copy(voicing)
                    # Forcefully make regions where pitch is <65 or greater than 500 unvoiced for relevant accurate pitch comparisons for our model
                    force_out_of_pitch = np.where(np.logical_or(pitch < 65,pitch > 500)==True)
                    voicing_all[force_out_of_pitch] = 0
                    C_all = C_all + np.where(voicing_all != 0)[0].shape[0]

                    # list_rca_model_all.append(sweep_rca(cent,model_cents,voicing_all,thresh,[0]))
                    C_correct = C_correct + rca(cent,model_cents,voicing_all,thresh)
                    # list_rca_model_all.append(np.count_nonzero(np.where(np.abs(cent - model_cents))))
                list_mean.append(C_correct/C_all)
        else:
            fname = f
            list_mean = []
            list_std = []
            for snr_dB in list_snr:
                C_all = 0
                C_correct = 0
                for idx in tqdm.trange(len(list_files)):
                    audio_file = list_files[idx]
                    file_name = os.path.basename(list_files[idx])[:-4]

                    audio = np.memmap(list_files[idx], dtype=np.int16)/(2**15 - 1)
                    offset = 432
                    audio = audio[offset:]
                    rmse = np.squeeze(librosa.feature.rms(y = audio,frame_length = 320,hop_length = 160))

                    if noise_type != 'synthetic':
                        list_noisefiles = noise_dataset + '*.wav'
                        noise_file = random.choice(glob.glob(list_noisefiles))
                        n = np.memmap(noise_file, dtype=np.int16,mode = 'r')/(2**15 - 1)
                        rand_range = np.random.randint(low = 0, high = (16000*60*5 - audio.shape[0])) # Last 1 minute of noise used for testing
                        n = n[rand_range:rand_range + audio.shape[0]]
                    else:
                        n = np.random.randn(audio.shape[0])
                        n = random_filter(n)

                    snr_multiplier = np.sqrt((np.sum(np.abs(audio)**2)/np.sum(np.abs(n)**2))*10**(-snr_dB/10))
                    audio = audio + snr_multiplier*n

                    if (f == 'crepe'):
                        _, model_frequency, _, _ = crepe.predict(np.concatenate([np.zeros(80),audio]), 16000, viterbi=True,center=True,verbose=0)
                        model_cents = 1200*np.log2(model_frequency/(16000/256) + 1.0e-8)
                    else:
                        data_temp = np.memmap('./temp.raw', dtype=np.int16, shape=(audio.shape[0]), mode='w+')
                        # data_temp[:audio.shape[0]] = (audio/(np.max(np.abs(audio)))*(2**15 - 1)).astype(np.int16)
                        data_temp[:audio.shape[0]] = ((audio)*(2**15 - 1)).astype(np.int16)

                        subprocess.run(["../../../lpcnet_xcorr_extractor", './temp.raw', './temp_xcorr.f32', './temp_period.f32'])
                        feature_xcorr = np.fromfile('./temp_period.f32', dtype='float32')
                        model_cents = 1200*np.log2((256/feature_xcorr +  1.0e-8) + 1.0e-8)

                        os.remove('./temp.raw')
                        os.remove('./temp_xcorr.f32')
                        os.remove('./temp_period.f32')


                    pitch_file_name = dir_f0 + "ref" + os.path.basename(list_files[idx])[3:-4] + ".f0"
                    pitch = np.loadtxt(pitch_file_name)[:,0]
                    voicing = np.loadtxt(pitch_file_name)[:,1]
                    indmin = min(voicing.shape[0],rmse.shape[0],pitch.shape[0])
                    pitch = pitch[:indmin]
                    voicing = voicing[:indmin]
                    rmse = rmse[:indmin]
                    voicing = voicing*(rmse > 0.05*np.max(rmse))
                    if "mic_F" in audio_file:
                        idx_correct = np.where(pitch < 125)
                        voicing[idx_correct] = 0

                    cent = np.rint(1200*np.log2(np.divide(pitch, (16000/256), out=np.zeros_like(pitch), where=pitch!=0) + 1.0e-8)).astype('int')
                    num_frames = min(cent.shape[0],model_cents.shape[0])
                    pitch = pitch[:num_frames]
                    cent = cent[:num_frames]
                    voicing = voicing[:num_frames]
                    model_cents = model_cents[:num_frames]

                    voicing_all = np.copy(voicing)
                    # Forcefully make regions where pitch is <65 or greater than 500 unvoiced for relevant accurate pitch comparisons for our model
                    force_out_of_pitch = np.where(np.logical_or(pitch < 65,pitch > 500)==True)
                    voicing_all[force_out_of_pitch] = 0
                    C_all = C_all + np.where(voicing_all != 0)[0].shape[0]

                    # list_rca_model_all.append(sweep_rca(cent,model_cents,voicing_all,thresh,[0]))
                    C_correct = C_correct + rca(cent,model_cents,voicing_all,thresh)
                    # list_rca_model_all.append(np.count_nonzero(np.where(np.abs(cent - model_cents))))
                list_mean.append(C_correct/C_all)
        dict_models[fname] = {}
        dict_models[fname]['list_SNR'] = list_mean[:-1]
        dict_models[fname]['inf'] = list_mean[-1]
    
    return dict_models