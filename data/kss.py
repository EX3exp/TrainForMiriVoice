import numpy as np
import os
import tgt
from scipy.io.wavfile import read
import pyworld as pw
import torch
import audio as Audio
from utils import get_alignment, standard_norm, remove_outlier, average_by_duration
import hparams as hp
from jamo import h2j
import codecs
import hparams as hp
from shutil import rmtree, move
mirivoice_directory_path = '/content/drive/MyDrive/MiriVoice'
from sklearn.preprocessing import StandardScaler

def build_from_path(in_dir, out_dir, meta):
    train, val, val_list = list(), list(), list()
    
    scalers = [StandardScaler(copy=False) for _ in range(3)]	# scalers for mel, f0, energy

    n_frames = 0
    
    
    #move files in val to WavsAndLabs and remove val folder
    files_from_val = 0
    for root, directories, files in os.walk(os.path.join(in_dir, 'val')):
      for file in files:
          if '.wav' in file:
            val_list.append(file)
            print(f"{file} >>> Changing directory to WavsAndLabs from val...")
            move(os.path.join(in_dir, 'val', file), os.path.join(in_dir, 'WavsAndLabs'))
            move(os.path.join(in_dir, 'val', file.replace('wav', 'lab')), os.path.join(in_dir, 'WavsAndLabs'))
            files_from_val += 1
     
    if files_from_val != 0:
        rmtree(os.path.join(in_dir, 'val'))
        print("{os.path.join(in_dir, 'val')}: Directory removed successfully.")
        
    #move files in train to WavsAndLabs and remove train folder
    files_from_train = 0
    for root, directories, files in os.walk(os.path.join(in_dir, 'train')):
      for file in files:
          if '.wav' in file:
            print(f"{file} >>> Changing directory to WavsAndLabs from train...")
            move(os.path.join(in_dir, 'train', file), os.path.join(in_dir, 'WavsAndLabs'))
            move(os.path.join(in_dir, 'train', file.replace('wav', 'lab')), os.path.join(in_dir, 'WavsAndLabs'))
            files_from_train += 1
     
    if files_from_train != 0:
        rmtree(os.path.join(in_dir, 'train'))
        print("{os.path.join(in_dir, 'train')}: Directory removed successfully.")
        
       
    #collect validations and trains       
    with open(os.path.join(in_dir, meta)) as f:
        meta_list = f.readlines()
        
    meta_max_num = len(meta_list)
    print("Read metadata successfully: {meta_max_num}lines exists.")
    
    for index, line in enumerate(meta_list):
        basename, text = line.split('|')   
        print('** Process: {index + 1} / {meta_max_num}')
        ret = process_utterance(in_dir, out_dir, basename, scalers)

        if ret is None:
            print("Notice: While processing utterance, returned None.")
            continue
        else:
            info, n = ret
      
        if basename in val_list:
            val.append(info)
            print('>> detected: validation sentence.')

        else: 
            train.append(info)
            print('>> detected: train sentence.')
                
      
    n_frames += n
    if len(val) == 0:
        print("Notice: There's no validation sentences in your dataset. plz check.")
    if len(train) == 0:
        print("Notice: There's no train sentences in your dataset. plz check.")
        
    param_list = [np.array([scaler.mean_, scaler.scale_]) for scaler in scalers]
    param_name_list = ['mel_stat.npy', 'f0_stat.npy', 'energy_stat.npy']
    [np.save(os.path.join(out_dir, param_name), param_list[idx]) for idx, param_name in enumerate(param_name_list)]

    return [r for r in train if r is not None], [r for r in val if r is not None]


def process_utterance(in_dir, out_dir, basename, scalers):
    wav_path = os.path.join(in_dir, 'WavsAndLabs', f'{basename}')
    textgrid_name= basename.replace('wav', 'TextGrid')

    # Get alignments
    textgrid = tgt.io.read_textgrid(f'{mirivoice_directory_path}/Dataset/PreprocessedDatas/TextGrids/{textgrid_name}')
    phone, duration, start, end = get_alignment(textgrid.get_tier_by_name('phones'))
    print(f"phone : {phone}\nduration: {duration}\nstart : {start}\n,end : {end}\n")

    text = '{'+ '}{'.join(phone) + '}' # '{A}{B}{$}{C}', $ represents silent phones
    text = text.replace('{$}', ' ')    # '{A}{B} {C}'
    text = text.replace('}{', ' ')     # '{A B} {C}'


    if start >= end:
        print(f"Notice: {basename} has no length!")
        return None

    # Read and trim wav files
    _, wav = read(wav_path)
    wav = wav[int(hp.sampling_rate*start):int(hp.sampling_rate*end)].astype(np.float32)

    # Compute fundamental frequency
    f0, _ = pw.dio(wav.astype(np.float64), hp.sampling_rate, frame_period=hp.hop_length/hp.sampling_rate*1000)
    f0 = f0[:sum(duration)]

    # Compute mel-scale spectrogram and energy
    mel_spectrogram, energy = Audio.tools.get_mel_from_wav(torch.FloatTensor(wav))
    mel_spectrogram = mel_spectrogram.numpy().astype(np.float32)[:, :sum(duration)]
    energy = energy.numpy().astype(np.float32)[:sum(duration)]

    f0, energy = remove_outlier(f0), remove_outlier(energy)
    f0, energy = average_by_duration(f0, duration), average_by_duration(energy, duration)

    if mel_spectrogram.shape[1] >= hp.max_seq_len:
        print("Notice: {basename}'s Mel Spectogram is longer than max sequence length!")
        return None

    # Save alignment
    ali_filename = '{}-ali-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'alignment', ali_filename), duration, allow_pickle=False)

    # Save fundamental prequency
    f0_filename = '{}-f0-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'f0', f0_filename), f0, allow_pickle=False)

    # Save energy
    energy_filename = '{}-energy-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'energy', energy_filename), energy, allow_pickle=False)

    # Save spectrogram
    mel_filename = '{}-mel-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'mel', mel_filename), mel_spectrogram.T, allow_pickle=False)
   
    mel_scaler, f0_scaler, energy_scaler = scalers

    mel_scaler.partial_fit(mel_spectrogram.T)
    f0_scaler.partial_fit(f0[f0!=0].reshape(-1, 1))
    energy_scaler.partial_fit(energy[energy != 0].reshape(-1, 1))

    return '|'.join([basename, text]), mel_spectrogram.shape[1]
