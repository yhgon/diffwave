# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import torch
import torchaudio as T
import torchaudio.transforms as TT

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from tqdm import tqdm
import os

from params import params


def transform(args, wavefilepath):
    full_dir, filename = os.path.split(wavefilepath) 
    filebody, ext = os.path.splitext(filename)
    melfilepath = os.path.join(args.mel_dir, "{}.npy".format(filebody) )    
    #print(full_filename, full_dir, filename, filebody, ext)

    audio, sr = T.load_wav(wavefilepath)
    if params.sample_rate != sr:
        raise ValueError(f'Invalid sample rate {sr}.')
    audio = torch.clamp(audio[0] / 32767.5, -1.0, 1.0)

    mel_args = {
      'sample_rate': params.sample_rate,  # torchaudio default 16000
      'n_fft':       params.n_fft,        # torchaudio default 400
      'win_length':  params.win_length,   # torchaudio default n_fft
      'hop_length':  params.hop_length,   # torchaudio default win_length/2

      'f_min':       params.f_min,        # torchaudio default 0
      'f_max':       params.f_max,        # torchaudio default None  
      'n_mels':      params.n_mels,       # torchaudio default 128        

      'power':       params.power,        # torchaudio default 2.0    
      'normalized':  params.normalized,   # torchaudio default False
    }
    mel_spec_transform = TT.MelSpectrogram(**mel_args)

    with torch.no_grad():
        spectrogram = mel_spec_transform(audio)
        spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
        spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)

    np.save(melfilepath, spectrogram.cpu().numpy())


def main(args):
    lists = sorted(glob(os.path.join(args.wav_dir, '*.wav')) ) 
    print( len(lists) )
    for i, wavefilepath in enumerate(lists):
        print("DEBUG", i, wavefilepath)
        transform(args, wavefilepath)


if __name__ == '__main__':
    parser = ArgumentParser(description='prepares a dataset to train DiffWave')
    parser.add_argument('-w', '--wav_dir', type=str, default='./',  help='Path to dataset')
    parser.add_argument('-m', '--mel_dir', type=str, default='./',  help='Path to dataset')  

    main(parser.parse_args())
