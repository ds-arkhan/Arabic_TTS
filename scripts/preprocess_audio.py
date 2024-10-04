# %%
import os
import torch
import torchaudio
import librosa
import numpy as np
from tqdm import tqdm

# %%

#txtpath = './data/train_phon.txt'
#wav_path = '/home/drmostafa/DataMasry/wav'
#wav_new_path = '/home/drmostafa/DataMasry/wav_new'

txtpath='D:/Arabic-TTS/TTS_EgyptianArabic_Tacotron2/data/train_new_phon.txt'
wav_path='D:/Arabic-TTS/TTS_EgyptianArabic_Tacotron2/data/dataset/train/'
wav_new_path='D:/Arabic-TTS/TTS_EgyptianArabic_Tacotron2/data/train_new/'

# %%

silence_audio_size = 256 * 3

resampler = torchaudio.transforms.Resample(48_000, 22_050,
                                           lowpass_filter_width=1024)

print("outputu",txtpath)
lines = open(txtpath).readlines()

for line in tqdm(lines):

    fname = line.split('" "')[0][1:-1]+"v"
    print(fname)
    fpath = os.path.join(wav_path, fname)
    wave, _ = torchaudio.load(fpath)

    wave = resampler(wave)

    wave_ = wave[0].numpy()
    wave_ = wave_ / np.abs(wave_).max() * 0.999
    wave_ = librosa.effects.trim(
        wave_, top_db=23, frame_length=1024, hop_length=256)[0]
    wave_ = np.append(wave_, [0.]*silence_audio_size)

    torchaudio.save(f'{wav_new_path}/{fname}',
                    torch.Tensor(wave_).unsqueeze(0), 22_050)


# %%
