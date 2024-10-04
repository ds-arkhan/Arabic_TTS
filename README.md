# TTS-Arabic

TTS model, trained on for direct TTS inference.


## Quick Setup
Required packages:
`torch torchaudio pyyaml`

~ for training: `librosa matplotlib tensorboard`




Download the pretrained weights for the Tacotron2 model for Arabic (https://drive.google.com/drive/folders/1BVz8QsKMxNWHyES1djciqRmFtG5ylhC8?usp=drive_link) then put it in pretrained folder


Either put them into `pretrained/hifigan-asc-v1` or edit the following lines in `configs/basic.yaml`.

```yaml
# vocoder
vocoder_state_path: pretrained/hifigan-asc-v1/hifigan-asc.pth
vocoder_config_path: pretrained/hifigan-asc-v1/config.json
```

## Training the model
Before training, the audio files must be resampled. The model was trained after preprocessing the files using `scripts/preprocess_audio.py`.

To train the model with options specified in the config file run:
```bash
python train.py
# default parameters:
python train.py --config configs/ARA.yaml
```



