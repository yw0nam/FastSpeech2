# %%
from dataset import Dataset, TextDataset
import yaml
from torch.utils.data import DataLoader
from model.loss import FastSpeech2Loss
# %%
preprocess_config = yaml.load(
    open("./config/LJSpeech/preprocess.yaml", "r"), Loader=yaml.FullLoader
)
train_config = yaml.load(
    open("./config/LJSpeech/train.yaml", "r"), Loader=yaml.FullLoader
)
model_config = yaml.load(
    open("./config/LJSpeech/model.yaml", "r"), Loader=yaml.FullLoader
)
# %%
train_dataset = Dataset(
    "train.txt", preprocess_config, train_config, sort=True, drop_last=True
)
# %%
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=train_dataset.collate_fn,
)
# %%
for i in train_loader:
    meta, inputs = i
    break
# %%
inputs.keys()

# %%
from model.fastspeech2 import FastSpeech2
# %%
model = FastSpeech2(preprocess_config, model_config)
# %%
out = model(**inputs)
# %%

# %%
from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
# %%
vocoder = get_vocoder(model_config, 'cpu')
# %%
vocoder.eval()
# %%
fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
    meta['ids'],
    inputs,
    out,
    vocoder,
    model_config,
    preprocess_config,
)
# %%
wav_reconstruction /  max(abs(wav_reconstruction))
# %%
import torchaudio
import torch
import librosa
# %%
librosa.output.write_wav('./temp.wav', wav_reconstruction /  max(abs(wav_reconstruction)), sr=22050)
# %%
loss = FastSpeech2Loss(preprocess_config, model_config)
# %%
loss(inputs, out)
# %%
a = [1 ,1]
b = [1, 2]
c = [1 ,3]
# %%
temp = [a, b, c]
# %%
a, b, c = temp
# %%
a
# %%
