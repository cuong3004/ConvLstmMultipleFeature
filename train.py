from custom_data import CustomData
import torch 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import librosa.display
from model import CnnLstm, CNNModel
from torch.utils.data import DataLoader 
import pytorch_lightning as pl
from litmodule import LitClassification
from torchvision.models import mobilenet_v2
import torch.nn as nn


batch_size = 32
num_classes = 10

melspectrogram_parameters = {
        "n_mels": 128,
        "fmin": 40,
        # "fmax": 32000
    }
npobj = np.load("normalize.npz")
mean, std = npobj['mean'], npobj['std']

print(mean, std)

transform_audio = A.Compose([
    #      RandomAudio(always_apply=True),
         NoiseInjection(p=1),
         ShiftingTime(p=0.5),
         PitchShift(p=0.5),
         MelSpectrogram(parameters=melspectrogram_parameters,always_apply=True),
        #  SpecAugment(p=0.6),
    #      SpectToImage(always_apply=True)
    ])

transform_image = A.Compose([
    A.Normalize(mean=127.5, std=127.5),
    ToTensorV2()
])


dataset = CustomData(
                csv_file="UrbanSound8K.csv",
                data_dir="urbansound8k",
                transform_audio=transform_audio,
                transform_image=transform_image)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# dataset[1]
# dataset[2]
# dataset[3]

# dataset.calculate_normalize()

# for id, (image, label) in tqdm(enumerate(dataset)):
#     # librosa.display.specshow(image)
#     # plt.savefig(f"image/{id}_{label}.png")
#     # plt.clf()
#     print(image.shape)
#     if id == 10:
#         break

# model_cnn = CnnLstm(cnn_model, lstm_model)
# cnn_model = mobilenet_v2()
# cnn_model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
# cnn_model.classifier[1] = torch.nn.Linear(1280,num_classes)
cnn_model = CNNModel()

model_cnnlstm = CnnLstm(cnn_model)
model = LitClassification(model_cnnlstm)

trainer = pl.Trainer(
                # gpus=1, 
                # callbacks = callbacks, 
                # max_epochs=epoch,
                )
trainer.fit(model, dataloader)