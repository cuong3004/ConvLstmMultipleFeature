#!/usr/bin/env python
# coding: utf-8

# In[1]:


from custom_data import CustomData, CustomDataMel
import torch 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import librosa.display
from model import CnnLstm, CNNModel, LstmModel
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from litmodule import LitClassification
from torchvision.models import mobilenet_v2
import torch.nn as nn
import os
import pandas as pd
from glob import glob
from callbacks import input_monitor_train, input_monitor_valid, checkpoint_callback, early_stop_callback


# In[2]:


batch_size = 64
num_classes = 10
num_workers = 2

npobj = np.load("normalize.npz")
mean, std = npobj['mean'], npobj['std']
print("mean, std : ", mean, std)


# In[3]:


melspectrogram_parameters = {
        "n_mels": 128,
        "fmin": 40,
        # "fmax": 32000
    }


# In[4]:


transform_audio = A.Compose([

         NoiseInjection(p=0.5),
         ShiftingTime(p=0.5),
         PitchShift(p=0.5),
         MelSpectrogram(parameters=melspectrogram_parameters, always_apply=True),
    #      SpectToImage(always_apply=True)
    ])

transform_image = A.Compose([
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])


# In[5]:


dataset = CustomDataMel(
                csv_file="UrbanSound8K.csv",
                data_dir="preprocessing_data",
                transform_audio=transform_audio,
                transform_image = transform_image
                )


# In[6]:


train_len = int(len(dataset)*0.8)
valid_len = len(dataset)-train_len
data_train, data_valid = random_split(dataset,[train_len, valid_len])


train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(data_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)


# In[7]:


cnn_model = mobilenet_v2()
cnn_model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
cnn_model.classifier[1] = torch.nn.Linear(1280,num_classes)
# model_lstm = LstmModel(n_feature=172, num_classes=10, n_hidden=256, n_layers=2)


# In[8]:


model = LitClassification(cnn_model)

callbacks = [input_monitor_train, input_monitor_valid, checkpoint_callback, early_stop_callback]


# In[9]:

gpus = 1 if torch.cuda.is_available() else 0
trainer = pl.Trainer(
                gpus=gpus, 
                callbacks = callbacks, 
                # max_epochs=1,
                )


# In[ ]:


trainer.fit(model, train_loader, valid_loader)


# In[ ]:




