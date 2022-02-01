from torch.utils.data import DataLoader, Dataset 
import pandas as pd
import os
from utils import *
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm

class CustomData(Dataset):
    def __init__(self, csv_file, transform):
        self.df = pd.read_csv(f"{csv_file}")
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

        path_file = self.df["path"].iloc[idx]
        npobj = np.load(path_file)
        image, label = npobj["image"], npobj["label"]

        if self.transform:
            image = self.transform(image=image)['image']
            label = torch.tensor(label)

        return image, label


class CustomDataMel(Dataset):
    def __init__(self, csv_file, data_dir, transform_audio=None, transform_image=None):
        self.df = pd.read_csv(f"{data_dir}/{csv_file}")
        self.transform_audio = transform_audio
        self.transform_image = transform_image
        self.data_dir = data_dir

        self.processing_df()
        self.audio_manipulation = AudioManipulation(
                                        num_channel=1,
                                        max_ms=4000
                                        ) 
     
    def __len__(self):
        return len(self.df)
    
    def processing_df(self):
        self.df['fold'] = self.df['fold'].astype(str)
        self.df["path"] = self.data_dir+"/fold"+self.df['fold']+"/"+self.df['slice_file_name']
        # print("show df")
        # print(self.df.head())
    
    def calculate_normalize(self):

        mean = 0.
        std = 0.
        print("Doing calculate mean std")
        for idx in tqdm(range(len(self.df))):
            file_audio = self.df["path"].iloc[idx]
            data = self.audio_manipulation.open_tranform_data(file_audio)
            image, _ = self.transform_audio(data=data)['data']

            mean += image.mean()
            std += image.std()

        mean /= len(self.df)
        std /= len(self.df)

        np.savez("normalize.npz", mean=mean, std=std)

    def __getitem__(self, idx):
        file_audio = self.df["path"].iloc[idx]
        data = self.audio_manipulation.open_tranform_data(file_audio)
        label = self.df["classID"].iloc[idx]

        if self.transform_audio:
            image, _ = self.transform_audio(data=data)['data']
            if self.transform_image:
                image = np.expand_dims(image, axis=-1)
                image = self.transform_image(image=image)['image']
        
                label = torch.tensor(label)
        

        return image, label


        
