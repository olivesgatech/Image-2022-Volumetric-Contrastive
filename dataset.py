import torch.utils.data as data
import os
from PIL import Image
import pandas as pd


class SeismicDataset(data.Dataset):
    def __init__(self, df,img_dir,target_dir, transform,target_transform):

        self.df = pd.read_csv(df)
        self.img_dir = img_dir
        self.target_dir = target_dir
        self.transforms = transform
        self.target_transforms=target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        img_path = os.path.join(self.img_dir,self.df.iloc[idx,0])
        label_path = os.path.join(self.target_dir,self.df.iloc[idx, 1])

        im = Image.open(img_path).convert("L")

        image = self.transforms(im)

        label_im = Image.open(label_path).convert("L")

        label_image = self.target_transforms(label_im)
        label_image = label_image*255

        pos_label = self.df.iloc[idx,2]
        return image, label_image, pos_label