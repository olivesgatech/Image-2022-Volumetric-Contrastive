import numpy as np
import os
import pdb

from os.path import join
import matplotlib.pyplot as plt
from mayavi import mlab

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import pandas  as pd
import cv2 as cv
from tqdm import tqdm
def generate_images_labels(train_dir, test_dir):
    training_data_path = join(train_dir, 'train_seismic.npy')
    training_label_path = join(train_dir, 'train_labels.npy')
    test1_seismic_path = join(test_dir, 'test1_seismic.npy')
    test1_label_path = join(test_dir, 'test1_labels.npy')
    test2_seismic_path = join(test_dir, 'test2_seismic.npy')
    test2_label_path = join(test_dir, 'test2_labels.npy')

    # load train data
    train_seismic = np.load(training_data_path)
    train_labels = np.load(training_label_path)

    # load test1 data
    test1_seismic = np.load(test1_seismic_path)
    test1_labels = np.load(test1_label_path)

    # load test2 data
    test2_seismic = np.load(test2_seismic_path)
    test2_labels = np.load(test2_label_path)

    # concatenate
    seismic_full = np.concatenate((np.concatenate((test1_seismic, train_seismic), axis=0), test2_seismic), axis=1)
    labels_full = np.concatenate((np.concatenate((test1_labels, train_labels), axis=0), test2_labels), axis=1)
    print(seismic_full.shape)

    # splits (only uncomment one of them!)
    # #1st fold
    test_seismic_1, test_labels_1 = seismic_full[:, :300, :], labels_full[:, :300, :]

    # 2nd fold
    test_seismic_2, test_labels_2 = seismic_full[:, -300:, :], labels_full[:, -300:, :]

    # #3rd fold
    test_seismic_3, test_labels_3 = seismic_full[:, 300:600, :], labels_full[:, 300:600, :]

    train_image_dir = '/data/Datasets/F3_Block/data/train/images'
    train_labels_dir = '/data/Datasets/F3_Block/data/train/labels'


    train_seismic = ((train_seismic - train_seismic.min()) / (train_seismic.max() - train_seismic.min()))*255

    test_seismic_1 = ((test_seismic_1 - test_seismic_1.min()) / (test_seismic_1.max() - test_seismic_1.min())) * 255

    test_seismic_2 = ((test_seismic_2 - test_seismic_2.min()) / (test_seismic_2.max() - test_seismic_2.min())) * 255

    test_seismic_3 = ((test_seismic_3 - test_seismic_3.min()) / (test_seismic_3.max() - test_seismic_3.min())) * 255
    # Generate Training Labels
    for i in tqdm(range(0,train_labels.shape[1])):
        image = train_labels[:,i,:].T
        im = Image.fromarray(image).convert('L')
        im = np.array(im)
        print(im)
        return 0
        im.save(join(train_labels_dir,str(i)+'.png'))
    # Generate Test Fold1 Images
    test_fold1_image_dir = '/data/Datasets/F3_Block/data/test_fold_1/images'
    test_fold1_labels_dir = '/data/Datasets/F3_Block/data/test_fold_1/labels'
    for i in tqdm(range(0,test_seismic_1.shape[1])):
        image = test_seismic_1[:,i,:].T
        im = Image.fromarray(image).convert('L')
        im.save(join(test_fold1_image_dir,str(i)+'.png'))

    # Generate Test Fold1 Labels
    for i in tqdm(range(0,test_labels_1.shape[1])):
        image = test_labels_1[:,i,:].T
        im = Image.fromarray(image).convert('L')
        im.save(join(test_fold1_labels_dir,str(i)+'.png'))
    # Generate Test Fold2 Images
    test_fold2_image_dir = '/data/Datasets/F3_Block/data/test_fold_2/images'
    test_fold2_labels_dir = '/data/Datasets/F3_Block/data/test_fold_2/labels'
    for i in tqdm(range(0,test_seismic_2.shape[1])):
        image = test_seismic_2[:,i,:].T
        im = Image.fromarray(image).convert('L')
        im.save(join(test_fold2_image_dir,str(i)+'.png'))
    # Generate Test Fold2 Labels
    for i in tqdm(range(0,test_labels_2.shape[1])):
        image = test_labels_2[:,i,:].T
        im = Image.fromarray(image).convert('L')
        im.save(join(test_fold2_labels_dir,str(i)+'.png'))
    # Generate Test Fold3 Images
    test_fold3_image_dir = '/data/Datasets/F3_Block/data/test_fold_3/images'
    test_fold3_labels_dir = '/data/Datasets/F3_Block/data/test_fold_3/labels'
    for i in tqdm(range(0,test_seismic_3.shape[1])):
        image = test_seismic_3[:,i,:].T
        im = Image.fromarray(image).convert('L')
        im.save(join(test_fold3_image_dir,str(i)+'.png'))
    # Generate Test Fold3 Labels
    for i in tqdm(range(0,test_labels_3.shape[1])):
        image = test_labels_3[:,i,:].T
        im = Image.fromarray(image).convert('L')
        im.save(join(test_fold3_labels_dir,str(i)+'.png'))

def split_csv(train_dir,label_dir):
    column_names = ["Image_Path", "Semantic_Label_Path", "Image_Label"]
    list_images = os.listdir(train_dir)
    image_path_list = []
    label_path_list = []
    index_list = []
    for i in tqdm(range(0,len(list_images))):
        image_path_list.append(join(train_dir,list_images[i]))
        label_path_list.append(join(label_dir,list_images[i]))
        j,_ = list_images[i].split('.')
        index_list.append(j)
    df = pd.DataFrame({'Image_Path': image_path_list, 'Semantic_Label_Path': label_path_list, 'Image_Label': index_list})
    df.to_csv('/home/kiran/Desktop/Dev/Image_2022_Volumetric_Contrastive/SupContrast/csv_files/test_fold_3.csv',index=False)

def train_pos_split(df_dir,N):
    df = pd.read_csv(df_dir)
    df['Image_label'] = pd.cut(df.Image_Label, bins=N, labels=False)
    df.to_csv('/home/kiran/Desktop/Dev/Image_2022_Volumetric_Contrastive/SupContrast/csv_files/train_'+str(N) + '.csv',index=False)

def check_image(path):
    im = cv.imread(path)
    print(im)
def process_volume(img_dir):


    training_label_path = join(img_dir, 'train_labels.npy')
    train_seismic = np.load(training_label_path)
    vol_array = train_seismic
    # Order Files in Correct Orientation of Volume

    for i in range(400, 600, 10):
        mlab.volume_slice(vol_array, plane_orientation='y_axes', slice_index=i, colormap='seismic')

    mlab.savefig(filename='test.png')
    mlab.show()

def format_csv(csv_file):
    df = pd.read_csv(csv_file)
    for i in range(len(df)):
        path_0 = df.iloc[i,0]
        path_1 = df.iloc[i,1]

        split_0 = path_0.split('/')
        split_1 = path_1.split('/')
        new_split_0 = os.path.join(split_0[3],split_0[4],split_0[5],split_0[6],split_0[7])
        new_split_1 = os.path.join(split_1[3], split_1[4], split_1[5], split_1[6], split_1[7])


        df.iloc[i,0] = new_split_0
        df.iloc[i,1] = new_split_1


    df.to_csv(csv_file,index=False)

if __name__ == '__main__':
    train_path = '/data/Datasets/F3_Block/data/train'
    test_path = '/data/Datasets/F3_Block/data/test_once'
    train_image_dir = '/data/Datasets/F3_Block/data/train/images'
    train_labels_dir = '/data/Datasets/F3_Block/data/train/labels'

    test_fold1_image_dir = '/data/Datasets/F3_Block/data/test_fold_1/images'
    test_fold1_labels_dir = '/data/Datasets/F3_Block/data/test_fold_1/labels'

    test_fold2_image_dir = '/data/Datasets/F3_Block/data/test_fold_2/images'
    test_fold2_labels_dir = '/data/Datasets/F3_Block/data/test_fold_2/labels'

    test_fold3_image_dir = '/data/Datasets/F3_Block/data/test_fold_3/images'
    test_fold3_labels_dir = '/data/Datasets/F3_Block/data/test_fold_3/labels'
    df_dir = '//SupContrast/csv_files/train_base.csv'

    csv_analyze = '/home/kiran/Desktop/Dev/Image_2022_Volumetric_Contrastive/SupContrast/csv_files/train_150.csv'

