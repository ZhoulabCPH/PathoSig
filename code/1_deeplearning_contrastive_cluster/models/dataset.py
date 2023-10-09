import os
import torchvision
import cv2
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset

is_amp = True
import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')

from augmentation import *

############################################################
####### Folds
############################################################
def make_fold_WSI(args):
    """
    Creates train and test dataframes for WSI (Whole Slide Image) data based on the specified fold.
    Returns the train and test dataframes.
    """
    file_name = os.listdir(args.TheLogSavePath)
    if "5Fold_Test_WSI.csv" in file_name:
        train_df = pd.read_csv(args.Train_LogData_WSI, sep=',').iloc[:, 1:]
        Index = np.arange(len(train_df))
        np.random.shuffle(Index)
        train_df = train_df.iloc[Index].reset_index(drop=True)

        test_df = pd.read_csv(args.Test_LogData_WSI, sep=',').iloc[:, 1:]
        Index = np.arange(len(test_df))
        np.random.shuffle(Index)
        test_df = test_df.iloc[Index].reset_index(drop=True)
    else:
        df = pd.read_csv(args.All_LogData)
        Name_Orial_ = [name.split('_')[0]+'_'+name.split('_')[1] for name in df.values[:,0]]
        Name = np.array(list(set(Name_Orial_)))
        arr_list = np.arange(len(Name))
        np.random.shuffle(arr_list)

        Test_List = arr_list[0:int(len(arr_list) // args.num_fold)]
 
        df.loc[:, 'fold'] = -1
        for i in Name[Test_List]:
            indexs = np.squeeze(np.argwhere(i==np.array(Name_Orial_)))
            if indexs.size == 1:
                df.iloc[int(indexs), -1] = 0
            else:
                df.iloc[indexs, -1] = 0
        train_df = df[df.fold == -1].reset_index(drop=True)

        Index = np.arange(len(train_df))
        np.random.shuffle(Index)
        train_df = train_df.iloc[Index].reset_index(drop=True)

        train_df.to_csv(args.Train_LogData_WSI.format(str(args.num_fold)))
        test_df = df[df.fold == 0].reset_index(drop=True)

        Index = np.arange(len(test_df))
        np.random.shuffle(Index)
        test_df = test_df.iloc[Index].reset_index(drop=True)
        test_df.to_csv(args.Test_LogData_WSI.format(str(args.num_fold)))
    return train_df, test_df

def make_fold_Patient(args):
    """
    Creates train and test dataframes for Patient data based on the specified fold.
    Returns the train and test dataframes.
    """
    file_name = os.listdir(args.TheLogSavePath)
    if "5Fold_Test_Patient.csv" in file_name:
        train_df = pd.read_csv(args.Train_LogData_Patient, sep=',').iloc[:, 1:]
        Index = np.arange(len(train_df))
        np.random.shuffle(Index)
        train_df = train_df.iloc[Index].reset_index(drop=True)

        test_df = pd.read_csv(args.Test_LogData_Patient, sep=',').iloc[:, 1:]
        Index = np.arange(len(test_df))
        np.random.shuffle(Index)
        test_df = test_df.iloc[Index].reset_index(drop=True)
    else:
        df = pd.read_csv(args.All_LogData)

        Name_Orial_ = [name.split('_')[0] for name in df.values[:,0]]
        Name = np.array(list(set(Name_Orial_)))
        arr_list = np.arange(len(Name))
        np.random.shuffle(arr_list)

        Test_List = arr_list[0:int(len(arr_list) // args.num_fold)]

        df.loc[:, 'fold'] = -1
        for i in Name[Test_List]:
            indexs = np.squeeze(np.argwhere(i == np.array(Name_Orial_)))
            if indexs.size == 1:
                df.iloc[int(indexs), -1] = 0
            else:
                df.iloc[indexs, -1] = 0
        train_df = df[df.fold == -1].reset_index(drop=True)

        Index = np.arange(len(train_df))
        np.random.shuffle(Index)
        train_df = train_df.iloc[Index].reset_index(drop=True)

        train_df.to_csv(args.Train_LogData_Patient.format(str(args.num_fold)))
        test_df = df[df.fold == 0].reset_index(drop=True)

        Index = np.arange(len(test_df))
        np.random.shuffle(Index)
        test_df = test_df.iloc[Index].reset_index(drop=True)
        test_df.to_csv(args.Test_LogData_Patient.format(str(args.num_fold)))

    return train_df, test_df

def pad_to_multiple(image, multiple=32, min_size=768):
    """
    Pads the image to the specified multiple size with a minimum size.
    Returns the padded image.
    """
    sh, sw, _ = image.shape
    ph = max(min_size, int(np.ceil(sh / 32)) * 32) - sh
    pw = max(min_size, int(np.ceil(sw / 32)) * 32) - sw

    image = np.pad(image, ((0, ph), (0, pw), (0, 0)), 'constant', constant_values=0)
    return image

############################################################
####### Random choice
############################################################
def valid_augment5(image):
    """
    Performs validation data augmentation on the image.
    Returns the augmented image.
    """
    return image

def train_augment5a(image):
    """
    Performs the first set of data augmentation on the image during training.
    Returns the augmented image.
    """
    image = do_random_flip(image)
    image = do_random_rot90(image)
    for fn in np.random.choice([
        lambda image: (image),
        lambda image: do_random_noise(image, mag=0.1),
        lambda image: do_random_contast(image, mag=0.25),
        lambda image: do_random_hsv(image, mag=[0.30, 0.30, 0])
    ], 2): image = fn(image)

    for fn in np.random.choice([
        lambda image: (image),
        lambda image: do_random_rotate_scale(image, angle=45, scale=[0.5, 2]),
    ], 1): image = fn(image)

    return image

def train_augment5b(image):
    """
    Performs the second set of data augmentation on the image during training.
    Returns the augmented image.
    """
    Ori_Image = image
    image = do_random_flip(image)
    image = do_random_rot90(image)

    for fn in np.random.choice([
        lambda image: (image),
        lambda image: do_random_noise(image, mag=0.1),
        lambda image: do_random_contast(image, mag=0.40),
        lambda image: do_random_hsv(image, mag=[0.40, 0.40, 0])
    ], 2): image = fn(image)
    Aug_Image1 = image
    for fn in np.random.choice([
        lambda image: do_random_revolve(image, s=0.5),
        lambda image: do_random_rotate_scale(image, angle=45, scale=[0.80, 2.4]),
    ], 1): image = fn(image)

    return image

class HubmapDataset(Dataset):
    def __init__(self, df, args, augment1=None, augment2=None):
        self.args = args
        data_path = self.args.data_path
        df['image_path'] = df['Sample_Name'].apply(lambda x: os.path.join(data_path, str(x) + '.png'))
        self.df = df
        self.augment1 = augment1
        self.augment2 = augment2
        self.length = len(self.df)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_path = self.df.loc[index, 'image_path']
        img_name = self.df.loc[index, 'Sample_Name']
        image = cv2.imread(img_path)

        image_size = self.args.image_size
        image = image.astype(np.float32) / 255
        image = cv2.resize(image, dsize=(image_size, image_size), interpolation=cv2.INTER_LINEAR)

        image1 = self.augment1(image)
        image2 = self.augment2(image)
        r = {}
        r['index'] = index
        r['image'] = image_to_tensor(image1)
        r['image_Argument'] = image_to_tensor(image2)
        r['name'] = img_name
        return r

tensor_list = ['image','image_Argument']

def image_to_tensor(image, mode='bgr'):
    """
    Converts the image to a PyTorch tensor.
    Returns the tensor.
    """
    if mode == 'bgr':
        image = image[:, :, ::-1]
    x = image
    x = x.transpose(2, 0, 1)
    x = np.ascontiguousarray(x)
    x = torch.tensor(x, dtype=torch.float)
    return x

def tensor_to_image(x, mode='bgr'):
    """
    Converts the tensor to an image.
    Returns the image.
    """
    image = x.data.cpu().numpy()
    image = image.transpose(1, 2, 0)
    if mode == 'bgr':
        image = image[:, :, ::-1]
    image = np.ascontiguousarray(image)
    image = image.astype(np.float32)
    return image

tensor_list = ['image','image_Argument']

def null_collate(batch):
    """
    Collates the batch of data.
    Returns the collated data.
    """
    d = {}
    key = batch[0].keys()
    for k in key:
        v = [b[k] for b in batch]
        if k in tensor_list:
            v = torch.stack(v)
        d[k] = v
    return d
