import os
import glob
from .dataloader import Ti_jpg, Ti_jpg_onehot, Ti_jpg_predict
import csv
import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from pydicom import dcmread
import pydicom
from PIL import Image


def get_predictation_dataset_jpg_ti_label(size, dataset_path='./png_crop',
                                          csv_path='taop-2021/100001/To user/'):
    """获取test set"""
    dataset_csv = os.path.join(csv_path, 'test2_data_info.csv') #修改
    image_path_list = []
    with open(dataset_csv) as csvfile:
        csv_reader = csv.reader(csvfile)
        csv_header = next(csv_reader)
        for row in csv_reader:
            image_path = os.path.join(dataset_path, row[2] + '.png')
            image_path_list.append(image_path)
        csvfile.close()
        # print(image_path_list)
        datasets = Ti_jpg_predict(
            image_list=image_path_list, augmentation=1, size=size)
    return datasets

def get_dataset_jpg_ti(dataset_path='./png_crop', input_size=512,
                       csv_path='taop-2021/100001/To user/', crossvalid=True):
    """获取train set和validation set"""
    dataset_csv = os.path.join(csv_path, 'train2_data_info.csv') #修改
    image_path_list = []
    image_label_list = []
    
    ####
    with open(dataset_csv) as csvfile:
        csv_reader = csv.reader(csvfile)
        csv_header = next(csv_reader)
        for row in csv_reader:
            image_path = os.path.join(dataset_path, row[2] + '.png')
            image_path_list.append(image_path)
            image_label_list.append(row[7])
    csvfile.close()
    if crossvalid == True:
        val_size = 0.1
        train_image_list, val_image_list, train_y_list, val_y_list = \
            train_test_split(image_path_list, image_label_list, test_size=val_size,random_state=1) #
        datasets = {}
        datasets['train'] = Ti_jpg(image_list=train_image_list, label=train_y_list, phase='train', augmentation=1,size=input_size)
        datasets['valid'] = Ti_jpg(image_list=val_image_list, label=val_y_list, phase='valid', augmentation=1,size=input_size)
    else:
        datasets = {}
        datasets['train'] = Ti_jpg(image_list=image_path_list, label=image_label_list, phase='train', augmentation=1,size=input_size)
        datasets['valid'] = []
    val_image_list = []
    # compute weights of different classes
    dict = {}
    for key in image_label_list:
        dict[key] = dict.get(key, 0) + 1
    print(dict)

    return datasets['train'], datasets['valid'], val_image_list


def get_dataset_jpg_ti_onehot(dataset_path='./png_crop', input_size=512,
                              csv_path='taop-2021/100001/To user/', crossvalid=True):
    """获取onehot label的train set和validation set"""
    dataset_csv = os.path.join(csv_path, 'train2_data_info.csv')
    image_path_list = []
    image_label_list = []
    with open(dataset_csv) as csvfile:
        csv_reader = csv.reader(csvfile)
        csv_header = next(csv_reader)
        for row in csv_reader:
            image_path = os.path.join(dataset_path, row[2] + '.png')
            image_path_list.append(image_path)
            image_label_list.append(row[7])
        csvfile.close()

    if crossvalid == False:
        val_size = 0.1
        train_image_list, val_image_list, train_y_list, val_y_list = \
            train_test_split(image_path_list, image_label_list,
                             test_size=val_size, random_state=5)

        datasets = {}
        datasets['train'] = Ti_jpg_onehot(image_list=train_image_list, label=train_y_list, phase='train',
                                          augmentation=1)
        datasets['valid'] = Ti_jpg_onehot(
            image_list=val_image_list, label=val_y_list, phase='valid', augmentation=2)

    # compute weights of different classes
    dict = {}
    for key in image_label_list:
        dict[key] = dict.get(key, 0) + 1
    print(dict)

    return datasets['train'], datasets['valid'], val_image_list


# 
# def convert_label_to_onehot(labels):
#     """将labels转换为one-hot的形式"""
#     grades = []   
#     # for i in range(labels.shape[0]):
#     for i in range(len(labels)):
#         # print(int(labels[i]))
#         grade = []
#         for j in range(1, 6):
#             if j == int(labels[i]):
#                 grade.append(1)
#             else:
#                 grade.append(0)
#         grades.append(grade)
#         # print(grade)

#     grades = np.array(grades)

#     return torch.from_numpy(grades).float()
