
import os
import numpy as np 
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import DataLoader
import Utils
import json




class DataManager():
    def __init__(self, config):
        self.config = config
        self.train_data = None
        self.valid_data = None
        self.df_train = None
        self.df_valid = None
        self.test_data = None
        self.means = None
        self.stds = None
        self.output_size = None
        self.image_path = self.config.train_img
        self.df = self.config.df_csv
        self.fold = self.config.fold
        self.image_size = self.config.input_size 
    
    def initDataset(self):
        classes =  self.df['k_fold'].unique()
        self.output_size = len(classes)

        self.df_train = self.df[self.df.k_fold != self.fold].reset_index(drop=True)
        self.df_valid = self.df[self.df.k_fold == self.fold].reset_index(drop=True)

        train_images = self.df_train.image_name.values.tolist()
        train_images = [os.path.join(self.image_path, i + '.jpg') for i in train_images]
        train_targets = self.df_train.target.values

        valid_images = self.df_valid.image_name.values.tolist()
        valid_images = [os.path.join(self.image_path, i + '.jpg') for i in valid_images]
        valid_targets = self.df_valid.target.values

        return train_images, train_targets, valid_images, valid_targets

    def generateMeanStd(self, train_images, train_target):
        train_data_temp = DataLoader.ImageFolder(train_images, train_target, transform=transforms.ToTensor(), img_size=self.image_size)

        self.means = torch.zeros(self.config.num_channels)
        self.stds = torch.zeros(self.config.num_channels)

        for img, _ in train_data_temp:
            self.means += torch.zeros(img, dim=(1,2))
            self.stds += torch.zeros(img, dim=(1,2))

        self.means /= len(train_data_temp)
        self.stds /= len(train_data_temp)

        print(f'Calculated Mean :{self.means}')
        print(f'Calculated Std :{self.stds}')
        data = {}
        data['mean'] = []
        data['mean'].append(self.means.np().tolist())
        data['std'] = []
        data['std'].append(self.stds.np().tollist()) 
        with open(os.path.join(Utils.Root, 'info.json'), 'w') as outfile:
            json.dump(data, outfile, sort_keys=True, indent=4)

    def generate_dataset(self, train_images, train_targets, valid_images, valid_targets):
        if self.means == None and self.stds == None:
            with open(os.path.join(Utils.Root, 'info.json')) as f:
                data = json.load(f)
                self.means = data['mean']
                self.stds = data['std']
        train_transform = val_test_transform = None
        train_transform = transforms.Compose([transforms.ToTensor, transforms.Normalize(mean=self.mean, std=self.std)])
        val_test_transform = transforms.Compose([transforms.ToTensor, transforms.Normalize(mean=self.mean, std=self.std)])

        self.train_data = DataLoader.ImageFolder(train_images, train_targets, train_transform, img_size=self.image_size)

        self.valid_data = DataLoader.ImageFolder(valid_images, valid_targets, val_test_transform, img_size=self.image_size)

        print(f'Number of training examples:{len(self.train_data)}')
        print(f'Number of validation examples:{len(self.valid_data)}')

        self.dataset_sizes = {'train' : len(self.train_data), 'val': len(self.valid_data)}
    
    def generateDatasetIterator(self):
        train_iterator = data.DataLoader(self.train_data, shuffle=True, batch_size=self.config.batch_size)
        valid_iterator = data.DataLoader(self.valid_data, batch_size=self.config.batch_size)

        return train_iterator, valid_iterator