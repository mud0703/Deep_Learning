import os
import numpy as np
import torch
import argparse
import random
import yaml
import sys
import Utils
import DataManager
import NetworkManager
from easydict import Easydict as edict
from termcolor import colored

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def config_file(filename):
    parser = None
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            parser = edict(yaml.load(f))
        for x in parser:
            print('{}: {}'.format(x, parser[x]))
        msg = 'Config file: {filename} was parsed successfully'
    else:
        Utils.create_config_file()
        msg = f"Config file: {filename} has been created successfully, edit if needed."
    return parser, msg

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    parser = argparse.ArgumentParser()
    config = argparse.ArgumentParser()
    config, msg = config_file('config.yml')
    print(colored(msg, 'green'))
    if config == None:
        sys.exit(0)
    data_manager = DataManager.DataManager(config)
    if not os.path.exists(os.path.join(Utils.Root, 'info.json')):
        data_manager.initDataset()
        data_manager.generateMeanStd()
    data_manager.generateDataset()
    train_iterator, val_iterator = data_manager.generateDatasetIterator()
    data_iterator = {}
    data_iterator['train'] = train_iterator
    data_iterator['val'] = val_iterator
    NetworkManager.train('resnet18', config, use_cuda, dataset=data_iterator, dataset_size=data_manager.dataset_sizes, num_classes = data_manager.output_size)
    print('$$$$ Finish §§§§§')
