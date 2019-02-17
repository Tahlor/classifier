import prep_data
import pandas
from torchvision import models
import torch.nn as nn
import train
# PyTorch
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler
import torch.nn as nn

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Data science tools
import numpy as np
import pandas as pd
import os

# Image manipulations
from PIL import Image

# Useful for examining network
#from torchsummary import summary
# Visualizations
import matplotlib.pyplot as plt
import cutils

config = cutils.get_config()

# Whether to train on a gpu
train_on_gpu = cuda.is_available()
multi_gpu = True if cuda.device_count() > 1 else False

print(config)
print(os.listdir(config["train_folder"]))
# Dataloader iterators
data = {
    'train':
    datasets.ImageFolder(root=config["train_folder"], transform=prep_data.image_transforms['train']),
    'test':
    datasets.ImageFolder(root=config["test_folder"], transform=prep_data.image_transforms['test'])
}

dataloaders = {
    'train': DataLoader(data['train'], batch_size=config["batch_size"], shuffle=True),
    'test': DataLoader(data['test'], batch_size=config["batch_size"], shuffle=True)
}

classes = 4
model = train.initialize_model(config["model"], n_classes=classes, train_on_gpu=train_on_gpu, multi_gpu=multi_gpu)

## Class procesing
model.class_to_idx = data['train'].class_to_idx
model.idx_to_class = {
    idx: class_
    for class_, idx in model.class_to_idx.items()
}

list(model.idx_to_class.items())[:10]

## Define optimizer
criterion = nn.NLLLoss()

## Load checkpoint as needed
if not config["load_checkpoint_path"] is None:
    model, optimizer = train.load_checkpoint(path=config["load_checkpoint_path"])
else:
    optimizer = optim.Adam(model.parameters())

## Train loop
trainiter = iter(dataloaders['train'])
features, labels = next(trainiter)

model, history = train.train(
    model,
    criterion,
    optimizer,
    dataloaders['train'],
    valid_loader=None,
    save_file_name=config["checkpoint_folder"],
    max_epochs_stop=5,
    n_epochs=config["epochs"],
    print_every=1,
    train_on_gpu=train_on_gpu)

