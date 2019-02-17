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
import data_loader
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
# data = {
#     'train':
#     datasets.ImageFolder(root=config["train_folder"], transform=prep_data.image_transforms['train']),
#     'test':
#     datasets.ImageFolder(root=config["test_folder"], transform=prep_data.image_transforms['test'])
# }
#

transformed_dataset = data_loader.CarDataLoader(csv_file=r"../data/carvana/metadata.csv",
                                           root_dir=r"../data/carvana/masked_images_small",
                                           transform=prep_data.image_transforms["train"])

# this returns data, target
dataloaders = {
    'train': DataLoader(transformed_dataset, batch_size=config["batch_size"], shuffle=True),
     'test': DataLoader(transformed_dataset, batch_size=config["batch_size"], shuffle=True)
}

#train_loader = torch.utils.data.DataLoader(dataset_h5(train_file), batch_size=16, shuffle=True)

classes = transformed_dataset.classes_count()
model = train.initialize_model(config["model"], n_classes=classes, train_on_gpu=train_on_gpu, multi_gpu=multi_gpu)

## Define optimizer
#criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()

## Load checkpoint as needed
if not config["load_checkpoint_path"] is None:
    model, optimizer = train.load_checkpoint(path=config["load_checkpoint_path"])
else:
    optimizer = optim.Adam(model.parameters())

## Train loop
trainiter = iter(dataloaders['train'])
features, labels = next(trainiter)

chk_pt = cutils.checkpoint(config["checkpoint_folder"])
print(chk_pt)
model, history = train.train(
    model,
    criterion,
    optimizer,
    dataloaders['train'],
    valid_loader=None,
    save_file_name=chk_pt,
    max_epochs_stop=5,
    n_epochs=config["epochs"],
    print_every=1,
    train_on_gpu=train_on_gpu)

