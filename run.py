## Learning Rate Scheduler
##

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
from torch.optim import lr_scheduler
import argparse

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

### Parser
parser = argparse.ArgumentParser()
parser.add_argument('--config',
    default='',
    help='verbose flag' )
args = parser.parse_args()

if args.config == "":
    raise Exception("Must specify config")
# Check config folder
if args.config[-5:]!=".yaml":
    args.config += ".yaml"
if not os.path.exists(args.config):
    args.config = os.path.join("./config", args.config)

config = cutils.get_config(args.config)

# Whether to train on a gpu
train_on_gpu = cuda.is_available()
multi_gpu = True if cuda.device_count() > 1 else False
if train_on_gpu:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


print(config)
#print(os.listdir(config["train_folder"]))

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

# test_dataset = data_loader.CarDataLoader(csv_file=r"../data/carvana/metadata.csv",
#                                            root_dir=r"../data/carvana/masked_images_small",
#                                            transform=prep_data.image_transforms["test"])
test_dataset = transformed_dataset


# this returns data, target
dataloaders = {
    'train': DataLoader(transformed_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"]),
     'test': DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"]),
     'validate': DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
}

#train_loader = torch.utils.data.DataLoader(dataset_h5(train_file), batch_size=16, shuffle=True)

classes = transformed_dataset.classes_count()


## Define optimizer
class_weights = torch.Tensor(transformed_dataset.class_weights) if config["class_weighting"] else None
criterion = nn.CrossEntropyLoss(weight=class_weights)

## Load checkpoint as needed
exp_lr_scheduler = None

# Load model
model = None

if not config["load_checkpoint_path"] is None:
    model, optimizer = train.load_checkpoint(path=config["load_checkpoint_path"], train_on_gpu=train_on_gpu, multi_gpu=multi_gpu)

    ## Reset scheduler
    if not model is None and config["scheduler_step"] and config["gamma"] and not config["load_schedule"]:
        model.scheduler = lr_scheduler.StepLR(optimizer, step_size=config["scheduler_step"], gamma=config["gamma"])


# If model was not loaded successfully, create new one
if model is None:
    model = train.initialize_model(config["model"], n_classes=classes, train_on_gpu=train_on_gpu, multi_gpu=multi_gpu)
    model.class_to_idx = transformed_dataset.class_to_idx
    model.idx_to_class = transformed_dataset.idx_to_class
    optimizer = optim.Adam(model.parameters())
    model.scheduler = None
    # Decay LR by a factor of 0.7 every 5 epochs
    if config["scheduler_step"] and config["gamma"]:
        model.scheduler = lr_scheduler.StepLR(optimizer, step_size=config["scheduler_step"], gamma=config["gamma"])

## Train loop
trainiter = iter(dataloaders['train'])
features, labels = next(trainiter)

chk_pt = cutils.checkpoint(config["checkpoint_folder"])
print("Checkpoint_path: {}".format(chk_pt))

for pg in optimizer.param_groups:
    print("LR {}".format(pg['lr']))

if config["train"]:
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
        train_on_gpu=train_on_gpu,
        early_stopping=config["early_stopping"])

if config["test"]:
    train.validate(model, criterion, dataloaders["train"], train_on_gpu=train_on_gpu)


