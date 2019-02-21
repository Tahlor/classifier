from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import os
from skimage import io, transform
import h5py
import numpy as np
import torch
from PIL import Image
import copy

class CarDataLoader(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_files = os.listdir(root_dir)
        self.meta_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # Remove missing labels
        self.meta_data.dropna(subset=['model'], inplace=True)
        self.meta_data["model_code"]=0
        self.meta_data.model_code = self.meta_data.model.astype('category').cat.codes.astype(np.long)

        # Exclude images without labels
        self.verify()
        self.labels = np.array((self.meta_data["model"]).drop_duplicates())

        ## Weight the loss
        self.counts = self.meta_data['model_code'].value_counts(normalize=True)
        self.inverted_weights = 1/self.counts
        self.class_weights = ((self.inverted_weights - self.inverted_weights.mean()) / self.inverted_weights.std() + 1.5).values

        ## Class to index
        self.idx_to_class = dict(zip(self.meta_data.model_code, self.meta_data.model))
        self.class_to_idx = dict(zip(self.meta_data.model, self.meta_data.model_code))
        print(len(self.idx_to_class.items()))
        #self.inverted_weights = self.inverted_weights/self.inverted_weights.sum()
        #print(self.meta_data.model_code)
        #print(self.inverted_weights)
        #print(self.inverted_weights.sum())

    def verify(self):
        bad_files = []
        for img in self.image_files:
            id = img[:-7]
            if not id in self.meta_data["id"].values:
                bad_files.append(img)
        for img in bad_files:
            self.image_files.remove(img)
        print("Bad files: {}".format(bad_files))

    def classes_count(self):
        return len(self.labels)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = self.image_files[idx]
        id = img[:-7]
        img_name = os.path.join(self.root_dir, img)
        #image = io.imread(img_name)
        image = pil_loader(img_name)
        model_idx = self.meta_data[self.meta_data['id'] == id]["model_code"].values[0]

        if self.transform:
            image = self.transform(image)

        #sample = {'image': image, 'target': model.values[0]}
        # the softmax+nnn loss uses one hot
        #one_hot = (model.values[0]==self.labels).astype(int)
        return image, model_idx

    def copy(self, new_folder, transform=None):
        new_loader = copy.deepcopy(self)
        new_loader.root_dir = new_folder
        new_loader.image_files = os.listdir(new_folder)
        new_loader.verify()
        new_loader.transform=transform
        return new_loader

class dataset_h5(torch.utils.data.Dataset):
    def __init__(self, in_file):
        super(dataset_h5, self).__init__()

        self.h5_file = h5py.File(in_file, 'r')
        self.n_images, self.nx, self.ny = self.h5_file['images'].shape
        self.model = self.h5_file.get('model')
        self.make = self.h5_file.get('year')
        self.year = self.h5_file.get('make')
        self.target = self.model

    def __getitem__(self, index):
        input = self.h5_file['images'][index, :, :]
        return input.astype('float32')

    def __len__(self):
        return self.n_images

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def create_h5(path, meta_data, input_folder):
    labels_df = pd.read_csv(meta_data)
    hf = h5py.File(path, 'w')

    for f in os.listdir(input_folder):
        path = os.path.join(input_folder, f)
        img = pil_loader(path)


if __name__=="__main__":
    import matplotlib.pyplot as plt
    meta = r"../data/carvana/metadata.csv"
    image_folder= r"../data/carvana/masked_images_small"
    #create_h5(r"../data/small.h5", meta, image_folder)
    d = CarDataLoader(meta, image_folder)
    #print(d[4])
    #labels = pd.read_csv(meta)
    print(d.classes_count())
    #print(labels)


