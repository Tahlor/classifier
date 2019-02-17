import pandas as pd
from torchvision import transforms, datasets, models
#from skimage import io, transform
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import cutils

def get_data(data=r"../data/carvana/metadata.csv"):
    df = pd.read_csv(data)
    return df


means = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
stds = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

def image_to_tensor(image_array):
    # crop again
    # img = img.crop((left, top, right, bottom))

    # Convert to numpy, transpose color dimension and normalize
    # img = np.array(img).transpose((2, 0, 1)) / 256

    # Standardization
    image = image_array/256 - means
    image = image / stds
    img_tensor = transforms.ToTensor(image)
    return img_tensor

def process_image(image_path, color=[255,255,255], output_path=None):
    color = np.array(color)

    # read in image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # in BGR
    alpha = image[:,:,3]
    image = image[:,:,0:3]

    print(image.shape)
    # handle alpha channel

    # choose color mask
    image[alpha==0,:]=color

    # crop to picture
    mask_coords = np.where(alpha != 0)
    x_min = np.min(mask_coords[0])
    x_max = np.max(mask_coords[0])
    y_min = np.min(mask_coords[1])
    y_max = np.max(mask_coords[1])

    padding = 40
    x_min = max(0, x_min-padding)
    x_max = min(image.shape[0], x_max + padding)
    y_min = max(0, y_min - padding)
    y_max = min(image.shape[1], y_max + padding)

    # crop to image
    image = image[x_min:x_max, y_min:y_max, :]

    short_axis = np.argmin(image.shape[:2])
    ratio = image.shape[0]/image.shape[1]
    if ratio > 1.25 or ratio < .8:
        diff = int(abs(image.shape[0]-image.shape[1])*.8/2)
        npad = [(0, 0), (0, 0), (0, 0)]
        npad[short_axis] = (diff,diff)
        image = np.pad(image, pad_width=npad, mode='constant', constant_values=0)

        if short_axis==0:
            image[0:diff] = color
            image[-diff:None] = color
        else:
            image[:,0:diff] = color
            image[:,-diff:None] = color

    # resize
    old_x = image.shape[0]
    old_y = image.shape[1]

    if old_x < old_y:
        x_size = 256
        y_size = int(256 * old_y/old_x)
    else:
        y_size = 256
        x_size = int(256 * old_x/old_y)

    image = cv2.resize(image, dsize=(y_size,x_size))

    if not output_path is None:
        cv2.imwrite(output_path, image)

    return image

image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    # Validation does not use augmentation
    'val':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Test does not use augmentation
    'test':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

if __name__=="__main__":
    input_folder = r"../data/carvana/masked_images2/"
    output_folder = r"../data/carvana/masked_images_small/"
    cutils.mkdir(output_folder)
    for ds,ss,fs in os.walk(input_folder):
        for f in fs:
            if f[-4:]==".png":
                output_path = os.path.join(output_folder, f)
                input_path = os.path.join(ds, f)
                if not os.path.exists(output_path):
                    process_image(input_path, output_path=output_path)
