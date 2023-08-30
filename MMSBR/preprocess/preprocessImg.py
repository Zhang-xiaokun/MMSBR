"""
MMSBR@TKDE2023 Beyond Co-occurrence: Multi-modal Session-based Recommendation
reference https://github.com/Lornatang/GoogLeNet-PyTorch
convert image to embeddings
@author: Xiaokun Zhang
"""
import json
from datetime import datetime
import os
import time
import pickle
import numpy as np
import pandas as pd
import random
import time
import re
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image

from googlenet_pytorch import GoogLeNet

batch_size = 512
datasets_name = 'Cell_Phones_and_Accessories'

image_path = datasets_name + '/img/'

asin_list_path = datasets_name +  '/asinlist.npy'
asin_list = np.load(asin_list_path,allow_pickle=True)
# Open image
preprocess = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
model = GoogLeNet.from_pretrained('googlenet')
img_name_list = asin_list
img_embeddings = []
img_name_list_good = []

num_img = len(img_name_list)
epoch = num_img//batch_size
not_RGB = 0
error_img_name = []
for i in tqdm(range(epoch)):
    img_temp_list = img_name_list[i*batch_size:(i+1)*batch_size]
    input_list = []
    for img_temp in tqdm(img_temp_list):
        image_temp_path = image_path + img_temp + '.jpg'
        try:
            input_image = Image.open(image_temp_path)
            if input_image.mode != 'RGB':
                not_RGB += 1
                input_image = input_image.convert('RGB')
            input_tensor = preprocess(input_image)
        except Exception as e:
            print('error!')
            error_img_name.append(img_temp)
            continue
        img_name_list_good.append(img_temp)
        input_list.append(input_tensor.tolist())

    input_batch = torch.Tensor(input_list) # [batch_size, 3, 224, 224]
    features = model.extract_features(input_batch)
    img_embeddings += features.tolist()
if num_img%batch_size != 0:
    img_temp_list = img_name_list[epoch * batch_size:]
    input_list = []
    for img_temp in tqdm(img_temp_list):
        image_temp_path = image_path + img_temp + '.jpg'
        try:
            input_image = Image.open(image_temp_path)
            if input_image.mode != 'RGB':
                not_RGB += 1
                input_image = input_image.convert('RGB')
            input_tensor = preprocess(input_image)
        except Exception as e:
            print('error!')
            error_img_name.append(img_temp)
            continue
        img_name_list_good.append(img_temp)
        input_list.append(input_tensor.tolist())

    input_batch = torch.Tensor(input_list)
    features = model.extract_features(input_batch)
    img_embeddings += features.tolist()

save_path = datasets_name +  '/imgMatrix.npy'
np.save(save_path, img_embeddings)

print("#not_RGB: ", str(not_RGB))
print("#total image: ", str(len(img_name_list_good)))
print("error image name:")
print(error_img_name)
print('done')