"""
MMSBR@TKDE2023 Beyond Co-occurrence: Multi-modal Session-based Recommendation
convert image to text using torchvision
@author: Xiaokun Zhang
"""

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
from torchvision import models
from PIL import Image

number_text = 2

batch_size = 512
datasets_name = 'Cell_Phones_and_Accessories'
image_path = datasets_name + '/img/'

asin_list_path = datasets_name +  '/asinlist.npy'
asin_list = np.load(asin_list_path,allow_pickle=True)



preprocess = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
classNet = models.googlenet(pretrained=True)
classNet.eval()

classes = list()
with open('class.txt') as f:
    lines = f.readlines()
    for line in lines:
        classes.append(line.strip())
img_name_list = asin_list
img2text = {}
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
        input_batch = torch.unsqueeze(input_tensor,0)
        out = classNet(torch.Tensor(input_batch))
        _, index = torch.sort(out, descending=True)
        # tok-2 as text
        text_list = ''
        for i in range(number_text):
            text_list += classes[index[0][i]] + ' '
        img2text[img_temp] = text_list
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
        input_batch = torch.unsqueeze(input_tensor, 0)
        out = classNet(torch.Tensor(input_batch))
        _, index = torch.sort(out, descending=True)
        img2text[img_temp] = classes[index[0][0]] + classes[index[0][1]]


save_path = datasets_name +  '/img2text.npy'
np.save(save_path, img2text)

print("#not_RGB: ", str(not_RGB))
print("#total image: ", str(len(img_name_list_good)))
print("error image name:")
print(error_img_name)
print('img 2 text done')