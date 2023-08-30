"""

MMSBR@TKDE2023 Beyond Co-occurrence: Multi-modal Session-based Recommendation
Created on 17 Mar, 2022

process the text
convert text to image
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
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

import jax
import jax.numpy as jnp

# check how many devices are available
# jax.local_device_count()

# Load models & tokenizer
from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel
from transformers import CLIPProcessor, FlaxCLIPModel

from flax.jax_utils import replicate
from functools import partial

import random
from dalle_mini import DalleBartProcessor

from flax.training.common_utils import shard_prng_key
import numpy as np
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# os.environ["CUDA_VISIBLE_DEVICES"] = "gpu:0"
torch.cuda.set_device(0)

# Open text

startTime = time.time()
datasets_name = 'Cell_Phones_and_Accessories'
id2text_path = datasets_name +  '/id2text.npy'
id2text_dict = np.load(id2text_path,allow_pickle=True).item()
asin_list_path = datasets_name +  '/asinlist.npy'
asin_list = np.load(asin_list_path,allow_pickle=True)
id_list = []
text_list = []
for key in tqdm(asin_list):
    value = id2text_dict[key]
    text_temp = " ".join(value)
    text_list.append(text_temp)
# # text 输入就是 句子的集合 s = [S1, S2, S3]
id_list = asin_list

# id_list = list(id2text_dict.keys())
# text_list = list(map(str,list(id2text_dict.values())))
total_num = 0
img_fold = 0
fold_num = 5000
batch_size = 50
num_img = len(text_list)
epoch = num_img//batch_size

image_save_path_base = datasets_name + '/textImg/'
if not os.path.exists(image_save_path_base):
    os.makedirs(image_save_path_base)

# dalle-mini applies text to generate image
# dalle-mega
DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"  # can be wandb artifact or 🤗 Hub or local folder or google bucket

# DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0"
DALLE_COMMIT_ID = None
# VQGAN model
VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"



# Load dalle-mini
model, params = DalleBart.from_pretrained(
    DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
)
# model = trans_to_cuda(model)

# Load VQGAN
vqgan, vqgan_params = VQModel.from_pretrained(
    VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
)
# vqgan = trans_to_cuda(vqgan)

params = replicate(params)
vqgan_params = replicate(vqgan_params)

# model inference
@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
def p_generate(
    tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
):
    return model.generate(
        **tokenized_prompt,
        prng_key=key,
        params=params,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        condition_scale=condition_scale,
    )

# decode image
@partial(jax.pmap, axis_name="batch")
def p_decode(indices, params):
    return vqgan.decode_code(indices, params=params)

# create a random key
seed = random.randint(0, 2**32 - 1)
key = jax.random.PRNGKey(seed)

processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)

gen_top_k = None
gen_top_p = None
temperature = None
cond_scale = 10.0





# image_save_path_base = datasets_name + '/textImg/' + str(img_fold) + '/'
# if not os.path.exists(image_save_path_base):
#     os.makedirs(image_save_path_base)
for i in tqdm(range(epoch)):
    id_temp_list = id_list[i*batch_size:(i+1)*batch_size]
    prompts = text_list[i*batch_size:(i+1)*batch_size]

    tokenized_prompts = processor(prompts)
    tokenized_prompt = replicate(tokenized_prompts)
    key, subkey = jax.random.split(key)
    # generate images
    encoded_images = p_generate(
        tokenized_prompt,
        shard_prng_key(subkey),
        params,
        gen_top_k,
        gen_top_p,
        temperature,
        cond_scale,
    )
    # remove BOS
    encoded_images = encoded_images.sequences[..., 1:]
    # decode images
    decoded_images = p_decode(encoded_images, vqgan_params)
    decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
    for img_name, decoded_img in zip(id_temp_list, decoded_images):
        # if total_num % fold_num == 0:
        #     img_fold += 1
        #     image_save_path_base = datasets_name + '/textImg/' + str(img_fold) + '/'
        #     if not os.path.exists(image_save_path_base):
        #         os.makedirs(image_save_path_base)
        img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
        file_path = image_save_path_base + str(img_name) + '.jpg'
        img.save(file_path)
        total_num += 1

if num_img%batch_size != 0:

    # image_save_path_base = datasets_name + '/textImg/x' + '/'
    # if not os.path.exists(image_save_path_base):
    #     os.makedirs(image_save_path_base)
    id_temp_list = id_list[epoch * batch_size:]
    prompts = text_list[epoch * batch_size:]

    tokenized_prompts = processor(prompts)
    tokenized_prompt = replicate(tokenized_prompts)
    key, subkey = jax.random.split(key)
    # generate images
    encoded_images = p_generate(
        tokenized_prompt,
        shard_prng_key(subkey),
        params,
        gen_top_k,
        gen_top_p,
        temperature,
        cond_scale,
    )
    # remove BOS
    encoded_images = encoded_images.sequences[..., 1:]
    # decode images
    decoded_images = p_decode(encoded_images, vqgan_params)
    decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
    for img_name, decoded_img in zip(id_temp_list, decoded_images):
        img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
        file_path = image_save_path_base + str(img_name) + '.jpg'
        img.save(file_path)
        total_num += 1






# number of predictions per prompt

# We can customize generation parameters (see https://huggingface.co/blog/how-to-generate)


# pd_dict = {'item_id':id_list, 'text_weights':text_embeddings}
# data=pd.DataFrame(pd_dict)
# save_path = datasets_name +  '/id_textEmbedding.csv'
# data.to_csv(save_path)

endTime = time.time()
consumeTime = endTime - startTime
print("running time: " + str(consumeTime) + 's')

print('total text: ',str(num_img))
print('total img: ',str(total_num))
print('convert text to image done')

