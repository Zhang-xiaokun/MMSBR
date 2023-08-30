"""
MMSBR@TKDE2023 Beyond Co-occurrence: Multi-modal Session-based Recommendation
pca
@author: Xiaokun Zhang
"""
import json
from datetime import datetime
import os
import time
import pickle
import numpy as np
import time
from sklearn.decomposition import PCA
from tqdm import tqdm


startTime = time.time()
# batch_size = 512
datasets_name = './Sports_and_Outdoors100cold/'

matrix_name_list = ['imgMatrix', 'textMatrix', 'imgTextMatrix', 'textImgMatrix']

dimension = 64
for matrix_name in tqdm(matrix_name_list):
    origin_mat_path = datasets_name + matrix_name + '.npy'
    origin_mat = np.load(origin_mat_path)
    pca = PCA(n_components = dimension)
    pca_matrix = pca.fit_transform(origin_mat)

    save_path = datasets_name +  matrix_name + 'pca.npy'
    np.save(save_path, pca_matrix)

endTime = time.time()
consumeTime = endTime - startTime
print("running time: " + str(consumeTime) + 's')
print('done')