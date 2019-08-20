import pickle
from collections import Counter
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import os
from PIL import Image
import numpy as np
import pandas as pd
import nsml
from nsml import DATASET_PATH

from utils import get_transforms
from utils import default_loader

from sklearn.model_selection import train_test_split
import glob

# DATASET_PATH = '/home/kwpark_mk2/airush2_temp'

csv_file = os.path.join(DATASET_PATH, 'train', 'train_data', 'train_data')

## check label
# label_data_path = os.path.join(DATASET_PATH, 'train', 
#     os.path.basename(os.path.normpath(csv_file)).split('_')[0] + '_label')

# label = pd.read_csv(label_data_path, dtype={'label': int}, sep='\t')

# print(label.columns)
# print(label.head())
# print('proportion of train: {}'.format(Counter(label.label)))
# print(label.label.value_counts(normalize=True))

## check user profile
df = pd.read_csv(csv_file,
                        dtype={
                            'article_id': str,
                            'hh': int, 'gender': str,
                            'age_range': str,
                            'read_article_ids': str
                        }, sep='\t')
print(df.head())