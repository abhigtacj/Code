import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, add
from tensorflow.keras.layers import Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm_notebook
from collections import Counter

# Paths
images_directory = 'flickr8k/Images/'
captions_path = 'flickr8k/captions.txt'

# Load captions from file
def load_captions(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # Skip header if present and clean each caption
    captions = [line.strip().lower() for line in lines[1:]]
    return captions

# Tokenize captions
def tokenize_captions(captions):
    tokenizer = Tokenizer(oov_token="<unk>")
    tokenizer.fit_on_texts(captions)
    return tokenizer

# Load and preview captions
captions = load_captions(captions_path)
print("Sample captions:")
for i in range(0, 15, 3):
    print(f"{i}: {captions[i]}")