import tensorflow as tf
import numpy as np
import scipy
from scipy import misc
import glob
from PIL import Image
import os
import matplotlib.pyplot as plt
import librosa
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import (Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, 
                          Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
import pydot
from IPython.display import SVG
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_uniform
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pydub import AudioSegment
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

train_dir = "./data/images_original/"
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir,target_size=(288,432),color_mode="rgba",class_mode='categorical',batch_size=50)

validation_dir = "./data/images_original_test/"
vali_datagen = ImageDataGenerator(rescale=1./255)
vali_generator = vali_datagen.flow_from_directory(validation_dir,target_size=(288,432),color_mode='rgba',class_mode='categorical',batch_size=50)

model = Sequential([
    Conv2D(32, (3,3),input_shape=(228,432,4), activation="relu"),
    MaxPooling2D((2,2)),
    
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    
    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    
    Conv2D(256, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    
    Flatten(),
    Dropout(0.3),
    Dense(9, activation="softmax")
    
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy")

model.fit(train_generator,validation_data=vali_generator, epochs=15)