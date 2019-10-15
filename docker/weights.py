#!/usr/bin/env python3

# Silence irrelevant warnings about NumPy size changes
# (See <https://stackoverflow.com/a/40846742>)
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# Download the weights for VGG16
from keras.applications.vgg16 import VGG16
model = VGG16(weights='imagenet')
