import numpy as np
import matplotlib.pyplot as plt
import random
import os
import PIL

from keras.utils import np_utils                          # tools for creating one-hot encoding
from keras.models import Sequential                       # Type of model we wish to use
from keras.layers.core import Dense, Dropout, Activation  # Types of layers we wish to use

from skimage.transform import resize                      # Used to scale/resize image arrays

from sklearn.metrics import confusion_matrix              # Used to quickly make confusion matrix