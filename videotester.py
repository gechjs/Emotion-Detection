import os
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")

# Load pre-trained emotion detection model
model = load_model("best_model.h5")
