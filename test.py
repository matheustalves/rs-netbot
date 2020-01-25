import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mss import mss
from pynput.mouse import Button, Controller
import time

bbox = {'top': 60, 'left': 40, 'width': 470, 'height': 282}
sct = mss()


sct_img = sct.grab(bbox)
output = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGR2GRAY)
cv2.waitKey(0)

start_row, start_col = int(0), int(0)

img_slices = []

for row in range(0,3):
    end_row = int(94*(1+row))
    for col in range(0,5):
        end_col = int(94*(1+col))
        cropped = output[start_row:end_row, start_col:end_col]
        img_slices.append(cropped)
        start_col = end_col
    start_row = end_row
    start_col = int(0)

print(img_slices)