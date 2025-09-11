from dataset_tools import ACTIONS, preprocess_raw_eeg

from brainflow import BoardShim, BrainFlowInputParams, BoardIds
from matplotlib import pyplot as plt
from timeit import default_timer as timer
from tensorflow import keras
import numpy as np
import threading
import argparse
import time
import cv2
import os

class GraphicalInterface:
    # huge thanks for the GUI to @Sentdex: https://github.com/Sentdex/BCI
    def __init__(self, WIDTH=500, HEIGHT=500, SQ_SIZE=40, MOVE_SPEED=5):
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.SQ_SIZE = SQ_SIZE
        self.MOVE_SPEED = MOVE_SPEED

        self.square = {'x1': int(int(WIDTH) / 2 - int(SQ_SIZE / 2)),
                       'x2': int(int(WIDTH) / 2 + int(SQ_SIZE / 2)),
                       'y1': int(int(HEIGHT) / 2 - int(SQ_SIZE / 2)),
                       'y2': int(int(HEIGHT) / 2 + int(SQ_SIZE / 2))}

        self.box = np.ones((self.square['y2'] - self.square['y1'],
                            self.square['x2'] - self.square['x1'], 3)) * np.random.uniform(size=(3,))
        self.horizontal_line = np.ones((HEIGHT, 10, 3)) * np.random.uniform(size=(3,))
        self.vertical_line = np.ones((10, WIDTH, 3)) * np.random.uniform(size=(3,))