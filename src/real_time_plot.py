# This is intended for OpenBCI Cyton Board,
# check: https://brainflow.readthedocs.io for other boards


from functions import ACTIONS, preprocess_raw_eeg

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


class Shared:
    def __init__(self):
        self.sample = None
        self.key = None


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


#############################################################

def acquire_signals():
    count = 0
    while True:
        with mutex:
            # print("acquisition_phase")
            if count == 0:
                time.sleep(2)
                count += 1
            # else:
                time.sleep(0.5)
            # get_current_board_data does not remove personal_dataset from board internal buffer
            # thus allowing us to acquire overlapped personal_dataset and compute more classification over 1 sec
            data = board.get_current_board_data(250)

            sample = []
            eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
            for channel in eeg_channels:
                sample.append(data[channel])

            shared_vars.sample = np.array(sample)

            # print(shared_vars.sample.shape)

            if shared_vars.key == ord("q"):
                break

            # print("sample_acquired")
        time.sleep(0.1)


def compute_signals():
    MODEL_NAME = "models/grey/80.0-72epoch-1764640193-loss-0.53.keras"
    model = keras.models.load_model(MODEL_NAME)
    count_down = 100  # restarts the GUI when reaches 0
    EMA = [-1, -1]  # exponential moving average over the probabilities of the model (2 classes: left/right)
    alpha = 0.4  # coefficient for the EMA
    gui = GraphicalInterface()
    first_run = True

    while True:
        with mutex:
            # print("computing_phase")

            if count_down == 0:
                gui = GraphicalInterface()
                count_down = 100

            env = np.zeros((gui.WIDTH, gui.HEIGHT, 3))

            # prediction on the task
            nn_input, ffts = preprocess_raw_eeg(shared_vars.sample.reshape((1, 8, 250)),
                                                fs=250, lowcut=8, highcut=30, coi3order=0)
            nn_input = nn_input.reshape((1, 8, 250, 1))  # 4D Tensor
            nn_out = model.predict(nn_input)[0]  # this is a probability array

            # Check raw prediction confidence - reset EMA if too low
            raw_confidence = np.max(nn_out)
            if raw_confidence < 0.80:
                # Reset EMA when raw confidence is low (headset likely off or no clear signal)
                EMA = [-1, -1]

            # computing exponential moving average
            if EMA[0] == -1:  # if this is the first iteration (base case)
                for i in range(len(EMA)):
                    EMA[i] = nn_out[i]
            else:
                for i in range(len(EMA)):
                    EMA[i] = alpha * nn_out[i] + (1 - alpha) * EMA[i]

            print(EMA)
            predicted_action = ACTIONS[np.argmax(EMA)]

            # Only move if both raw confidence and EMA confidence are above threshold
            if raw_confidence > 0.80 and EMA[int(np.argmax(EMA))] > 0.80:
                if predicted_action == "left":
                    gui.square['x1'] -= gui.MOVE_SPEED
                    gui.square['x2'] -= gui.MOVE_SPEED

                elif predicted_action == "right":
                    gui.square['x1'] += gui.MOVE_SPEED
                    gui.square['x2'] += gui.MOVE_SPEED

                print(predicted_action)

            count_down -= 1

            env[:, gui.HEIGHT // 2 - 5:gui.HEIGHT // 2 + 5, :] = gui.horizontal_line
            env[gui.WIDTH // 2 - 5:gui.WIDTH // 2 + 5, :, :] = gui.vertical_line
            env[gui.square['y1']:gui.square['y2'], gui.square['x1']:gui.square['x2']] = gui.box

            cv2.imshow('', env)

            if first_run:
                first_run = False
                start = timer()
            else:
                end = timer()
                print("\rFPS: ", 1 // (end - start), " ", end='')
                start = timer()

            shared_vars.key = cv2.waitKey(1) & 0xFF
            if shared_vars.key == ord("q"):
                cv2.destroyAllWindows()
                break

        # time.sleep(0.1)


#############################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial-port', type=str, help='serial port',
                        required=False, default='COM4')

    args = parser.parse_args()
    params = BrainFlowInputParams()
    params.serial_port = args.serial_port

    board = BoardShim(BoardIds.CYTON_BOARD.value, params)
    board.prepare_session()

    shared_vars = Shared()
    mutex = threading.Lock()

    board.start_stream()  # use this for default options

    acquisition = threading.Thread(target=acquire_signals)
    acquisition.start()
    computing = threading.Thread(target=compute_signals)
    computing.start()

    acquisition.join()
    computing.join()
    board.stop_stream()
    board.stop_stream()