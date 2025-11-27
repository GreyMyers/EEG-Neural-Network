# This is intended for OpenBCI Cyton Board,
# check: https://brainflow.readthedocs.io for other boards


from functions import ACTIONS, preprocess_raw_eeg
from motor_control import MotorController

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


def detect_jaw_clench(sample, channel_idx=7, threshold=100, window_variance_threshold=500):
    """
    Detect jaw clench from electrode 8 (index 7)
    
    Args:
        sample: EEG data array (8 channels x 250 samples)
        channel_idx: Channel index for jaw electrode (default 7 for electrode 8)
        threshold: Amplitude threshold for detection
        window_variance_threshold: Variance threshold for muscle activity
    
    Returns:
        bool: True if jaw clench detected
    """
    if sample is None or len(sample.shape) < 2:
        return False
    
    jaw_channel = sample[channel_idx]
    
    # Check for high amplitude (EMG artifact from jaw clench)
    max_amplitude = np.max(np.abs(jaw_channel))
    
    # Check for high variance (muscle activity creates noise)
    variance = np.var(jaw_channel)
    
    # Jaw clench produces high amplitude and high variance
    is_clenched = (max_amplitude > threshold) or (variance > window_variance_threshold)
    
    return is_clenched


def compute_signals(use_motor_control=True):
    global JAW_THRESHOLD_AMP, JAW_THRESHOLD_VAR
    
    MODEL_NAME = "models/Grey/85.0-76epoch-1763074954-loss-0.43.keras"
    model = keras.models.load_model(MODEL_NAME)
    count_down = 100  # restarts the GUI when reaches 0
    EMA = [-1, -1]  # exponential moving average over the probabilities of the model (2 classes: left/right)
    alpha = 0.4  # c ,oefficient for the EMA
    gui = GraphicalInterface()
    first_run = True
    
    # Jaw clench detection parameters (from command line or defaults)
    try:
        JAW_AMPLITUDE_THRESHOLD = JAW_THRESHOLD_AMP
        JAW_VARIANCE_THRESHOLD = JAW_THRESHOLD_VAR
    except NameError:
        JAW_AMPLITUDE_THRESHOLD = 100  # μV - default
        JAW_VARIANCE_THRESHOLD = 500   # default
    
    # Initialize motor controller if enabled
    motor = None
    if use_motor_control:
        try:
            print("\n" + "="*50)
            print("Initializing Motor Controller...")
            print("="*50)
            motor = MotorController(auto_connect=True)
            if not motor.connected:
                print("\nMotor controller failed to connect. Running in GUI-only mode.")
                print("To use motor control, ensure ESP32 is paired via Bluetooth.")
                motor = None
            else:
                print("Motor controller ready!")
                print("\nControls:")
                print("  - JAW CLENCH (electrode 8) → Forward")
                print("  - LEFT motor imagery → Turn Left")
                print("  - RIGHT motor imagery → Turn Right")
                print(f"\nJaw Clench Thresholds:")
                print(f"  - Amplitude: {JAW_AMPLITUDE_THRESHOLD}")
                print(f"  - Variance: {JAW_VARIANCE_THRESHOLD}")
                print("="*50)
        except Exception as e:
            print(f"Error initializing motor controller: {e}")
            print("Running in GUI-only mode.")
            motor = None

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

            # computing exponential moving average
            if EMA[0] == -1:  # if this is the first iteration (base case)
                for i in range(len(EMA)):
                    EMA[i] = nn_out[i]
            else:
                for i in range(len(EMA)):
                    EMA[i] = alpha * nn_out[i] + (1 - alpha) * EMA[i]

            print(EMA)
            predicted_action = ACTIONS[np.argmax(EMA)]
            
            # Check for jaw clench first (priority control)
            jaw_clenched = detect_jaw_clench(shared_vars.sample, 
                                            channel_idx=7,  # Electrode 8 (0-indexed)
                                            threshold=JAW_AMPLITUDE_THRESHOLD,
                                            window_variance_threshold=JAW_VARIANCE_THRESHOLD)
            
            if jaw_clenched:
                # JAW CLENCH DETECTED → MOVE FORWARD
                gui.square['y1'] -= gui.MOVE_SPEED  # Move up on screen
                gui.square['y2'] -= gui.MOVE_SPEED
                
                if motor:
                    motor.execute_action("forward", throttled=True)
                
                print("JAW CLENCH → FORWARD")
            
            elif EMA[int(np.argmax(EMA))] > 0.67:
                # CONFIDENT LEFT/RIGHT PREDICTION
                if predicted_action == "left":
                    gui.square['x1'] -= gui.MOVE_SPEED
                    gui.square['x2'] -= gui.MOVE_SPEED
                    
                    # Send motor command
                    if motor:
                        motor.execute_action("left", throttled=True)

                elif predicted_action == "right":
                    gui.square['x1'] += gui.MOVE_SPEED
                    gui.square['x2'] += gui.MOVE_SPEED
                    
                    # Send motor command
                    if motor:
                        motor.execute_action("right", throttled=True)

                print(predicted_action)
            else:
                # Stop motors if no confident signal
                if motor:
                    motor.execute_action("stop", throttled=True)

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
                # Disconnect motor controller on exit
                if motor:
                    motor.disconnect()
                break

        # time.sleep(0.1)


#############################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BCI Motor Control System')
    parser.add_argument('--serial-port', type=str, help='serial port for OpenBCI',
                        required=False, default='COM3')
    parser.add_argument('--motor-control', action='store_true', 
                        help='enable motor control via Bluetooth (ESP32)')
    parser.add_argument('--no-motor-control', action='store_true',
                        help='disable motor control (GUI only)')
    parser.add_argument('--jaw-threshold', type=float, default=100.0,
                        help='amplitude threshold for jaw clench detection (default: 100)')
    parser.add_argument('--jaw-variance', type=float, default=500.0,
                        help='variance threshold for jaw clench detection (default: 500)')

    args = parser.parse_args()
    
    # Store jaw clench thresholds as global variables for compute_signals
    global JAW_THRESHOLD_AMP, JAW_THRESHOLD_VAR
    JAW_THRESHOLD_AMP = args.jaw_threshold
    JAW_THRESHOLD_VAR = args.jaw_variance
    
    # Determine motor control setting
    use_motor = not args.no_motor_control  # Default to True unless explicitly disabled
    if args.motor_control:
        use_motor = True
    
    params = BrainFlowInputParams()
    params.serial_port = args.serial_port

    board = BoardShim(BoardIds.CYTON_BOARD.value, params)
    board.prepare_session()

    shared_vars = Shared()
    mutex = threading.Lock()

    board.start_stream()  # use this for default options

    acquisition = threading.Thread(target=acquire_signals)
    acquisition.start()
    computing = threading.Thread(target=compute_signals, args=(use_motor,))
    computing.start()

    acquisition.join()
    computing.join()
    board.stop_stream()
    board.stop_stream()