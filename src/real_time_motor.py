# This is intended for OpenBCI Cyton Board,
# check: https://brainflow.readthedocs.io for other boards
# Motor control version - sends commands to ESP32 based on EEG predictions

from functions import ACTIONS, preprocess_raw_eeg

from brainflow import BoardShim, BrainFlowInputParams, BoardIds
from tensorflow import keras
from motor_control import MotorController
import numpy as np
import threading
import argparse
import time


class Shared:
    def __init__(self):
        self.sample = None


#############################################################

def acquire_signals():
    count = 0
    while True:
        with mutex:
            if count == 0:
                time.sleep(2)
                count += 1
            time.sleep(0.5)
            # get_current_board_data does not remove data from board internal buffer
            # thus allowing us to acquire overlapped data and compute more classification over 1 sec
            data = board.get_current_board_data(250)

            sample = []
            eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
            for channel in eeg_channels:
                sample.append(data[channel])

            shared_vars.sample = np.array(sample)
        time.sleep(0.1)


def compute_signals(motor_controller=None):
    MODEL_NAME = "models/grey/77.5-149epoch-1764644886-loss-0.54.keras"
    model = keras.models.load_model(MODEL_NAME)
    EMA = [-1, -1]  # exponential moving average over the probabilities of the model (2 classes: left/right)
    alpha = 0.4  # coefficient for the EMA
    last_command = None  # Track last sent command to avoid duplicates

    while True:
        with mutex:
            # prediction on the task
            nn_input, ffts = preprocess_raw_eeg(shared_vars.sample.reshape((1, 8, 250)),
                                                fs=250, lowcut=8, highcut=30, coi3order=0)
            nn_input = nn_input.reshape((1, 8, 250, 1))  # 4D Tensor
            nn_out = model.predict(nn_input, verbose=0)[0]  # this is a probability array

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

            # Determine command based on confidence
            current_command = None
            if raw_confidence > 0.80 and EMA[int(np.argmax(EMA))] > 0.80:
                if predicted_action == "left":
                    current_command = "left"
                elif predicted_action == "right":
                    current_command = "right"
            else:
                # Stop motors when confidence is low
                current_command = "stop"

            # Only send command if it's different from the last one
            if motor_controller and motor_controller.connected:
                if current_command != last_command:
                    motor_controller.execute_action(current_command, throttled=False)
                    print(f"Motor: {current_command.upper()}")
                    last_command = current_command
            else:
                print(predicted_action)

        time.sleep(0.1)


#############################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EEG-based motor control using OpenBCI')
    parser.add_argument('--serial-port', type=str, help='EEG board serial port',
                        required=False, default='COM4')
    parser.add_argument('--motor-port', type=str, help='Motor controller COM port (auto-detect if not specified)',
                        required=False, default='COM5')

    args = parser.parse_args()
    params = BrainFlowInputParams()
    params.serial_port = args.serial_port

    board = BoardShim(BoardIds.CYTON_BOARD.value, params)
    board.prepare_session()

    shared_vars = Shared()
    mutex = threading.Lock()

    # Initialize motor controller
    print("Initializing motor controller...")
    motor_controller = MotorController(auto_connect=False)
    if args.motor_port:
        motor_controller.connect(port=args.motor_port)
    else:
        motor_controller.connect()
    
    if not motor_controller.connected:
        print("ERROR: Motor controller not connected. Exiting.")
        board.release_session()
        exit(1)
    else:
        print("Motor controller connected successfully!")

    board.start_stream()  # use this for default options

    try:
        acquisition = threading.Thread(target=acquire_signals)
        acquisition.start()
        computing = threading.Thread(target=compute_signals, args=(motor_controller,))
        computing.start()

        acquisition.join()
        computing.join()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Cleanup
        board.stop_stream()
        if motor_controller and motor_controller.connected:
            motor_controller.stop()  # Stop motors before disconnecting
            motor_controller.disconnect()
            print("Motor controller disconnected.")
        board.release_session()
        print("Shutdown complete.")
