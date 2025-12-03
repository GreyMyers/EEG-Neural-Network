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

def detect_jaw_clench(sample, channel_idx=7, threshold=100, window_variance_threshold=500):
    """
    Detect jaw clench from electrode 8 (channel index 7)
    Jaw clench produces high amplitude EMG artifact
    
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


def compute_signals(motor_controller=None, jaw_amp_threshold=100, jaw_var_threshold=500):
    MODEL_NAME = "models/grey/77.5-149epoch-1764644886-loss-0.54.keras"
    model = keras.models.load_model(MODEL_NAME)
    EMA = [-1, -1]  # exponential moving average over the probabilities of the model (2 classes: left/right)
    alpha = 0.4  # coefficient for the EMA
    last_command = None  # Track last sent command to avoid duplicates

    while True:
        with mutex:
            # Check for jaw clench FIRST (priority control - channel 8)
            jaw_clenched = detect_jaw_clench(
                shared_vars.sample,
                channel_idx=7,  # Electrode 8 (0-indexed)
                threshold=jaw_amp_threshold,
                window_variance_threshold=jaw_var_threshold
            )
            
            if jaw_clenched:
                # JAW CLENCH DETECTED → FORWARD (W command)
                current_command = "forward"
                if motor_controller and motor_controller.connected:
                    if current_command != last_command:
                        motor_controller.execute_action(current_command, throttled=False)
                        print(f"JAW CLENCH → Motor: FORWARD (W)")
                        last_command = current_command
                else:
                    print("JAW CLENCH → FORWARD")
                time.sleep(0.1)
                continue  # Skip neural network prediction when jaw is clenched
            
            # prediction on the task (left/right motor imagery)
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
    parser.add_argument('--jaw-threshold', type=float, default=100.0,
                        help='Amplitude threshold for jaw clench detection (default: 100)')
    parser.add_argument('--jaw-variance', type=float, default=500.0,
                        help='Variance threshold for jaw clench detection (default: 500)')

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
        print(f"\nControls:")
        print(f"  - JAW CLENCH (channel 8) → FORWARD (W)")
        print(f"  - LEFT motor imagery    → LEFT (A)")
        print(f"  - RIGHT motor imagery   → RIGHT (D)")
        print(f"\nJaw Clench Thresholds:")
        print(f"  - Amplitude: {args.jaw_threshold}")
        print(f"  - Variance: {args.jaw_variance}")

    board.start_stream()  # use this for default options

    try:
        acquisition = threading.Thread(target=acquire_signals)
        acquisition.start()
        computing = threading.Thread(target=compute_signals, 
                                     args=(motor_controller, args.jaw_threshold, args.jaw_variance))
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
