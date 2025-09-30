from brainflow import BoardShim, BrainFlowInputParams, BoardIds
import numpy as np
import time
import os

# Settings
TRIAL_DURATION = 4       # seconds of data per trial
SAMPLES_PER_SEC = 250    # Cyton default
NUM_TRIALS = 20          # number of trials per class
SAVE_DIR = "motor_imagery_dataset"
CHANNELS = [1, 2]        # adjust for your Cyton inputs (e.g. C3, C4)

# Target shape for BrainPad compatibility
TARGET_SHAPE = (len(CHANNELS), 250)   # (2, 250)

def save_trial(data, label):
    """Save one trial of data with its label"""
    label_dir = os.path.join(SAVE_DIR, label)
    os.makedirs(label_dir, exist_ok=True)
    filename = os.path.join(label_dir, f"{int(time.time())}.npy")
    np.save(filename, data)
    print(f"Saved {data.shape} samples to {filename}")

def preprocess_trial(raw_data):
    """
    Select only chosen channels and force trial to TARGET_SHAPE.
    raw_data: full board.get_board_data() output
    """
    trial_data = np.array([raw_data[ch] for ch in CHANNELS])

    desired_len = TARGET_SHAPE[1]

    # Trim or pad to match target length
    if trial_data.shape[1] > desired_len:
        trial_data = trial_data[:, :desired_len]
    elif trial_data.shape[1] < desired_len:
        pad_width = desired_len - trial_data.shape[1]
        trial_data = np.pad(trial_data, ((0, 0), (0, pad_width)), mode="constant")

    return trial_data

if __name__ == "__main__":
    params = BrainFlowInputParams()
    params.serial_port = "COM3"  # adjust for your system

    board = BoardShim(BoardIds.CYTON_BOARD.value, params)
    board.prepare_session()
    board.start_stream()

    print("Get ready for motor imagery experiment...")
    time.sleep(2)

    labels = ["left", "right"]
    try:
        for label in labels:
            print(f"--- Starting {NUM_TRIALS} {label.upper()} trials ---")
            for t in range(NUM_TRIALS):
                input(f"Press Enter to start trial {t+1}/{NUM_TRIALS} ({label})")

                print(f"Imagine moving your {label} hand for {TRIAL_DURATION} seconds...")
                time.sleep(1)  # short delay before start

                board.get_board_data()  # clear old buffer
                time.sleep(TRIAL_DURATION)

                raw_data = board.get_board_data()
                trial_data = preprocess_trial(raw_data)

                save_trial(trial_data, label)

    except KeyboardInterrupt:
        print("Experiment interrupted.")

    board.stop_stream()
    board.release_session()
    print("Session finished.")
