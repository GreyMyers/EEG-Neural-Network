from brainflow import BoardShim, BrainFlowInputParams, BoardIds
import numpy as np
import time
import os

# Settings
TRIAL_DURATION = 4      # seconds of data per trial
SAMPLES_PER_SEC = 250   # Cyton default
NUM_TRIALS = 20         # number of trials per class
SAVE_DIR = "motor_imagery_dataset"
CHANNELS = [1, 2]       # adjust for your Cyton inputs (C3, C4)

def save_trial(data, label):
    """Save one trial of data with its label"""
    
    label_dir = os.path.join(SAVE_DIR, label)
    os.makedirs(label_dir, exist_ok=True)
    filename = os.path.join(label_dir, f"{int(time.time())}.npy")
    np.save(filename, data)
    print(f"Saved {data.shape} samples to {filename}")

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

                data = board.get_board_data()  # all new samples
                trial_data = np.array([data[ch] for ch in CHANNELS])
                save_trial(trial_data, label)

    except KeyboardInterrupt:
        print("Experiment interrupted.")

    board.stop_stream()
    board.release_session()
    print("Session finished.")
