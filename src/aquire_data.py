from brainflow import BoardShim, BrainFlowInputParams, BoardIds
import numpy as np
import time
import os

# Settings
TRIAL_DURATION = 4        # seconds per trial
SAMPLES_PER_SEC = 250     # Cyton default
NUM_TRIALS = 20           # trials per class

# Folder structure that matches train.py
DATASET_ROOT = r"C:\Users\greym\Xavier\datasets"
MOTOR_DIR = os.path.join(DATASET_ROOT, "motor")
FRONTAL_DIR = os.path.join(DATASET_ROOT, "frontal")

def save_trial(data, region, label):
    """Save one trial of EEG data in the proper region folder."""
    if region == "motor":
        label_dir = os.path.join(MOTOR_DIR, label)
    elif region == "frontal":
        label_dir = os.path.join(FRONTAL_DIR, label)
    else:
        raise ValueError("region must be 'motor' or 'frontal'")

    os.makedirs(label_dir, exist_ok=True)
    filename = os.path.join(label_dir, f"{int(time.time()*1000)}.npy")
    np.save(filename, data)
    print(f"Saved {data.shape} samples to {filename}")

def preprocess_trial(raw_data, region):
    """Select only relevant channels for region and resize/pad to (2, 250)."""
    if region == "motor":
        CHANNELS = [1, 2]  # C3, C4
    elif region == "frontal":
        CHANNELS = [3, 4]  # Fp1, Fp2
    else:
        raise ValueError("region must be 'motor' or 'frontal'")

    trial_data = np.array([raw_data[ch] for ch in CHANNELS])
    desired_len = 250  # samples per trial

    # Trim or pad
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

    print("Get ready for EEG data collection...")
    time.sleep(2)

    # --- Motor cortex (C3, C4) ---
    region = "motor"
    motor_labels = ["left", "right"]
    for label in motor_labels:
        print(f"\n--- Starting {NUM_TRIALS} {label.upper()} trials ({region}) ---")
        for t in range(NUM_TRIALS):
            input(f"Press Enter to start trial {t+1}/{NUM_TRIALS} ({label})")
            print(f"Imagine moving your {label} hand for {TRIAL_DURATION} seconds...")
            time.sleep(1)
            board.get_board_data()  # clear old buffer
            time.sleep(TRIAL_DURATION)
            raw_data = board.get_board_data()
            trial_data = preprocess_trial(raw_data, region)
            save_trial(trial_data, region, label)

    # --- Frontal lobe (Fp1, Fp2) ---
    region = "frontal"
    frontal_labels = ["forward", "stop"]
    for label in frontal_labels:
        print(f"\n--- Starting {NUM_TRIALS} {label.upper()} trials ({region}) ---")
        for t in range(NUM_TRIALS):
            input(f"Press Enter to start trial {t+1}/{NUM_TRIALS} ({label})")
            if label == "forward":
                print("Concentrate or focus for forward command...")
            else:
                print("Relax or clear your mind for stop command...")

            time.sleep(1)
            board.get_board_data()
            time.sleep(TRIAL_DURATION)
            raw_data = board.get_board_data()
            trial_data = preprocess_trial(raw_data, region)
            save_trial(trial_data, region, label)

    board.stop_stream()
    board.release_session()
    print("\nSession finished")
