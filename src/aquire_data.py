from brainflow import BoardShim, BrainFlowInputParams, BoardIds
import numpy as np
import time, os

TRIAL_DURATION = 1
SAMPLES_PER_SEC = 250
NUM_TRIALS = 100
# For Windows: DATASET_ROOT = r"c:/Users/Gm08348/EEG-Neural-Network/datasets"
# For Linux: DATASET_ROOT = r"/home/grey/EEG-Neural-Network/datasets"
DATASET_ROOT = r"c:/Users/Gm08348/EEG-Neural-Network/datasets"

def save_trial(data, label, region_dir):
    os.makedirs(os.path.join(region_dir, label), exist_ok=True)
    filename = os.path.join(region_dir, label, f"{int(time.time()*1000)}.npy")
    np.save(filename, data)
    print(f"Saved {data.shape} samples to {filename}")

if __name__ == "__main__":
    params = BrainFlowInputParams()
    
    # On linux, the serial port is typically /dev/ttyUSB0
    # On windows, the serial port is typically COM3

    params.serial_port = "COM4"  # Adjust as needed
    board = BoardShim(BoardIds.CYTON_BOARD.value, params)
    board.prepare_session()
    board.start_stream()

    try:
        name = input("Enter patient name: ")
        root_dir = os.path.join(DATASET_ROOT, name)
        motor_dir = os.path.join(root_dir, "motor")

        # Collects all channels
        # eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
        
        # Selected all 8 channels
        selected_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
        
        motor_labels = ["left", "right"]
        print("Get ready for EEG data collection...")

        for label in motor_labels:
            print(f"\n--- Starting {NUM_TRIALS} {label.upper()} trials ---")
            for t in range(NUM_TRIALS):
                # input(f"Press Enter to start trial {t+1}/{NUM_TRIALS} ({label})")
                print(f"Imagine moving your {label} hand for {TRIAL_DURATION} seconds...")
                board.get_board_data()  # flush
                time.sleep(TRIAL_DURATION)
                eeg_data = board.get_current_board_data(250)[selected_channels, :]

                desired = SAMPLES_PER_SEC
                current = eeg_data.shape[1]

                if current > desired:
                    eeg_data = eeg_data[:, -desired:]  # crop extra
                elif current < desired:
                    pad = np.tile(eeg_data[:, -1:], (1, desired - current))
                    eeg_data = np.concatenate((eeg_data, pad), axis=1)
                
                save_trial(eeg_data, label, motor_dir)

    finally:
        board.stop_stream()
        board.release_session()
        print("\nSession finished safely.")