"""
visualize_filters.py
Shows raw EEG vs. postprocessed EEG after filtering + denoising
"""

from brainflow import BoardShim, BrainFlowInputParams, BoardIds
from functions import preprocess_raw_eeg
import matplotlib.pyplot as plt
import numpy as np
import time


# ------------------------
# USER SETTINGS
# ------------------------
BOARD_ID = BoardIds.CYTON_BOARD.value  # or SYNTHETIC_BOARD for testing
PORT = "COM3"                          # change to your Cyton port
DURATION = 5                           # seconds of data to capture
FS = 250                               # Cyton sample rate
CHANNELS = [1, 2, 3, 4]                # Example: C3, C4, Fp1, Fp2
LOWCUT, HIGHCUT = 7, 45                # same bandpass as your model
COI3ORDER = 3                          # wavelet denoising level


# ------------------------
# MAIN SCRIPT
# ------------------------
def main():
    params = BrainFlowInputParams()
    params.serial_port = PORT
    board = BoardShim(BOARD_ID, params)

    print("Preparing board session...")
    board.prepare_session()
    board.start_stream()
    time.sleep(DURATION)
    data = board.get_board_data()  # pull entire buffer
    board.stop_stream()
    board.release_session()
    print(f"Collected {data.shape[1]} samples total")

    eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
    raw_data = np.array([data[ch] for ch in eeg_channels])

    # Trim to desired channels and time window
    raw_data = raw_data[:len(CHANNELS), -FS * DURATION:]
    samples = raw_data.shape[1]
    t = np.linspace(0, samples / FS, samples)

    # ----------------------------
    # Apply your preprocessing chain
    # ----------------------------
    pre_data, fft_data = preprocess_raw_eeg(
        raw_data.reshape((1, len(CHANNELS), samples)),
        fs=FS,
        lowcut=LOWCUT,
        highcut=HIGHCUT,
        coi3order=COI3ORDER
    )

    # Extract processed signal
    processed = pre_data[0]  # shape (channels, samples)

    # ----------------------------
    # Visualization
    # ----------------------------
    fig, axs = plt.subplots(len(CHANNELS), 1, figsize=(10, 7), sharex=True)
    fig.suptitle("EEG Filtering Visualization", fontsize=14)

    for i, ch in enumerate(CHANNELS):
        axs[i].plot(t, raw_data[i], color='gray', alpha=0.7, label='Raw EEG')
        axs[i].plot(t, processed[i], color='blue', linewidth=1.2, label='Filtered EEG')
        axs[i].set_ylabel(f"Ch {ch}")
        axs[i].grid(True)
        if i == 0:
            axs[i].legend()

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # Optional FFT view
    # ----------------------------
    plt.figure(figsize=(8, 4))
    plt.title("FFT Magnitude Spectrum (First Channel)")
    freqs = np.arange(fft_data.shape[2])
    plt.plot(freqs, fft_data[0, 0], color='purple')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
