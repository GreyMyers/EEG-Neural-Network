from brainflow import DataFilter, FilterTypes, AggOperations
from scipy.signal import butter, lfilter
from matplotlib import pyplot as plt
from scipy.fft import fft
from scipy.signal import iirnotch, filtfilt
import numpy as np
import os
import shutil
import pkg_resources

ACTIONS = ["left", "right", "forward", "stop"]

def split_data(starting_dir="personal_dataset", splitting_percentage=(70, 20, 10), shuffle=True,
               coupling=False, division_factor=0):
    """
    Split EEG dataset into training, validation, and test folders.
    Overwrites previous splits every time.
    """

    training_per, validation_per, test_per = splitting_percentage

    # --- Always start fresh: remove old split folders ---
    for folder in ["training_data", "validation_data", "test_data"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)

    # Create new base split folders
    os.makedirs("training_data", exist_ok=True)
    os.makedirs("validation_data", exist_ok=True)
    os.makedirs("test_data", exist_ok=True)

    print(f"SPLIT_DATA(): starting_dir={starting_dir}")
    print(f"ACTIONS={ACTIONS}")

    for action in ACTIONS:
        action_data = []

        data_dir = os.path.join(starting_dir, action)
        if not os.path.exists(data_dir):
            print(f"Warning: missing folder {data_dir}, skipping.")
            continue

        # Load all trials for this action
        all_action_data = []
        for file in sorted(os.listdir(data_dir)):
            if file.endswith(".npy"):
                arr = np.load(os.path.join(data_dir, file))
                all_action_data.append(arr)

        if not all_action_data:
            print(f"No data files found for {action} in {data_dir}.")
            continue

        # Handle coupling (deprecated for EEG)
        if coupling:
            coupled_actions = []
            first = True
            for i in range(len(all_action_data)):
                if division_factor != 0 and i % division_factor != 0:
                    continue
                if first:
                    tmp_act = all_action_data[i]
                    first = False
                else:
                    coupled_actions.append([tmp_act, all_action_data[i]])
                    first = True
            if shuffle:
                np.random.shuffle(coupled_actions)
            for pair in coupled_actions:
                action_data.extend(pair)
        else:
            # Normal (non-coupled) branch
            if division_factor != 0:
                action_data = [all_action_data[i] for i in range(0, len(all_action_data), division_factor)]
            else:
                action_data = all_action_data.copy()
            if shuffle:
                np.random.shuffle(action_data)

        # --- Split into train/val/test ---
        num_total = len(action_data)
        num_train = int(num_total * training_per / 100)
        num_val = int(num_total * validation_per / 100)
        num_test = num_total - num_train - num_val

        print(f"Splitting '{action}' ({num_total} samples) -> "
              f"{num_train} train, {num_val} val, {num_test} test")

        splits = [
            ("training_data", range(0, num_train)),
            ("validation_data", range(num_train, num_train + num_val)),
            ("test_data", range(num_train + num_val, num_total))
        ]

        for split_name, idx_range in splits:
            dst_dir = os.path.join(split_name, action)
            os.makedirs(dst_dir, exist_ok=True)
            for idx in idx_range:
                np.save(os.path.join(dst_dir, f"{idx}.npy"), action_data[idx])

def load_data(starting_dir, shuffle=True, balance=False):
    """
        This function loads the personal_dataset from a directory where the classes
        have been split into different folders where each file is a sample

    :param starting_dir: the path of the personal_dataset you want to load
    :param shuffle: bool, decides if the personal_dataset will be shuffled
    :param balance: bool, decides if samples should be equal in cardinality between classes
    :return: X, y: both python lists
    """

    data = [[] for i in range(len(ACTIONS))]
    for i, action in enumerate(ACTIONS):

        data_dir = os.path.join(starting_dir, action)
        for file in sorted(os.listdir(data_dir)):
            data[i].append(np.load(os.path.join(data_dir, file)))

    if balance:
        lengths = [len(data[i]) for i in range(len(ACTIONS))]
        print(lengths)

        # this is required if one class has more samples than the others
        for i in range(len(ACTIONS)):
            data[i] = data[i][:min(lengths)]

        lengths = [len(data[i]) for i in range(len(ACTIONS))]
        print(lengths)

    # this is needed to shuffle the personal_dataset between classes, so the model
    # won't train first on one single class and then pass to the next one
    # but it trains on all classes "simultaneously"
    combined_data = []

    # we are using one hot encodings
    for i in range(len(ACTIONS)):
        lbl = np.zeros(len(ACTIONS), dtype=int)
        lbl[i] = 1
        for sample in data[i]:
            combined_data.append([sample, lbl])

    if shuffle:
        np.random.shuffle(combined_data)

    # create X, y:
    X = []
    y = []
    for sample, label in combined_data:
        X.append(sample)
        y.append(label)

    return np.array(X), np.array(y)

def standardize(data, std_type="channel_wise"):
    """
    Normalize EEG data safely to zero mean and unit variance.
    Prevents NaN propagation if std == 0.
    """

    if std_type == "feature_wise":
        for j in range(len(data[0, 0, :])):
            mean = data[:, :, j].mean()
            std = data[:, :, j].std()
            std = std if std != 0 else 1e-6  # prevent divide by zero
            data[:, :, j] = (data[:, :, j] - mean) / std

    elif std_type == "sample_wise":
        for k in range(len(data)):
            mean = np.mean(data[k])
            std = np.std(data[k])
            std = std if std != 0 else 1e-6
            data[k] = (data[k] - mean) / std

    elif std_type == "channel_wise":
        for k in range(len(data)):
            sample = np.array(data[k])
            for i in range(sample.shape[0]):  # each channel
                mean = np.mean(sample[i])
                std = np.std(sample[i])
                if np.isnan(mean) or np.isnan(std):
                    sample[i] = np.zeros_like(sample[i])
                else:
                    # clamp near-zero variance channels to zero to avoid blow-ups
                    if std < 1e-3:
                        sample[i] = np.zeros_like(sample[i])
                    else:
                        sample[i] = (sample[i] - mean) / std

            data[k] = sample

    return np.nan_to_num(data)

def visualize_data(data, file_name, title, length):
    # takes a look at the personal_dataset
    for i in range(len(data[0])):
        plt.plot(np.arange(len(data[0][i])), data[0][i].reshape(length))

    plt.title(title)
    plt.savefig(file_name + ".png")
    plt.clf()

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def notch_filter_scipy(signal, fs=250, freq=60.0, Q=30.0):
    """Fallback notch filter using SciPy if BrainFlow fails."""
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, signal)

def preprocess_raw_eeg(data, fs=250, lowcut=2.0, highcut=65.0,
                       MAX_FREQ=60, power_hz=60, coi3order=3):
    """
    Processes raw EEG data for model input:
      • Standardize per channel
      • 60-Hz notch filter (SciPy implementation)
      • Broad 2-120 Hz bandpass, optional wavelet denoising
      • Tight 2-65 Hz bandpass
      • Compute FFTs (first MAX_FREQ bins)

    :param data: ndarray, shape = (samples, channels, values)
    :param fs: sampling rate (Hz)
    :param lowcut: low cutoff for final bandpass
    :param highcut: high cutoff for final bandpass
    :param MAX_FREQ: number of FFT bins to keep
    :param power_hz: power-line frequency (60 Hz US / 50 Hz EU)
    :param coi3order: wavelet denoising order (0 = disabled)
    :return: tuple (filtered_data, fft_data)
    """

    data = standardize(data)  # normalize each channel
    n_samples, n_chans, n_points = data.shape
    fft_data = np.zeros((n_samples, n_chans, MAX_FREQ))

    for sample in range(n_samples):
        for channel in range(n_chans):
            # Ensure numpy float64 array
            signal = np.asarray(data[sample][channel], dtype=np.float64)

            # --- Stable 60-Hz notch (SciPy only) ---
            try:
                signal = notch_filter_scipy(signal, fs=fs, freq=power_hz)
            except Exception as e:
                print(f"[WARN] SciPy notch filter failed on sample {sample}, ch {channel}: {e}")
                continue

            # --- Broad band-pass 2–120 Hz ---
            try:
                signal = butter_bandpass_filter(signal, 2, 120, fs, order=5)
            except Exception as e:
                print(f"[WARN] Broad bandpass failed on sample {sample}, ch {channel}: {e}")
                continue

            # --- Optional wavelet denoising ---
            if coi3order != 0:
                try:
                    DataFilter.perform_wavelet_denoising(signal, 'coif3', coi3order)
                except Exception as e:
                    print(f"[WARN] Wavelet denoising failed: {e}")

            # --- Final tight band-pass 2–65 Hz ---
            try:
                signal = butter_bandpass_filter(signal, lowcut, highcut, fs, order=5)
            except Exception as e:
                print(f"[WARN] Final bandpass failed on sample {sample}, ch {channel}: {e}")
                continue

            # Save filtered signal
            data[sample][channel] = signal

            # --- FFT ---
            fft_data[sample][channel] = np.abs(fft(signal)[:MAX_FREQ])

    return data, fft_data
