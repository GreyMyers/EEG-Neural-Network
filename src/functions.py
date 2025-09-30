from brainflow import DataFilter, FilterTypes, AggOperations
from scipy.signal import butter, lfilter
from matplotlib import pyplot as plt
from scipy.fft import fft
from scipy.signal import iirnotch, filtfilt
import numpy as np
import os
import pkg_resources

ACTIONS = ["left" , "right"]

def split_data(starting_dir="personal_dataset", splitting_percentage=(70, 20, 10), shuffle=True, coupling=False,
               division_factor=0):
    """
        This function splits the dataset in three folders, training, validation, test
        Has to be run just everytime the dataset is changed

    :param starting_dir: string, the directory of the dataset
    :param splitting_percentage:  tuple, (training_percentage, validation_percentage, test_percentage)
    :param shuffle: bool, decides if the personal_dataset will be shuffled
    :param coupling: bool, decides if samples are shuffled singularly or by couples
    :param division_factor: int, if the personal_dataset used is made of FFTs which are taken from multiple sittings
                            one sample might be very similar to an adjacent one, so not all the samples
                            should be considered because some very similar samples could fall both in
                            validation and training, thus the division_factor divides the personal_dataset.
                            if division_factor == 0 the function will maintain all the personal_dataset

    """
    training_per, validation_per, test_per = splitting_percentage

    if not os.path.exists("training_data") and not os.path.exists("validation_data") \
            and not os.path.exists("test_data"):

        # creating directories

        os.mkdir("training_data")
        os.mkdir("validation_data")
        os.mkdir("test_data")

        for action in ACTIONS:

            action_data = []
            all_action_data = []
            # this will contain all the samples relative to the action

            data_dir = os.path.join(starting_dir, action)
            # sorted will make sure that the personal_dataset is appended in the order of acquisition
            # since each sample file is saved as "timestamp".npy
            for file in sorted(os.listdir(data_dir)):
                # each item is a ndarray of shape (8, 90) that represents ≈1sec of acquisition
                all_action_data.append(np.load(os.path.join(data_dir, file)))

            # TODO: make this coupling part readable
            # coupling was used when overlapping FFTs were used
            # is now deprecated with EEG models and very time-distant acquisitions
            if coupling:
                # coupling near time acquired samples to reduce the probability of having
                # similar samples in both train and validation sets
                coupled_actions = []
                first = True
                for i in range(len(all_action_data)):
                    if division_factor != 0:
                        if i % division_factor == 0:
                            if first:
                                tmp_act = all_action_data[i]
                                first = False
                            else:
                                coupled_actions.append([tmp_act, all_action_data[i]])
                                first = True
                    else:
                        if first:
                            tmp_act = all_action_data[i]
                            first = False
                        else:
                            coupled_actions.append([tmp_act, all_action_data[i]])
                            first = True

                if shuffle:
                    np.random.shuffle(coupled_actions)

                # reformatting all the samples in a single list
                for i in range(len(coupled_actions)):
                    for j in range(len(coupled_actions[i])):
                        action_data.append(coupled_actions[i][j])

            else:
                for i in range(len(all_action_data)):
                    if division_factor != 0:
                        if i % division_factor == 0:
                            action_data.append(all_action_data[i])
                    else:
                        action_data = all_action_data

                if shuffle:
                    np.random.shuffle(action_data)

            num_training_samples = int(len(action_data) * training_per / 100)
            num_validation_samples = int(len(action_data) * validation_per / 100)
            num_test_samples = int(len(action_data) * test_per / 100)

            # creating subdirectories for each action
            tmp_dir = os.path.join("training_data", action)
            if not os.path.exists(tmp_dir):
                os.mkdir(tmp_dir)
            for sample in range(num_training_samples):
                np.save(file=os.path.join(tmp_dir, str(sample)), arr=action_data[sample])

            tmp_dir = os.path.join("validation_data", action)
            if not os.path.exists(tmp_dir):
                os.mkdir(tmp_dir)
            for sample in range(num_training_samples, num_training_samples + num_validation_samples):
                np.save(file=os.path.join(tmp_dir, str(sample)), arr=action_data[sample])

            if test_per != 0:
                tmp_dir = os.path.join("test_data", action)
                if not os.path.exists(tmp_dir):
                    os.mkdir(tmp_dir)
                for sample in range(num_training_samples + num_validation_samples,
                                    num_training_samples + num_validation_samples + num_test_samples):
                    np.save(file=os.path.join(tmp_dir, str(sample)), arr=action_data[sample])

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
    if std_type == "feature_wise":
        for j in range(len(data[0, 0, :])):
            mean = data[:, :, j].mean()
            std = data[:, :, j].std()
            for k in range(len(data)):
                for i in range(len(data[0])):
                    data[k, i, j] = (data[k, i, j] - mean) / std

    if std_type == "sample_wise":
        for k in range(len(data)):
            mean = data[k].mean()
            std = data[k].std()
            data[k] -= mean
            data[k] /= std

    if std_type == "channel_wise":
        # this type of standardization prevents some channels to have more importance over others,
        # i.e. back head channels have more uVrms because of muscle tension in the back of the head
        # this way we prevent the network from concentrating too much on those features
        for k in range(len(data)):
            sample = data[k]
            for i in range(len(sample)):
                mean = sample[i].mean()
                std = sample[i].std()
                for j in range(len(sample[0])):
                    data[k, i, j] = (sample[i, j] - mean) / std
    
    return data

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
    - Standardize per channel
    - 60 Hz notch filter (BrainFlow if possible, SciPy fallback)
    - Bandpass filter (2–65 Hz by default)
    - Optional wavelet denoising
    - Compute FFTs

    :param data: ndarray, shape = (samples, channels, values)
    :param fs: sampling rate (Hz)
    :param lowcut: low cutoff for bandpass
    :param highcut: high cutoff for bandpass
    :param MAX_FREQ: number of FFT bins to keep
    :param power_hz: powerline noise frequency (60 Hz in US, 50 in EU)
    :param coi3order: wavelet denoising order (0 to disable)
    :return: tuple (filtered_data, fft_data)
    """

    data = standardize(data)  # normalize each channel

    n_samples, n_chans, n_points = data.shape
    fft_data = np.zeros((n_samples, n_chans, MAX_FREQ))

    for sample in range(n_samples):
        for channel in range(n_chans):
            # Ensure numpy float64 array
            signal = np.array(data[sample][channel], dtype=np.float64)

            # --- Notch filter (BrainFlow first, fallback to SciPy) ---
            try:
                DataFilter.perform_bandstop(
                    signal,
                    fs,
                    power_hz,          # center frequency
                    2.0,               # bandwidth (Hz)
                    2,                 # order (2–3 is stable for 250 samples)
                    FilterTypes.BUTTERWORTH.value,
                    0
                )
            except Exception as e:
                print(f"[WARN] BrainFlow notch failed on sample {sample}, ch {channel}: {e}")
                signal = notch_filter_scipy(signal, fs=fs, freq=power_hz)

            # --- Bandpass filtering (broad 2–120 Hz before denoise, then tighter 2–65 Hz) ---
            signal = butter_bandpass_filter(signal, 2, 120, fs, order=5)

            if coi3order != 0:
                try:
                    DataFilter.perform_wavelet_denoising(signal, 'coif3', coi3order)
                except Exception as e:
                    print(f"[WARN] Wavelet denoising failed: {e}")

            signal = butter_bandpass_filter(signal, lowcut, highcut, fs, order=5)

            # Save filtered signal
            data[sample][channel] = signal

            # --- FFT ---
            fft_data[sample][channel] = np.abs(fft(signal)[:MAX_FREQ])

    return np.array(data), np.array(fft_data)