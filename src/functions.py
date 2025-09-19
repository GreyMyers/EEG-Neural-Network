from brainflow import DataFilter, FilterTypes, AggOperations
from scipy.signal import butter, lfilter
from matplotlib import pyplot as plt
from scipy.fft import fft

import numpy as np
import os
import pkg_resources

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

def preprocess_raw_eeg(data, fs=250, lowcut=2.0, highcut=65.0, MAX_FREQ=60, power_hz=60, coi3order=3):
    """
        Processes raw EEG data for model input, filters 60Hz noise from electroncis in US.
        :param data: ndarray, input dataset in to filter with shape=(samples, channels, values)
        :param fs: int, sampling rate
        :param lowcut: float, lower extreme for the bandpass filter
        :param highcut: float, higher extreme for the bandpass filter
        :param MAX_FREQ: int, maximum frequency for the FFTs
        :return: tuple, (ndarray, ndarray), process personal_dataset and FFTs respectively
    """

    data = standardize(data)

    fft_data = np.zeros((len(data), len(data[0]), MAX_FREQ))

    for sample in range(len(data)):
        for channel in range(len(data[0])):
            # bandpass filter
            DataFilter.perform_bandstop(data[sample][channel], 250, power_hz, 2.0, 5, FilterTypes.BUTTERWORTH.value, 0)

            data[sample][channel] = butter_bandpass_filter(data[sample][channel], 2, 120, fs, order=5)

            if coi3order != 0:
                DataFilter.perform_wavelet_denoising(data[sample][channel], 'coif3', coi3order)

            data[sample][channel] = butter_bandpass_filter(data[sample][channel], lowcut, highcut, fs, order=5)

            fft_data[sample][channel] = np.abs(fft(data[sample][channel])[:MAX_FREQ])

            visualize_data(data,
                   file_name="pictures/after_bandpass",
                   title=f'After bandpass from {lowcut}Hz to {highcut}Hz',
                   length=len(data[0, 0]))
            # visualize_data(fft_data,
            #               file_name="pictures/ffts",
            #               title="FFTs",
            #               length=len(fft_data[0, 0]))

        return np.array(data), np.array(fft_data)