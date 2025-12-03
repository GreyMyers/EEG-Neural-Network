"""
Visualize EEG Signal Processing Pipeline
Shows each step of the preprocessing for motor imagery classification
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, lfilter
from scipy.fft import fft
from brainflow import DataFilter, FilterTypes, WaveletTypes, NoiseEstimationLevelTypes, WaveletExtensionTypes, WaveletDenoisingTypes, ThresholdTypes
import os

# Set style for better visualization
plt.style.use('seaborn-v0_8-darkgrid')


def butter_bandpass(lowcut, highcut, fs, order=5):
    """Design butterworth bandpass filter"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply butterworth bandpass filter"""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def load_sample_data(dataset_dir="datasets/grey/motor"):
    """Load a sample EEG recording for visualization"""
    # Try to load a real sample
    left_dir = os.path.join(dataset_dir, "left")
    right_dir = os.path.join(dataset_dir, "right")
    
    sample = None
    label = None
    
    if os.path.exists(left_dir):
        files = sorted(os.listdir(left_dir))
        if files:
            sample = np.load(os.path.join(left_dir, files[0]))
            label = "left"
    
    if sample is None and os.path.exists(right_dir):
        files = sorted(os.listdir(right_dir))
        if files:
            sample = np.load(os.path.join(right_dir, files[0]))
            label = "right"
    
    if sample is None:
        # Generate synthetic EEG-like data for demo
        print("No dataset found, generating synthetic EEG data...")
        fs = 250
        t = np.linspace(0, 1, fs)
        sample = np.zeros((8, fs))
        for ch in range(8):
            # Mix of frequencies typical in EEG
            sample[ch] = (
                50 * np.sin(2 * np.pi * 10 * t) +  # Alpha (10 Hz)
                30 * np.sin(2 * np.pi * 20 * t) +  # Beta (20 Hz)
                20 * np.sin(2 * np.pi * 60 * t) +  # 60 Hz noise
                np.random.randn(fs) * 10           # Random noise
            )
        label = "synthetic"
    
    return sample, label


def visualize_processing_pipeline(sample, label="sample", save_path="pictures/signal_processing_pipeline.png"):
    """
    Visualize each step of the EEG signal processing pipeline
    """
    fs = 250  # Sampling frequency
    num_samples = sample.shape[1]
    t = np.arange(num_samples) / fs  # Time in seconds
    
    # Select channel to visualize (using channel 1, index 0)
    channel_idx = 0
    channel_name = f"Channel {channel_idx + 1}"
    
    # Store each processing stage
    stages = {}
    
    # =========================================
    # STAGE 0: Raw Signal
    # =========================================
    stages['0_raw'] = sample[channel_idx].copy()
    
    # =========================================
    # STAGE 1: 60 Hz Notch Filter (Remove power line noise)
    # =========================================
    stage1 = sample.copy()
    for ch in range(8):
        # BrainFlow bandstop filter for 60 Hz
        DataFilter.perform_bandstop(stage1[ch], fs, 59.0, 61.0, 5, FilterTypes.BUTTERWORTH.value, 0)
    stages['1_notch'] = stage1[channel_idx].copy()
    
    # =========================================
    # STAGE 2: Bandpass 7-45 Hz (Motor imagery band)
    # =========================================
    stage2 = stage1.copy()
    for ch in range(8):
        stage2[ch] = butter_bandpass_filter(stage2[ch], 7, 45, fs, order=5)
    stages['2_bandpass'] = stage2[channel_idx].copy()
    
    # =========================================
    # STAGE 3: Wavelet Denoising (Coif3)
    # =========================================
    stage3 = stage2.copy()
    for ch in range(8):
        DataFilter.perform_wavelet_denoising(
            stage3[ch], 
            WaveletTypes.COIF3.value,
            3,  # decomposition level
            WaveletDenoisingTypes.SURESHRINK.value,
            ThresholdTypes.SOFT.value,
            WaveletExtensionTypes.SYMMETRIC.value,
            NoiseEstimationLevelTypes.FIRST_LEVEL.value
        )
    stages['3_wavelet'] = stage3[channel_idx].copy()
    
    # =========================================
    # STAGE 4: Channel-wise Normalization
    # =========================================
    stage4 = stage3.copy()
    for ch in range(8):
        mean = stage4[ch].mean()
        std = stage4[ch].std()
        if std < 1e-10:
            std = 1.0
        stage4[ch] = (stage4[ch] - mean) / std
    stages['4_normalized'] = stage4[channel_idx].copy()
    
    # =========================================
    # STAGE 5: Model-ready shape (8 x 250) - ALL CHANNELS
    # =========================================
    # Store all 8 channels for visualization
    stages['5_model_ready'] = stage4.copy()  # All 8 channels
    
    # =========================================
    # STAGE 6: FFT Features
    # =========================================
    MAX_FREQ = 60
    fft_result = np.abs(fft(stage4[channel_idx])[:MAX_FREQ])
    stages['6_fft'] = fft_result
    
    # =========================================
    # CREATE VISUALIZATION
    # =========================================
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    # Color scheme
    colors = {
        '0_raw': '#e74c3c',        # Red
        '1_notch': '#e67e22',      # Orange
        '2_bandpass': '#f1c40f',   # Yellow
        '3_wavelet': '#2ecc71',    # Green
        '4_normalized': '#3498db', # Blue
        '5_model_ready': '#9b59b6',# Purple
        '6_fft': '#1abc9c'         # Teal
    }
    
    titles = {
        '0_raw': '1. Raw EEG Signal',
        '1_notch': '2. After 60 Hz Notch Filter\n(Remove power line noise)',
        '2_bandpass': '3. After 7-45 Hz Bandpass\n(Extract motor imagery band)',
        '3_wavelet': '4. After Wavelet Denoising (Coif3)\n(Remove remaining artifacts)',
        '4_normalized': '5. Channel-wise Normalization\n(Zero mean, unit variance)',
        '5_model_ready': '6. Model-Ready: All 8 Channels\n(Input shape: 8 x 250)',
        '6_fft': '7. FFT Features\n(Frequency domain representation)'
    }
    
    # Plot time-domain signals
    plot_positions = [
        (0, 0), (0, 1),  # Raw, Notch
        (1, 0), (1, 1),  # Bandpass, Wavelet
        (2, 0), (2, 1),  # Normalized, Model-ready
    ]
    
    time_stages = ['0_raw', '1_notch', '2_bandpass', '3_wavelet', '4_normalized', '5_model_ready']
    
    for i, stage_key in enumerate(time_stages):
        row, col = plot_positions[i]
        ax = fig.add_subplot(gs[row, col])
        
        if stage_key == '5_model_ready':
            # Plot all 8 channels with different colors
            channel_colors = plt.cm.tab10(np.linspace(0, 1, 8))
            for ch in range(8):
                ax.plot(t, stages[stage_key][ch], color=channel_colors[ch], 
                       linewidth=0.6, alpha=0.8, label=f'Ch{ch+1}')
            ax.legend(loc='upper right', fontsize=6, ncol=2)
            stats_text = f'Shape: 8 x {num_samples}'
        else:
            ax.plot(t, stages[stage_key], color=colors[stage_key], linewidth=0.8)
            data = stages[stage_key]
            stats_text = f'μ={data.mean():.2f}, σ={data.std():.2f}'
        
        ax.set_title(titles[stage_key], fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('Amplitude (μV)' if i < 4 else 'Normalized', fontsize=9)
        ax.set_xlim([0, t[-1]])
        ax.grid(True, alpha=0.3)
        
        # Add statistics annotation
        ax.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction',
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot FFT (bottom row, full width)
    ax_fft = fig.add_subplot(gs[3, :])
    freqs = np.arange(MAX_FREQ)
    ax_fft.bar(freqs, stages['6_fft'], color=colors['6_fft'], alpha=0.7, width=0.8)
    ax_fft.set_title(titles['6_fft'], fontsize=11, fontweight='bold', pad=10)
    ax_fft.set_xlabel('Frequency (Hz)', fontsize=9)
    ax_fft.set_ylabel('Magnitude', fontsize=9)
    ax_fft.set_xlim([0, MAX_FREQ])
    ax_fft.grid(True, alpha=0.3)
    
    # Highlight motor imagery bands
    ax_fft.axvspan(8, 12, alpha=0.2, color='red', label='Alpha (8-12 Hz)')
    ax_fft.axvspan(12, 30, alpha=0.2, color='blue', label='Beta (12-30 Hz)')
    ax_fft.legend(loc='upper right', fontsize=8)
    
    # Main title
    fig.suptitle(f'EEG Signal Processing Pipeline - {channel_name} ({label} imagery)',
                fontsize=14, fontweight='bold', y=0.98)
    
    # Save figure
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved pipeline visualization to: {save_path}")
    plt.show()
    
    return stages


def visualize_all_channels(sample, label="sample", save_path="pictures/all_channels_processed.png"):
    """
    Visualize all 8 channels before and after processing
    """
    fs = 250
    num_samples = sample.shape[1]
    t = np.arange(num_samples) / fs
    
    # Process all channels
    processed = sample.copy()
    for ch in range(8):
        # 60 Hz notch
        DataFilter.perform_bandstop(processed[ch], fs, 59.0, 61.0, 5, FilterTypes.BUTTERWORTH.value, 0)
        # Bandpass 7-45 Hz
        processed[ch] = butter_bandpass_filter(processed[ch], 7, 45, fs, order=5)
        # Wavelet denoising
        DataFilter.perform_wavelet_denoising(
            processed[ch],
            WaveletTypes.COIF3.value,
            3,
            WaveletDenoisingTypes.SURESHRINK.value,
            ThresholdTypes.SOFT.value,
            WaveletExtensionTypes.SYMMETRIC.value,
            NoiseEstimationLevelTypes.FIRST_LEVEL.value
        )
        # Normalize
        mean, std = processed[ch].mean(), processed[ch].std()
        if std < 1e-10:
            std = 1.0
        processed[ch] = (processed[ch] - mean) / std
    
    # Create figure
    fig, axes = plt.subplots(8, 2, figsize=(14, 16), sharex=True)
    
    channel_names = [f'Ch {i+1}' for i in range(8)]
    
    for ch in range(8):
        # Raw signal
        axes[ch, 0].plot(t, sample[ch], 'r-', linewidth=0.5, alpha=0.8)
        axes[ch, 0].set_ylabel(channel_names[ch], fontsize=9)
        axes[ch, 0].grid(True, alpha=0.3)
        
        # Processed signal
        axes[ch, 1].plot(t, processed[ch], 'b-', linewidth=0.5, alpha=0.8)
        axes[ch, 1].grid(True, alpha=0.3)
    
    # Column titles
    axes[0, 0].set_title('Raw EEG Signal', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Processed Signal\n(Notch → Bandpass → Wavelet → Normalize)', fontsize=12, fontweight='bold')
    
    # X-axis labels
    axes[7, 0].set_xlabel('Time (s)', fontsize=10)
    axes[7, 1].set_xlabel('Time (s)', fontsize=10)
    
    fig.suptitle(f'All 8 EEG Channels - {label} Motor Imagery', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved all channels visualization to: {save_path}")
    plt.show()


if __name__ == '__main__':
    print("=" * 60)
    print("EEG Signal Processing Pipeline Visualization")
    print("=" * 60)
    
    # Load sample data
    sample, label = load_sample_data()
    print(f"Loaded sample: shape={sample.shape}, label={label}")
    
    # Visualize processing pipeline
    print("\nGenerating pipeline visualization...")
    visualize_processing_pipeline(sample, label)
    
    # Visualize all channels
    print("\nGenerating all channels visualization...")
    visualize_all_channels(sample, label)
    
    print("\nDone!")

