"""
Extract C3, C4, Fp1, Fp2 channels from PhysioNet EEG Motor Movement/Imagery Dataset
and convert to the format used by this project.

Dataset: https://physionet.org/content/eegmmidb/1.0.0/
"""

import os
import numpy as np
from scipy.signal import resample
import argparse

try:
    import mne
except ImportError:
    print("ERROR: mne not installed. Install with: pip install mne")
    exit(1)

# Channel mapping: 10-10 system channel names to indices
# Based on the PhysioNet documentation, channels are in 10-10 system order
# Common mapping (may need adjustment based on actual file):
CHANNEL_NAMES_10_10 = [
    'Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8',
    'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8',
    'PO9', 'PO10', 'O1', 'Oz', 'O2'
]

# Target channels we want
TARGET_CHANNELS = {
    'C3': None,  # Will be set after reading file
    'C4': None,
    'Fp1': None,
    'Fp2': None
}

# PhysioNet dataset structure
# Tasks: T0 = rest, T1 = left fist, T2 = right fist
# Runs 3, 4, 7, 8, 11, 12: left/right fist tasks
# Runs 5, 6, 9, 10, 13, 14: both fists/both feet tasks

SAMPLING_RATE_ORIGINAL = 160  # Hz (PhysioNet dataset)
SAMPLING_RATE_TARGET = 250    # Hz (your project uses 250 Hz)
TRIAL_DURATION = 4            # seconds (matches your project)


def find_channel_indices(channel_labels):
    """
    Find indices of C3, C4, Fp1, Fp2 in the channel labels.
    Handles PhysioNet naming conventions (e.g., 'C3..', 'Fc5.')
    
    Args:
        channel_labels: List of channel names from the EDF file
        
    Returns:
        dict mapping channel names to indices
    """
    channel_map = {}
    for target in TARGET_CHANNELS.keys():
        found = False
        
        # Try exact match first
        if target in channel_labels:
            channel_map[target] = channel_labels.index(target)
            found = True
        else:
            # Try case-insensitive exact match
            target_lower = target.lower()
            for i, label in enumerate(channel_labels):
                if label.lower() == target_lower:
                    channel_map[target] = i
                    found = True
                    break
            
            if not found:
                # Try matching with trailing periods (PhysioNet uses 'C3..', 'Fc5.', etc.)
                # Strip periods from both for comparison
                target_clean = target.upper().rstrip('.')
                for i, label in enumerate(channel_labels):
                    label_clean = label.upper().rstrip('.')
                    if label_clean == target_clean:
                        channel_map[target] = i
                        found = True
                        break
                
                if not found:
                    # Try partial match (e.g., "C3.." contains "C3")
                    target_upper = target.upper()
                    for i, label in enumerate(channel_labels):
                        label_upper = label.upper().rstrip('.')
                        if target_upper in label_upper or label_upper.startswith(target_upper):
                            channel_map[target] = i
                            found = True
                            break
        
        if not found:
            # Channel not found - will be handled by caller
            pass
    
    return channel_map


def extract_channels_from_edf(edf_path, target_channels):
    """
    Extract target channels from an EDF file using MNE.
    
    Args:
        edf_path: Path to EDF file
        target_channels: Dict of {channel_name: index}
        
    Returns:
        tuple: (data_dict, channel_labels) where data_dict maps channel_name to numpy_array
    """
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as e:
        print(f"Error reading {edf_path}: {e}")
        return None
    
    # Get channel names
    channel_labels = raw.ch_names
    channel_map = find_channel_indices(channel_labels)
    
    # Verify we found at least the motor channels (C3, C4) - frontal channels are optional
    missing_motor = [ch for ch in ['C3', 'C4'] if ch not in channel_map]
    if missing_motor:
        print(f"Warning: Could not find motor channels {missing_motor} in {edf_path}")
        print(f"Available channels: {channel_labels[:10]}...")  # Show first 10
        return None
    
    # Frontal channels (Fp1, Fp2) are optional - warn but don't fail
    missing_frontal = [ch for ch in ['Fp1', 'Fp2'] if ch not in channel_map]
    if missing_frontal:
        print(f"Warning: Could not find frontal channels {missing_frontal} in {edf_path}")
        print(f"Available channels: {channel_labels[:10]}...")  # Show first 10
        # Continue processing - we'll just skip frontal trials
    
    # Extract data
    data = {}
    sfreq = raw.info['sfreq']  # Sampling frequency
    
    for ch_name, ch_idx in channel_map.items():
        signal = raw.get_data(ch_idx)[0]  # Get channel data
        
        # Resample from original rate to 250 Hz if needed
        if sfreq != SAMPLING_RATE_TARGET:
            num_samples_target = int(len(signal) * SAMPLING_RATE_TARGET / sfreq)
            signal = resample(signal, num_samples_target)
        
        data[ch_name] = signal
    
    return (data, channel_labels)


def extract_trials_from_annotations(edf_path, data_dict):
    """
    Extract trials based on annotations in the EDF file using MNE.
    
    Args:
        edf_path: Path to EDF file
        data_dict: Dict of channel data
        
    Returns:
        List of tuples: (trial_data, label, region)
        where trial_data is shape (2, 250) for motor or frontal
    """
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    except:
        return []
    
    # Get annotations from MNE
    annotations = raw.annotations
    
    # MNE annotations format: Onset (seconds), Duration (seconds), Description (str)
    # T0 = rest, T1 = left fist, T2 = right fist
    
    trials = []
    fs = SAMPLING_RATE_TARGET
    
    for onset, duration, description in zip(annotations.onset, annotations.duration, annotations.description):
        if not description.startswith('T'):
            continue
        
        # Convert onset to sample index
        onset_sample = int(onset * fs)
        duration_samples = int(duration * fs)
        end_sample = onset_sample + duration_samples
        
        # Extract trial window (4 seconds)
        trial_samples = TRIAL_DURATION * fs  # 4 * 250 = 1000 samples
        
        # Take middle portion of the annotation if it's longer
        if duration_samples > trial_samples:
            start_sample = onset_sample + (duration_samples - trial_samples) // 2
            end_sample = start_sample + trial_samples
        else:
            start_sample = onset_sample
            end_sample = start_sample + trial_samples
        
        # Get channel data
        if description == 'T1':  # Left fist
            # Motor region: C3, C4
            if 'C3' in data_dict and 'C4' in data_dict:
                motor_data = np.array([
                    data_dict['C3'][start_sample:end_sample],
                    data_dict['C4'][start_sample:end_sample]
                ])
                # Trim/pad to exactly 250 samples (1 second)
                if motor_data.shape[1] > 250:
                    motor_data = motor_data[:, :250]
                elif motor_data.shape[1] < 250:
                    pad_width = 250 - motor_data.shape[1]
                    motor_data = np.pad(motor_data, ((0, 0), (0, pad_width)), mode='constant')
                trials.append((motor_data, 'left', 'motor'))
                
        elif description == 'T2':  # Right fist
            # Motor region: C3, C4
            if 'C3' in data_dict and 'C4' in data_dict:
                motor_data = np.array([
                    data_dict['C3'][start_sample:end_sample],
                    data_dict['C4'][start_sample:end_sample]
                ])
                if motor_data.shape[1] > 250:
                    motor_data = motor_data[:, :250]
                elif motor_data.shape[1] < 250:
                    pad_width = 250 - motor_data.shape[1]
                    motor_data = np.pad(motor_data, ((0, 0), (0, pad_width)), mode='constant')
                trials.append((motor_data, 'right', 'motor'))
        
        # For frontal region, we can use T0 (rest) as "stop" and T1/T2 (task) as "forward"
        # Or use baseline runs (eyes open/closed)
        if description == 'T0':  # Rest
            if 'Fp1' in data_dict and 'Fp2' in data_dict:
                frontal_data = np.array([
                    data_dict['Fp1'][start_sample:end_sample],
                    data_dict['Fp2'][start_sample:end_sample]
                ])
                if frontal_data.shape[1] > 250:
                    frontal_data = frontal_data[:, :250]
                elif frontal_data.shape[1] < 250:
                    pad_width = 250 - frontal_data.shape[1]
                    frontal_data = np.pad(frontal_data, ((0, 0), (0, pad_width)), mode='constant')
                trials.append((frontal_data, 'stop', 'frontal'))
        elif description in ['T1', 'T2']:  # Task
            if 'Fp1' in data_dict and 'Fp2' in data_dict:
                frontal_data = np.array([
                    data_dict['Fp1'][start_sample:end_sample],
                    data_dict['Fp2'][start_sample:end_sample]
                ])
                if frontal_data.shape[1] > 250:
                    frontal_data = frontal_data[:, :250]
                elif frontal_data.shape[1] < 250:
                    pad_width = 250 - frontal_data.shape[1]
                    frontal_data = np.pad(frontal_data, ((0, 0), (0, pad_width)), mode='constant')
                trials.append((frontal_data, 'forward', 'frontal'))
    
    return trials


def process_physionet_dataset(physionet_dir, output_dir, user_name="physionet"):
    """
    Process all EDF files from PhysioNet dataset.
    
    Args:
        physionet_dir: Directory containing PhysioNet S001, S002, etc. folders
        output_dir: Output directory (will create datasets/{user_name}/motor and frontal)
        user_name: Name for the output dataset
    """
    motor_dir = os.path.join(output_dir, user_name, "motor")
    frontal_dir = os.path.join(output_dir, user_name, "frontal")
    
    os.makedirs(os.path.join(motor_dir, "left"), exist_ok=True)
    os.makedirs(os.path.join(motor_dir, "right"), exist_ok=True)
    os.makedirs(os.path.join(frontal_dir, "forward"), exist_ok=True)
    os.makedirs(os.path.join(frontal_dir, "stop"), exist_ok=True)
    
    # Count trials
    motor_left_count = 0
    motor_right_count = 0
    frontal_forward_count = 0
    frontal_stop_count = 0
    
    # Process each subject folder (S001, S002, etc.)
    subject_folders = sorted([f for f in os.listdir(physionet_dir) 
                             if os.path.isdir(os.path.join(physionet_dir, f)) and f.startswith('S')])
    
    print(f"Found {len(subject_folders)} subject folders")
    
    for subject_folder in subject_folders:
        subject_path = os.path.join(physionet_dir, subject_folder)
        edf_files = [f for f in os.listdir(subject_path) if f.endswith('.edf')]
        
        print(f"\nProcessing {subject_folder} ({len(edf_files)} files)...")
        
        for edf_file in edf_files:
            edf_path = os.path.join(subject_path, edf_file)
            
            # Extract channels (will return None if motor channels missing, but continue if only frontal missing)
            result = extract_channels_from_edf(edf_path, TARGET_CHANNELS)
            if result is None:
                continue  # Skip this file if we can't extract motor channels
            
            data_dict, channel_labels = result
            
            # Extract trials
            trials = extract_trials_from_annotations(edf_path, data_dict)
            
            # Save trials
            for trial_data, label, region in trials:
                if region == 'motor':
                    if label == 'left':
                        filename = os.path.join(motor_dir, "left", 
                                                f"{subject_folder}_{edf_file}_{motor_left_count}.npy")
                        motor_left_count += 1
                    else:  # right
                        filename = os.path.join(motor_dir, "right", 
                                                f"{subject_folder}_{edf_file}_{motor_right_count}.npy")
                        motor_right_count += 1
                else:  # frontal
                    if label == 'forward':
                        filename = os.path.join(frontal_dir, "forward", 
                                               f"{subject_folder}_{edf_file}_{frontal_forward_count}.npy")
                        frontal_forward_count += 1
                    else:  # stop
                        filename = os.path.join(frontal_dir, "stop", 
                                               f"{subject_folder}_{edf_file}_{frontal_stop_count}.npy")
                        frontal_stop_count += 1
                
                np.save(filename, trial_data)
        
        print(f"  {subject_folder}: {motor_left_count + motor_right_count} motor, "
              f"{frontal_forward_count + frontal_stop_count} frontal trials so far")
    
    print(f"\n=== Extraction Complete ===")
    print(f"Motor left: {motor_left_count} trials")
    print(f"Motor right: {motor_right_count} trials")
    print(f"Frontal forward: {frontal_forward_count} trials")
    print(f"Frontal stop: {frontal_stop_count} trials")
    print(f"\nData saved to: {os.path.join(output_dir, user_name)}")


def download_with_mne(output_dir):
    """
    Download PhysioNet dataset using MNE's built-in downloader.
    Requires: pip install mne[data]
    """
    try:
        from mne.datasets import eegbci
        print("Downloading PhysioNet EEG Motor Movement/Imagery Dataset using MNE...")
        print("This may take a while (dataset is ~1.9 GB)...")
        
        # MNE stores data in a specific location
        # We'll download it and then copy/move to our desired location
        data_path = eegbci.load_data(1, path=output_dir, update_path=True)
        print(f"Downloaded to: {data_path}")
        return data_path
    except ImportError:
        print("ERROR: MNE data downloader not available.")
        print("Install with: pip install mne[data]")
        return None
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Extract C3, C4, Fp1, Fp2 from PhysioNet EEG Motor Movement/Imagery Dataset'
    )
    parser.add_argument('--physionet-dir', type=str, default=r"C:\Users\greym\Xavier\physionet_data\eegmmidb\1.0.0",
                       help='Directory containing PhysioNet dataset (with S001, S002, etc. folders). If not provided, will try to auto-download.')
    parser.add_argument('--output-dir', type=str, default='datasets',
                       help='Output directory (default: datasets)')
    parser.add_argument('--user-name', type=str, default='physionet',
                       help='Name for the output dataset folder (default: physionet)')
    parser.add_argument('--download', action='store_true',
                       help='Auto-download dataset using MNE (requires mne[data])')
    
    args = parser.parse_args()
    
    # Auto-download if requested or if directory not provided
    if args.download or args.physionet_dir is None:
        print("Attempting to auto-download dataset...")
        download_dir = os.path.join(args.output_dir, "physionet_raw")
        os.makedirs(download_dir, exist_ok=True)
        
        downloaded_path = download_with_mne(download_dir)
        if downloaded_path:
            # MNE downloads to a specific structure, find the actual data directory
            if os.path.isdir(downloaded_path):
                args.physionet_dir = downloaded_path
            else:
                # Try to find S001 folder
                parent_dir = os.path.dirname(downloaded_path) if os.path.isfile(downloaded_path) else downloaded_path
                subject_dirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d)) and d.startswith('S')]
                if subject_dirs:
                    args.physionet_dir = parent_dir
                else:
                    print(f"Warning: Could not find subject folders. Please specify --physionet-dir manually.")
                    print(f"Downloaded files are in: {parent_dir}")
                    return
        else:
            print("\nAuto-download failed. Please download manually:")
            print("  1. Go to: https://physionet.org/content/eegmmidb/1.0.0/")
            print("  2. Download the ZIP file (1.9 GB)")
            print("  3. Extract to a folder and use --physionet-dir to point to it")
            return
    
    if not os.path.exists(args.physionet_dir):
        print(f"Error: PhysioNet directory not found: {args.physionet_dir}")
        print("\nTo download the dataset:")
        print("  1. Go to: https://physionet.org/content/eegmmidb/1.0.0/")
        print("  2. Download the ZIP file (1.9 GB)")
        print("  3. Extract and use --physionet-dir to point to the folder containing S001, S002, etc.")
        print("\nOr use --download flag to auto-download (requires mne[data])")
        return
    
    # Verify directory has subject folders
    subject_folders = [f for f in os.listdir(args.physionet_dir) 
                      if os.path.isdir(os.path.join(args.physionet_dir, f)) and f.startswith('S')]
    if not subject_folders:
        print(f"Error: No subject folders (S001, S002, etc.) found in {args.physionet_dir}")
        print("Please ensure the directory contains the extracted PhysioNet dataset.")
        return
    
    print(f"Found {len(subject_folders)} subject folders in {args.physionet_dir}")
    process_physionet_dataset(args.physionet_dir, args.output_dir, args.user_name)


if __name__ == "__main__":
    main()

