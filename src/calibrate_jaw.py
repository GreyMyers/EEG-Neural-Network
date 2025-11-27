"""
Jaw Clench Calibration Script
Run this to find the optimal thresholds for jaw clench detection
"""

from brainflow import BoardShim, BrainFlowInputParams, BoardIds
import numpy as np
import time
import argparse


def calibrate_jaw_clench(serial_port='COM3', duration=30):
    """
    Calibrate jaw clench detection thresholds
    
    Args:
        serial_port: COM port for OpenBCI board
        duration: Calibration duration in seconds
    """
    print("="*60)
    print("JAW CLENCH CALIBRATION")
    print("="*60)
    
    # Setup board
    params = BrainFlowInputParams()
    params.serial_port = serial_port
    board = BoardShim(BoardIds.CYTON_BOARD.value, params)
    
    print(f"\nConnecting to OpenBCI on {serial_port}...")
    board.prepare_session()
    board.start_stream()
    
    print("\n" + "="*60)
    print("CALIBRATION INSTRUCTIONS:")
    print("="*60)
    print("1. First 10 seconds: RELAX your jaw (baseline)")
    print("2. Next 10 seconds: CLENCH your jaw repeatedly")
    print("3. Last 10 seconds: RELAX again")
    print("="*60)
    
    input("\nPress ENTER to start calibration...")
    
    relaxed_amplitudes = []
    relaxed_variances = []
    clenched_amplitudes = []
    clenched_variances = []
    
    start_time = time.time()
    phase = "BASELINE"
    phase_start = start_time
    
    try:
        while True:
            elapsed = time.time() - start_time
            
            # Update phase
            if elapsed < 10:
                if phase != "BASELINE":
                    phase = "BASELINE"
                    phase_start = time.time()
                    print("\n>>> RELAX YOUR JAW (baseline) <<<")
            elif elapsed < 20:
                if phase != "CLENCH":
                    phase = "CLENCH"
                    phase_start = time.time()
                    print("\n>>> CLENCH YOUR JAW REPEATEDLY <<<")
            elif elapsed < 30:
                if phase != "REST":
                    phase = "REST"
                    phase_start = time.time()
                    print("\n>>> RELAX AGAIN <<<")
            else:
                break
            
            # Get data
            time.sleep(0.5)
            data = board.get_current_board_data(250)
            
            if data.shape[1] < 250:
                continue
            
            # Get electrode 8 (channel index 7)
            eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
            jaw_channel = data[eeg_channels[7]]  # Electrode 8
            
            # Calculate metrics
            amplitude = np.max(np.abs(jaw_channel))
            variance = np.var(jaw_channel)
            
            # Store data
            if phase == "BASELINE" or phase == "REST":
                relaxed_amplitudes.append(amplitude)
                relaxed_variances.append(variance)
                status = "RELAXED"
            elif phase == "CLENCH":
                clenched_amplitudes.append(amplitude)
                clenched_variances.append(variance)
                status = "CLENCHED"
            
            # Display current values
            phase_elapsed = time.time() - phase_start
            print(f"\r[{phase}] Time: {phase_elapsed:.1f}s | "
                  f"Amplitude: {amplitude:.2f} | Variance: {variance:.2f} | "
                  f"Status: {status}     ", end='')
    
    except KeyboardInterrupt:
        print("\n\nCalibration interrupted!")
    
    finally:
        board.stop_stream()
        board.release_session()
    
    # Calculate statistics
    print("\n\n" + "="*60)
    print("CALIBRATION RESULTS")
    print("="*60)
    
    if relaxed_amplitudes and clenched_amplitudes:
        relaxed_amp_mean = np.mean(relaxed_amplitudes)
        relaxed_amp_std = np.std(relaxed_amplitudes)
        relaxed_amp_max = np.max(relaxed_amplitudes)
        
        relaxed_var_mean = np.mean(relaxed_variances)
        relaxed_var_std = np.std(relaxed_variances)
        relaxed_var_max = np.max(relaxed_variances)
        
        clenched_amp_mean = np.mean(clenched_amplitudes)
        clenched_amp_min = np.min(clenched_amplitudes)
        
        clenched_var_mean = np.mean(clenched_variances)
        clenched_var_min = np.min(clenched_variances)
        
        print(f"\nRELAXED JAW:")
        print(f"  Amplitude - Mean: {relaxed_amp_mean:.2f}, Std: {relaxed_amp_std:.2f}, Max: {relaxed_amp_max:.2f}")
        print(f"  Variance  - Mean: {relaxed_var_mean:.2f}, Std: {relaxed_var_std:.2f}, Max: {relaxed_var_max:.2f}")
        
        print(f"\nCLENCHED JAW:")
        print(f"  Amplitude - Mean: {clenched_amp_mean:.2f}, Min: {clenched_amp_min:.2f}")
        print(f"  Variance  - Mean: {clenched_var_mean:.2f}, Min: {clenched_var_min:.2f}")
        
        # Recommend thresholds (between max relaxed and min clenched)
        recommended_amp = (relaxed_amp_max + clenched_amp_min) / 2
        recommended_var = (relaxed_var_max + clenched_var_min) / 2
        
        print(f"\n" + "="*60)
        print("RECOMMENDED THRESHOLDS:")
        print("="*60)
        print(f"  --jaw-threshold {recommended_amp:.1f}")
        print(f"  --jaw-variance {recommended_var:.1f}")
        
        print(f"\n" + "="*60)
        print("USAGE EXAMPLE:")
        print("="*60)
        print(f"python src/real_time_plot.py --serial-port {serial_port} \\")
        print(f"       --jaw-threshold {recommended_amp:.1f} \\")
        print(f"       --jaw-variance {recommended_var:.1f}")
        print("="*60)
        
    else:
        print("\nERROR: Not enough data collected!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calibrate jaw clench detection')
    parser.add_argument('--serial-port', type=str, default='COM3',
                        help='Serial port for OpenBCI (default: COM3)')
    parser.add_argument('--duration', type=int, default=30,
                        help='Calibration duration in seconds (default: 30)')
    
    args = parser.parse_args()
    
    calibrate_jaw_clench(args.serial_port, args.duration)

