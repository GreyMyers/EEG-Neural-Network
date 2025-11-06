# ============================================
#  EEG Model Testing & Validation Script
# ============================================
# Computer-guided testing to evaluate model accuracy
# Tests: left, right, forward, stop commands

# Standard library imports
import argparse
import os
import time
from datetime import datetime

# Third-party imports
import numpy as np
from brainflow import BoardShim, BrainFlowInputParams, BoardIds
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow import keras
import seaborn as sns

# Local imports
from functions import preprocess_raw_eeg


# ============================================
# Configuration
# ============================================
TRIAL_DURATION = 4  # seconds per trial
SAMPLES_PER_SEC = 250
NUM_TRIALS_PER_CLASS = 5  # number of trials per action type
MOTOR_CH = [0, 1]  # C3, C4 (indices in eeg_channels array)
FRONTAL_CH = [2, 3]  # Fp1, Fp2 (indices in eeg_channels array)

# Action mappings
MOTOR_LABELS = ["left", "right"]
FRONTAL_LABELS = ["forward", "stop"]


# ============================================
# Data Collection
# ============================================
def collect_trial(board, trial_type):
    """
    Collect one trial of EEG data.
    
    Args:
        board: BoardShim instance
        trial_type: tuple of (motor_action, frontal_action) or None for rest
        
    Returns:
        numpy array of shape (channels, samples) - should be (4, 250) for C3, C4, Fp1, Fp2
    """
    board.get_board_data()  # clear old buffer
    time.sleep(TRIAL_DURATION)  # wait for trial duration
    raw_data = board.get_board_data()  # get all accumulated data
    
    eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD)
    sample = []
    for channel in eeg_channels:
        sample.append(raw_data[channel])
    
    sample_array = np.array(sample)
    
    # Extract exactly 250 samples (1 second at 250 Hz) from the trial
    # Take the last 250 samples from the collected data
    expected_samples = TRIAL_DURATION * SAMPLES_PER_SEC  # e.g., 4 * 250 = 1000 samples
    
    if sample_array.shape[1] >= SAMPLES_PER_SEC:
        # Take the last 250 samples (most recent 1 second)
        sample_array = sample_array[:, -SAMPLES_PER_SEC:]
    else:
        # If we don't have enough samples, pad with zeros
        print(f"Warning: Only collected {sample_array.shape[1]} samples, expected at least {SAMPLES_PER_SEC}")
        pad_width = SAMPLES_PER_SEC - sample_array.shape[1]
        sample_array = np.pad(sample_array, ((0, 0), (0, pad_width)), mode='constant')
    
    # Ensure we have exactly 4 channels (C3, C4, Fp1, Fp2)
    # Channels should be: [1, 2, 3, 4] from board, which map to indices [0, 1, 2, 3] in sample_array
    if sample_array.shape[0] < 4:
        print(f"Warning: Only {sample_array.shape[0]} channels available, expected 4")
    
    return sample_array


def run_test_sequence(board, motor_model, frontal_model, user_name, num_trials):
    """
    Run a guided test sequence for all action types.
    
    Args:
        num_trials: Number of trials per class
    
    Returns:
        Dictionary with test results
    """
    print("\n" + "="*60)
    print("EEG MODEL TESTING SEQUENCE")
    print("="*60)
    print(f"\nThis test will collect {num_trials} trials for each action type.")
    print("Follow the on-screen instructions for each trial.\n")
    
    input("Press Enter when ready to begin...")
    time.sleep(2)
    
    results = {
        'motor': {'true_labels': [], 'predicted_labels': [], 'confidences': []},
        'frontal': {'true_labels': [], 'predicted_labels': [], 'confidences': []},
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'user_name': user_name,
        'num_trials': num_trials
    }
    
    # Test sequences
    test_sequences = [
        # Motor cortex tests (left/right)
        ("motor", "left", "Imagine moving your LEFT hand"),
        ("motor", "right", "Imagine moving your RIGHT hand"),
        # Frontal lobe tests (forward/stop)
        ("frontal", "forward", "Concentrate or FOCUS (forward command)"),
        ("frontal", "stop", "RELAX or clear your mind (stop command)"),
    ]
    
    for region, action, instruction in test_sequences:
        print(f"\n{'='*60}")
        print(f"Testing {region.upper()} - {action.upper()}")
        print(f"{'='*60}")
        print(f"Instruction: {instruction}\n")
        
        for trial_num in range(num_trials):
            print(f"Trial {trial_num + 1}/{num_trials}")
            input(f"Press Enter when ready to perform {action.upper()}...")
            
            print(f"Performing {action}... ({TRIAL_DURATION} seconds)")
            sample = collect_trial(board, action)
            
            # Make prediction
            if region == "motor":
                motor_data = sample[MOTOR_CH, :]
                motor_input, _ = preprocess_raw_eeg(
                    motor_data.reshape((1, 2, 250)), fs=250, lowcut=7, highcut=45, coi3order=0
                )
                motor_input = motor_input.reshape((1, 2, 250, 1))
                prediction = motor_model.predict(motor_input, verbose=0)[0]
                
                predicted_idx = np.argmax(prediction)
                predicted_label = MOTOR_LABELS[predicted_idx]
                confidence = prediction[predicted_idx]
                
                results['motor']['true_labels'].append(action)
                results['motor']['predicted_labels'].append(predicted_label)
                results['motor']['confidences'].append(float(confidence))
                
                print(f"  → Predicted: {predicted_label.upper()} ({confidence*100:.1f}% confidence)")
                print(f"  → Expected:  {action.upper()}")
                print(f"  → {'✓ CORRECT' if predicted_label == action else '✗ INCORRECT'}\n")
                
            elif region == "frontal":
                frontal_data = sample[FRONTAL_CH, :]
                frontal_input, _ = preprocess_raw_eeg(
                    frontal_data.reshape((1, 2, 250)), fs=250, lowcut=7, highcut=45, coi3order=0
                )
                frontal_input = frontal_input.reshape((1, 2, 250, 1))
                prediction = frontal_model.predict(frontal_input, verbose=0)[0]
                
                predicted_idx = np.argmax(prediction)
                predicted_label = FRONTAL_LABELS[predicted_idx]
                confidence = prediction[predicted_idx]
                
                results['frontal']['true_labels'].append(action)
                results['frontal']['predicted_labels'].append(predicted_label)
                results['frontal']['confidences'].append(float(confidence))
                
                print(f"  → Predicted: {predicted_label.upper()} ({confidence*100:.1f}% confidence)")
                print(f"  → Expected:  {action.upper()}")
                print(f"  → {'✓ CORRECT' if predicted_label == action else '✗ INCORRECT'}\n")
            
            time.sleep(1)  # Brief pause between trials
    
    return results


# ============================================
# Results Analysis & Visualization
# ============================================
def calculate_metrics(results):
    """Calculate accuracy metrics for motor and frontal models."""
    metrics = {}
    
    for region in ['motor', 'frontal']:
        true_labels = results[region]['true_labels']
        predicted_labels = results[region]['predicted_labels']
        confidences = results[region]['confidences']
        
        # Convert to numeric labels
        if region == 'motor':
            label_map = {label: idx for idx, label in enumerate(MOTOR_LABELS)}
        else:
            label_map = {label: idx for idx, label in enumerate(FRONTAL_LABELS)}
        
        true_numeric = [label_map[label] for label in true_labels]
        pred_numeric = [label_map[label] for label in predicted_labels]
        
        # Overall accuracy
        accuracy = accuracy_score(true_numeric, pred_numeric)
        
        # Per-class accuracy
        class_accuracies = {}
        labels = MOTOR_LABELS if region == 'motor' else FRONTAL_LABELS
        for label in labels:
            label_idx = label_map[label]
            true_mask = np.array(true_numeric) == label_idx
            if np.any(true_mask):
                class_pred = np.array(pred_numeric)[true_mask]
                class_acc = np.mean(class_pred == label_idx)
                class_accuracies[label] = class_acc
        
        # Average confidence
        avg_confidence = np.mean(confidences)
        
        # Confusion matrix
        cm = confusion_matrix(true_numeric, pred_numeric, labels=list(range(len(labels))))
        
        metrics[region] = {
            'accuracy': accuracy,
            'class_accuracies': class_accuracies,
            'avg_confidence': avg_confidence,
            'confusion_matrix': cm,
            'labels': labels
        }
    
    return metrics


def plot_results(results, metrics, output_dir):
    """Generate and save visualization plots."""
    user_name = results['user_name']
    timestamp = results['timestamp']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Colors
    colors = {'motor': '#2E86AB', 'frontal': '#A23B72'}
    
    for idx, region in enumerate(['motor', 'frontal']):
        region_metrics = metrics[region]
        labels = region_metrics['labels']
        
        # 1. Confusion Matrix
        ax1 = fig.add_subplot(gs[idx, 0])
        cm = region_metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels,
                   ax=ax1, cbar_kws={'label': 'Count'})
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        ax1.set_title(f'{region.capitalize()} Confusion Matrix\n'
                     f'Accuracy: {region_metrics["accuracy"]*100:.1f}%')
        
        # 2. Per-class Accuracy Bar Chart
        ax2 = fig.add_subplot(gs[idx, 1])
        class_names = list(region_metrics['class_accuracies'].keys())
        class_accs = list(region_metrics['class_accuracies'].values())
        bars = ax2.bar(class_names, class_accs, color=colors[region], alpha=0.7)
        ax2.set_ylim([0, 1])
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'{region.capitalize()} Per-Class Accuracy')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars, class_accs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc*100:.1f}%', ha='center', va='bottom')
        
        # 3. Confidence Distribution
        ax3 = fig.add_subplot(gs[idx, 2])
        confidences = results[region]['confidences']
        ax3.hist(confidences, bins=20, color=colors[region], alpha=0.7, edgecolor='black')
        ax3.axvline(region_metrics['avg_confidence'], color='red', 
                   linestyle='--', linewidth=2, label=f'Mean: {region_metrics["avg_confidence"]:.3f}')
        ax3.set_xlabel('Confidence')
        ax3.set_ylabel('Frequency')
        ax3.set_title(f'{region.capitalize()} Confidence Distribution')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
    
    # 4. Bottom area left blank (text summary moved exclusively to .txt report)
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    plt.suptitle(f'EEG Model Testing Results - {user_name}', fontsize=16, fontweight='bold', y=0.995)
    
    # Save plot
    filename = f"pictures/{user_name}_{timestamp}_experiment_results.png"
    os.makedirs("pictures", exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nResults plot saved to: {filename}")
    
    plt.close()


def save_results_to_file(results, metrics, output_dir):
    """Save detailed results to a text file."""
    user_name = results['user_name']
    timestamp = results['timestamp']
    
    filename = f"pictures/{user_name}_{timestamp}_experiment_report.txt"
    
    with open(filename, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"EEG MODEL TESTING REPORT - {user_name.upper()}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Trials per class: {results['num_trials']}\n\n")
        
        for region in ['motor', 'frontal']:
            f.write(f"\n{'-'*60}\n")
            f.write(f"{region.upper()} MODEL RESULTS\n")
            f.write(f"{'-'*60}\n\n")
            
            f.write(f"Overall Accuracy: {metrics[region]['accuracy']*100:.2f}%\n")
            f.write(f"Average Confidence: {metrics[region]['avg_confidence']*100:.2f}%\n\n")
            
            f.write("Per-Class Accuracy:\n")
            for label, acc in metrics[region]['class_accuracies'].items():
                f.write(f"  {label.capitalize()}: {acc*100:.2f}%\n")
            
            f.write(f"\nConfusion Matrix:\n")
            f.write(f"  Labels: {metrics[region]['labels']}\n")
            cm_str = '\n'.join(['  ' + ' '.join(map(str, row)) for row in metrics[region]['confusion_matrix']])
            f.write(cm_str + "\n")
            
            f.write(f"\nDetailed Trial Results:\n")
            for i, (true, pred, conf) in enumerate(zip(
                results[region]['true_labels'],
                results[region]['predicted_labels'],
                results[region]['confidences']
            )):
                correct = "✓" if true == pred else "✗"
                f.write(f"  Trial {i+1}: {true.upper()} → {pred.upper()} "
                       f"({conf*100:.1f}%) {correct}\n")
    
    print(f"Detailed report saved to: {filename}")


# ============================================
# Main Entry
# ============================================
def main():
    parser = argparse.ArgumentParser(description='EEG Model Testing & Validation')
    parser.add_argument('--serial-port', type=str, default='COM3',
                       help='Serial port for OpenBCI board (default: COM3)')
    parser.add_argument('--user-name', type=str, default=None,
                       help='User name to load models from (default: prompts for input)')
    parser.add_argument('--trials', type=int, default=NUM_TRIALS_PER_CLASS,
                       help=f'Number of trials per class (default: {NUM_TRIALS_PER_CLASS})')
    args = parser.parse_args()
    
    num_trials = args.trials
    
    # Get user name
    if args.user_name:
        user_name = args.user_name
    else:
        user_name = input("Enter patient/user name to test models: ").strip()
    
    if not user_name:
        print("Error: User name cannot be empty")
        return
    
    # Load models
    user_models_dir = os.path.join("models", user_name)
    motor_model_path = os.path.join(user_models_dir, "motor_model.keras")
    frontal_model_path = os.path.join(user_models_dir, "frontal_model.keras")
    
    if not os.path.exists(motor_model_path):
        print(f"Error: Motor model not found at {motor_model_path}")
        return
    
    if not os.path.exists(frontal_model_path):
        print(f"Error: Frontal model not found at {frontal_model_path}")
        return
    
    print(f"\nLoading models from {user_models_dir}...")
    motor_model = keras.models.load_model(motor_model_path)
    frontal_model = keras.models.load_model(frontal_model_path)
    print("Models loaded successfully!")
    
    # Initialize board
    params = BrainFlowInputParams()
    params.serial_port = args.serial_port
    
    board = BoardShim(BoardIds.CYTON_BOARD.value, params)
    print(f"\nConnecting to board on {args.serial_port}...")
    board.prepare_session()
    print("Board session prepared. Starting stream...")
    board.start_stream()
    print("Stream started! USB dongle should be flashing red.\n")
    
    try:
        # Run test sequence
        results = run_test_sequence(board, motor_model, frontal_model, user_name, num_trials)
        
        # Calculate metrics
        print("\n" + "="*60)
        print("CALCULATING RESULTS...")
        print("="*60)
        metrics = calculate_metrics(results)
        
        # Print summary
        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY")
        print("="*60)
        print(f"\nMotor Model Accuracy: {metrics['motor']['accuracy']*100:.1f}%")
        print(f"Frontal Model Accuracy: {metrics['frontal']['accuracy']*100:.1f}%")
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        plot_results(results, metrics, "pictures")
        
        # Save detailed report
        save_results_to_file(results, metrics, "pictures")
        
        print("\n" + "="*60)
        print("TESTING COMPLETE!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    finally:
        board.stop_stream()
        board.release_session()


if __name__ == '__main__':
    main()

