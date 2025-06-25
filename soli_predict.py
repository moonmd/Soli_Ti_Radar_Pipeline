#!/usr/bin/env python3
"""
Modified Soli Model Script for Single File Prediction
Windows Anaconda Compatible Version
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import argparse
import os
import sys
from pathlib import Path
import datetime

# Handle different Keras/TensorFlow versions
try:
    from keras.models import load_model
except ImportError:
    try:
        from tensorflow.keras.models import load_model
    except ImportError:
        print("Error: Could not import Keras/TensorFlow. Please install with:")
        print("conda install tensorflow")
        print("or")
        print("pip install tensorflow")
        sys.exit(1)

def log_print(message, logger=None):
    """
    Print message to screen and append to log file if logger is provided.
    """
    print(message)
    if logger is not None:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(logger, 'a', encoding='utf-8') as f:
            f.write(f"{message}\n")
            #f.write(f"[{timestamp}] {message}\n")

def check_dependencies(logger=None):
    """Check if all required packages are available"""
    required_packages = {
        'numpy': np,
        'h5py': h5py,
        'matplotlib': plt,
    }

    missing = []
    for name, module in required_packages.items():
        if module is None:
            missing.append(name)

    if missing:
        log_print("Missing required packages:", logger)
        for pkg in missing:
            log_print(f"  - {pkg}", logger)
        log_print("\nInstall with conda:", logger)
        log_print(f"conda install {' '.join(missing)}", logger)
        log_print("\nOr with pip:", logger)
        log_print(f"pip install {' '.join(missing)}", logger)
        return False
    return True

# Define gesture class names
CLASS_NAMES = [
    'Pinch Index', 'Pinch Pinky', 'Finger Slide', 'Finger Rub',
    'Slow Swipe', 'Fast Swipe', 'Push', 'Pull',
    'Palm Tilt', 'Circle', 'Palm Hold'
]

def load_h5_file(file_path, max_frames=20):
    """
    Load data from a single h5 file and extract all 4 channels

    Args:
        file_path: Path to the h5 file
        max_frames: Maximum number of frames to extract (default: 20)

    Returns:
        data: numpy array of shape (max_frames, 32, 32, 4)
        labels: numpy array of labels for each frame
    """
    data_channels = []
    labels = None

    with h5py.File(file_path, 'r') as f:
        # Extract data from all 4 channels
        for ch in range(4):
            channel_data = f[f'ch{ch}'][()]
            channel_data = channel_data.reshape(channel_data.shape[0], 32, 32)

            # Take only the first max_frames frames
            if channel_data.shape[0] >= max_frames:
                channel_data = channel_data[:max_frames, :, :]
            else:
                # Pad with zeros if fewer frames available
                padded_data = np.zeros((max_frames, 32, 32))
                padded_data[:channel_data.shape[0], :, :] = channel_data
                channel_data = padded_data

            data_channels.append(channel_data)

        # Extract labels (only need to do this once)
        labels = f['label'][()]
        if len(labels) >= max_frames:
            labels = labels[:max_frames]
        else:
            # Pad labels if needed
            padded_labels = np.zeros(max_frames)
            padded_labels[:len(labels)] = labels
            labels = padded_labels

    # Combine all channels: shape will be (max_frames, 32, 32, 4)
    combined_data = np.stack(data_channels, axis=-1)

    return combined_data, labels

def predict_single_file(model_path, h5_file_path, verbose=True, logger=None):
    """
    Make prediction on a single h5 file

    Args:
        model_path: Path to the trained Keras model
        h5_file_path: Path to the h5 file to predict
        verbose: Whether to print detailed results
        logger: Path to log file (optional)

    Returns:
        prediction: Predicted class index
        confidence: Prediction confidence (probability)
        prediction_name: Human-readable class name
    """

    # Convert paths to Path objects for cross-platform compatibility
    model_path = Path(model_path)
    h5_file_path = Path(h5_file_path)

    # Load the trained model
    if verbose:
        log_print(f"Loading model from: {model_path}", logger)
    model = load_model(str(model_path))

    # Load and preprocess the h5 file
    if verbose:
        log_print(f"Loading data from: {h5_file_path}", logger)
    data, labels = load_h5_file(str(h5_file_path))

    if verbose:
        log_print(f"Data shape: {data.shape}", logger)
        log_print(f"Labels shape: {labels.shape}", logger)
        log_print(f"Unique labels in file: {np.unique(labels)}", logger)

    # Reshape data for model input (add batch dimension)
    # Model expects shape: (batch_size, frames, height, width, channels)
    input_data = np.expand_dims(data, axis=0)

    if verbose:
        log_print(f"Input shape for model: {input_data.shape}", logger)

    # Make prediction
    predictions = model.predict(input_data, verbose=0)

    # Get the predicted class and confidence
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    prediction_name = CLASS_NAMES[predicted_class]

    if verbose:
        log_print(f"\n=== PREDICTION RESULTS ===", logger)
        log_print(f"Predicted Class: {predicted_class}", logger)
        log_print(f"Prediction Name: {prediction_name}", logger)
        log_print(f"Confidence: {confidence:.4f}", logger)
        log_print(f"All Probabilities:", logger)
        for i, prob in enumerate(predictions[0]):
            log_print(f"  {CLASS_NAMES[i]}: {prob:.4f}", logger)

    return predicted_class, confidence, prediction_name

def visualize_data(h5_file_path, num_frames=5, channel=0):
    """
    Visualize frames from the h5 file

    Args:
        h5_file_path: Path to the h5 file
        num_frames: Number of frames to display
        channel: Which channel to visualize (0-3)
    """
    data, labels = load_h5_file(h5_file_path)

    plt.figure(figsize=(15, 3))
    for i in range(min(num_frames, data.shape[0])):
        plt.subplot(1, num_frames, i+1)
        plt.imshow(data[i, :, :, channel], cmap='hot')
        label_val = labels[i].item() if hasattr(labels[i], 'item') else float(labels[i])
        plt.title(f'Frame {i}\nLabel: {int(label_val)}')
        plt.axis('off')

    plt.suptitle(f'Visualization of {num_frames} frames from channel {channel}')
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Make prediction on a single Soli h5 file')
    parser.add_argument('--model_path', required=True, help='Path to the trained Keras model')
    parser.add_argument('--h5_file', required=True, help='Path to the h5 file to predict')
    parser.add_argument('--visualize', action='store_true', help='Visualize the input data')
    parser.add_argument('--channel', type=int, default=0, help='Channel to visualize (0-3)')
    parser.add_argument('--frames', type=int, default=5, help='Number of frames to visualize')
    parser.add_argument('--log', type=str, default=None, help='Path to log file (optional)')

    args = parser.parse_args()

    # Convert to Path objects for cross-platform compatibility
    model_path = Path(args.model_path)
    h5_file_path = Path(args.h5_file)
    logger = args.log

    # Check if files exist
    if not model_path.exists():
        log_print(f"Error: Model file not found: {model_path}", logger)
        return

    if not h5_file_path.exists():
        log_print(f"Error: H5 file not found: {h5_file_path}", logger)
        return

    # Make prediction
    try:
        predicted_class, confidence, prediction_name = predict_single_file(
            str(model_path),
            str(h5_file_path),
            verbose=True,
            logger=logger
        )

        # Visualize if requested
        if args.visualize:
            visualize_data(str(h5_file_path), args.frames, args.channel)

    except Exception as e:
        log_print(f"Error during prediction: {str(e)}", logger)
        import traceback
        traceback.print_exc()

# Example usage functions for Jupyter notebook
def predict_example(model_path, h5_file_path, logger=None):
    """
    Simple function for Jupyter notebook usage
    """
    return predict_single_file(model_path, h5_file_path, verbose=True, logger=logger)

def batch_predict(model_path, h5_files_list, logger=None):
    """
    Make predictions on multiple files

    Args:
        model_path: Path to the trained model
        h5_files_list: List of h5 file paths
        logger: Path to log file (optional)

    Returns:
        results: List of (filename, predicted_class, confidence, prediction_name) tuples
    """
    results = []
    model = load_model(model_path)

    for h5_file in h5_files_list:
        try:
            data, _ = load_h5_file(h5_file)
            input_data = np.expand_dims(data, axis=0)
            predictions = model.predict(input_data, verbose=0)

            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            prediction_name = CLASS_NAMES[predicted_class]

            results.append((h5_file, predicted_class, confidence, prediction_name))
            log_print(f"{h5_file}: {prediction_name} (confidence: {confidence:.4f})", logger)

        except Exception as e:
            log_print(f"Error processing {h5_file}: {str(e)}", logger)
            results.append((h5_file, -1, 0.0, "ERROR"))

    return results

if __name__ == "__main__":
    main()