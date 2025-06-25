#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mmWave to Soli-format Converter

This script converts raw radar data from Texas Instruments mmWave sensors
(specifically IWR6843AOP with DCA1000) into the HDF5-based Soli format.
"""

import argparse
import datetime
import logging
import json
import os
import sys
import time
from typing import Dict, List, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.signal import butter, filtfilt

from mmwave_config_parser import parse_config_file
from mmwave_shared import calculate_derived_parameters

# Constants
# IWR6843AOP has 3 TX and 4 RX antennas
NUM_TX = 3
NUM_RX = 4
EPSILON = 1e-10  # Small constant to prevent log(0)
BACKGROUND_FRAMES = 3  # Number of frames to use for background model initialization
BACKGROUND_LEARNING_RATE = 0.061  # Learning rate for background model update
DEFAULT_FRAMES = 40  # Default number of frames in output sequence


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Convert mmWave radar data to Soli HDF5 format.'
    )

    # Required arguments
    parser.add_argument('--cfg', required=True, help='Path to mmWaveStudio TXT configuration file')
    parser.add_argument('--bin', required=True, help='Path to raw adc_data.bin file')
    parser.add_argument('--out', required=True, help='Path for destination HDF5 file in Soli format')
    parser.add_argument('--label', required=True, type=int, help='Numeric gesture label')

    # Optional arguments
    parser.add_argument('--frames', type=int, default=DEFAULT_FRAMES,
                        help=f'Fixed number of frames for output sequence (default: {DEFAULT_FRAMES})')
    parser.add_argument('--show', action='store_true',
                        help='Display intermediate Range-Doppler images')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save intermediate Range-Doppler images to files instead of displaying them')
    parser.add_argument('--debug', action='store_true',
                        help='Enable additional details in plots when --show or --save-plots is active')
    parser.add_argument('--log', type=str, default=None, help='Optional log file path (all logs go here)')
    parser.add_argument('--verbosity', type=str, default='INFO', help='Console log level (DEBUG, INFO, WARNING, ERROR)')

    args = parser.parse_args()

    # Validate arguments
    if not os.path.isfile(args.cfg):
        print(f"Error: Configuration file {args.cfg} not found.", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.bin):
        print(f"Error: Binary data file {args.bin} not found.", file=sys.stderr)
        sys.exit(1)

    if args.frames <= 0:
        print("Error: Frames must be a positive integer.", file=sys.stderr)
        sys.exit(1)

    return args


def setup_logging(console_level: str, logfile: str = ""):
    """Set up logging with different levels for console and file."""
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    ch_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(ch_formatter)
    root_logger.addHandler(ch)
    # File handler (if specified)
    if logfile:
        fh = logging.FileHandler(logfile, mode='w')
        fh.setLevel(logging.DEBUG)  # Log everything to file
        fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(fh_formatter)
        root_logger.addHandler(fh)
    root_logger.setLevel(logging.DEBUG)


def print_derived_parameters(params: Dict) -> None:
    """
    Print derived radar parameters to console and log profile/chirp/frame config at DEBUG level.

    Args:
        params: Dictionary containing derived radar parameters
    """
    derived = params['derived']
    logger = logging.getLogger(__name__)
    logger.info("==========   Profile Parameters   ==========")
    logger.info(json.dumps(params, indent=4))
    logger.info("__________   Profile Parameters   __________")
    logger.info("Derived Radar Parameters:")
    logger.info(f"1. Sampling rate: {derived['sampling_rate_msps']:.2f} Msps")
    logger.info(f"2. ADC samples per chirp: {derived['adc_samples_per_chirp']}")
    logger.info(f"3. Chirp slope: {derived['chirp_slope_mhz_us']:.2f} MHz/µs")
    logger.info(f"4. Number of TX x RX antennas: {derived['num_tx']}x{derived['num_rx']}")
    logger.info(f"5. Range resolution: {derived['range_resolution_mm']:.2f} mm")
    logger.info(f"6. Maximum unambiguous range: {derived['max_range_mm']:.2f} mm")
    logger.info(f"7. PRF / Chirp repetition rate: {derived['prf_hz']:.2f} Hz")
    logger.info(f"8. Frame period: {derived['frame_period_ms']:.2f} ms, FPS: {derived['fps']:.2f}")
    logger.info(f"9. Total chirps per frame: {derived['chirps_per_frame']}")
    logger.info(f"10. Expected raw file size: {derived['expected_raw_size_bytes']} bytes")


def read_binary_data(bin_path: str, params: Dict) -> np.ndarray:
    """
    Read and reshape raw ADC data from binary file.

    Args:
        bin_path: Path to the binary file
        params: Dictionary containing radar parameters

    Returns:
        Reshaped raw data array (complex64)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Reading binary data from: {bin_path}")

    # Get file size
    actual_size_bytes = os.path.getsize(bin_path)
    expected_size_bytes = params['derived']['expected_raw_size_bytes']

    # Check file size
    if actual_size_bytes != expected_size_bytes:
        logger.warning(f"Warning: Expected raw file size {expected_size_bytes} bytes, "
                      f"actual size {actual_size_bytes} bytes.")

    # Read binary data as little-endian 16-bit signed integers
    raw_data = np.fromfile(bin_path, dtype=np.int16)

    # Diagnostic: Print first 16 raw int16 values
    logger.info(f"First 16 raw int16 values: {raw_data[:16]}")

    # Calculate expected dimensions
    num_rx = params['derived']['num_rx']
    adc_samples = params['derived']['adc_samples_per_chirp']
    chirps_per_frame = params['derived']['chirps_per_frame']
    samples_per_frame = num_rx * adc_samples * chirps_per_frame * 2  # 2 for I/Q

    # Calculate number of frames based on actual data size
    total_samples = raw_data.shape[0]
    if total_samples % samples_per_frame != 0:
        num_full_frames = total_samples // samples_per_frame
        truncated_samples = num_full_frames * samples_per_frame
        logger.warning(f"Truncating raw data: {total_samples} samples -> {truncated_samples} samples ({num_full_frames} full frames)")
        raw_data = raw_data[:truncated_samples]
    num_frames = raw_data.shape[0] // samples_per_frame

    logger.info(f"Detected {num_frames} frames in binary data")

    # Diagnostic: Print first 2 frames, 2 chirps, 2 rx, 4 adc, 2 (I/Q) values
    try:
        reshaped_preview = raw_data[:2*chirps_per_frame*num_rx*adc_samples*2].reshape(2, chirps_per_frame, num_rx, adc_samples, 2)
        logger.info(f"Reshaped preview shape: {reshaped_preview.shape}, dtype: {reshaped_preview.dtype}")
        logger.info(f"First frame, first chirp, first rx, first 8 adc I/Q pairs: {reshaped_preview[0,0,0,:8,:]}")
    except Exception as e:
        logger.warning(f"Could not print reshaped preview: {e}")

    # Reshape to [frames, chirps, rx, adc, 2] (I/Q)
    try:
        reshaped = raw_data.reshape(num_frames, chirps_per_frame, num_rx, adc_samples, 2)
        # Diagnostic: Print first 4 I/Q pairs for [0,0,0,:4,:]
        logger.info(f"First 4 I/Q pairs after reshape: {reshaped[0,0,0,:4,:]}")
        # Create dumps directory
        dump_dir = os.path.join(os.path.dirname(bin_path), "dumps")
        os.makedirs(dump_dir, exist_ok=True)
        # Dump ADC data for frames [0, 9, 19, 29, 39] for direct comparison
        adc_dump_indices = [0, 9, 19, 29, 39]
        for idx in adc_dump_indices:
            if idx < reshaped.shape[0]:
                np.save(os.path.join(dump_dir, f"adc_imported_frame{idx}.npy"), reshaped[idx])
                logger.info(f"Saved imported ADC data for frame {idx} to {os.path.join(dump_dir, f'adc_imported_frame{idx}.npy')}")
                # Print min/max/avg/pow for diagnostics
                adc_frame = reshaped[idx]
                min_val = adc_frame.min()
                max_val = adc_frame.max()
                avg_val = adc_frame.mean()
                pow_val = np.mean(np.abs(adc_frame)**2)
                logger.info(f"ADC frame {idx}: min={min_val:.2f}, max={max_val:.2f}, mean={avg_val:.2f}, pow={pow_val:.2f}")
        # Reconstruct complex data
        reshaped_data = reshaped[..., 0].astype(np.float32) + 1j * reshaped[..., 1].astype(np.float32)
        logger.info(f"First 4 complex values after reconstruction: {reshaped_data[0,0,0,:4]}")
        logger.info(f"Successfully reshaped data to shape: {reshaped_data.shape} (complex)")
        # Print stats for gesture and non-gesture frames if possible
        gesture_start = 5
        gesture_end = gesture_start + max(10, num_frames // 4)
        gesture_end = min(num_frames, gesture_end)
        if num_frames > gesture_start:
            gesture_slice = reshaped_data[gesture_start:gesture_end]
            non_gesture_slice = reshaped_data[:gesture_start]
            logger.info(f"Raw data stats (after decode):")
            logger.info(f"  Gesture frames {gesture_start}-{gesture_end-1}: max={gesture_slice.max()}, min={gesture_slice.min()}, mean={gesture_slice.mean():.1f}")
            logger.info(f"  Non-gesture frames 0-{gesture_start-1}: max={non_gesture_slice.max()}, min={non_gesture_slice.min()}, mean={non_gesture_slice.mean():.1f}")
        return reshaped_data
    except ValueError as e:
        logger.error(f"Error reshaping data: {e}")
        sys.exit(1)


def range_fft(raw_data, adc_samples):
    """Apply window and perform Range FFT along the last axis (ADC samples)."""
    window = signal.windows.hann(adc_samples)
    windowed_data = raw_data * window[np.newaxis, np.newaxis, np.newaxis, :]
    range_fft_data = np.fft.fft(windowed_data, axis=-1)
    return range_fft_data


def dc_removal(range_fft_data):
    """Subtract mean along chirp axis before Doppler FFT."""
    # Transpose to (num_frames, adc_samples, num_rx, chirps)
    range_fft_data = np.transpose(range_fft_data, (0, 3, 2, 1))
    range_fft_data = range_fft_data - np.mean(range_fft_data, axis=-1, keepdims=True)
    return range_fft_data


def doppler_fft(range_fft_data):
    """Perform Doppler FFT along chirps dimension (last axis)."""
    # Input shape: (num_frames, adc_samples, num_rx, chirps_per_frame)
    # Perform FFT along the last axis (chirps_per_frame)
    doppler_fft_data = np.fft.fft(range_fft_data, axis=-1)
    return doppler_fft_data


def doppler_bandpass_filter(doppler_fft_data, params, low_hz, high_hz):
    """Apply Butterworth band-pass filter on Doppler axis (before fftshift)."""
    num_doppler_bins = doppler_fft_data.shape[-1]
    logger = logging.getLogger(__name__)

    # Ensure 'prf_hz' is present and valid; it's critical for correct filter design.
    try:
        prf = float(params['derived']['prf_hz'])
    except KeyError:
        logger.error("Error: 'prf_hz' not found in derived parameters. This is essential for Doppler processing.")
        raise ValueError("'prf_hz' is missing from derived radar parameters.")
    except TypeError: # Handles cases where params['derived'] might be None or prf_hz is not float-convertible
        logger.error(f"Error: Could not interpret 'prf_hz' from derived parameters. Value: {params['derived'].get('prf_hz')}")
        raise ValueError("Invalid value or type for 'prf_hz' in derived radar parameters.")

    if prf <= 0:
        logger.error(f"Error: 'prf_hz' must be positive. Value: {prf}")
        raise ValueError("'prf_hz' must be a positive value.")

    nyq = 0.5 * prf
    if nyq <= 0: # Should be caught by prf <= 0, but as a safeguard
        logger.error(f"Error: Nyquist frequency must be positive. Calculated nyq: {nyq} from prf: {prf}")
        raise ValueError("Nyquist frequency must be positive.")

    low_norm = low_hz / nyq
    high_norm = high_hz / nyq

    # Validate normalized frequencies
    if not (0 < low_norm < 1):
        logger.error(f"Error: Normalized low cutoff frequency ({low_norm:.4f}) is out of valid range (0, 1). Original low_hz: {low_hz}, nyq: {nyq}")
        raise ValueError(f"Normalized low cutoff frequency ({low_norm:.4f}) is out of valid range (0, 1).")
    if not (0 < high_norm): # high_norm can be > 1, will be clamped
        logger.error(f"Error: Normalized high cutoff frequency ({high_norm:.4f}) must be positive. Original high_hz: {high_hz}, nyq: {nyq}")
        raise ValueError(f"Normalized high cutoff frequency ({high_norm:.4f}) must be positive.")
    if high_norm >= 1.0:
        logger.warning(f"Normalized high cutoff frequency ({high_norm:.4f}) is >= 1.0. Clamping to 0.99999. Original high_hz: {high_hz}, nyq: {nyq}")
        high_norm = 0.99999 # Use a value very close to 1 but not 1 itself for stability
    if low_norm >= high_norm:
        logger.error(f"Error: Normalized low cutoff ({low_norm:.4f}) must be less than normalized high cutoff ({high_norm:.4f}). Original low_hz: {low_hz}, high_hz: {high_hz}, nyq: {nyq}")
        raise ValueError(f"Normalized low cutoff ({low_norm:.4f}) must be less than normalized high cutoff ({high_norm:.4f}).")

    b, a = butter(N=4, Wn=[low_norm, high_norm], btype='band')
    logger.info(f"Applying Butterworth band-pass filter on Doppler axis: {low_hz}-{high_hz} Hz (normalized: {low_norm:.4f}-{high_norm:.4f}) BEFORE fftshift")

    # filtfilt handles complex data correctly by filtering real and imaginary parts.
    # Apply along the last axis (Doppler bins).
    # Using method="gust" can be faster for large inputs.
    doppler_fft_data_filtered = filtfilt(b, a, doppler_fft_data, axis=-1, method="gust")

    return doppler_fft_data_filtered


def center_zero_doppler(doppler_fft_data):
    """Center zero Doppler using fftshift (after filtering)."""
    return np.fft.fftshift(doppler_fft_data, axes=-1)


def suppress_zero_doppler(doppler_fft_data, left_width, right_width):
    """Zero out a band of Doppler bins around the center, with separate left and right widths."""
    num_doppler_bins = doppler_fft_data.shape[-1]
    center_bin = num_doppler_bins // 2
    left_start = max(center_bin - left_width, 0)
    right_end = min(center_bin + right_width + 1, num_doppler_bins)
    doppler_fft_data[..., left_start:right_end] = 0
    logger = logging.getLogger(__name__)
    logger.info(f"Suppressed Doppler bins {left_start} to {right_end-1} (zero-Doppler region, left_width={left_width}, right_width={right_width})")
    return doppler_fft_data


def magnitude_log_scaling(data, offset):
    """Calculate magnitude and apply log scaling."""
    magnitude_data = np.abs(data)
    log_data = 20 * np.log10(magnitude_data + offset)
    return log_data


def range_gating(data, params, max_distance_m):
    """Apply range gating to the data before smoothing, up to max_distance_m (meters)."""
    range_resolution_m = params['derived']['range_resolution_mm'] / 1000
    max_range_bin = int(max_distance_m / range_resolution_m)
    max_range_bin = min(max_range_bin, data.shape[1])
    gated = data[:, :max_range_bin, :, :]
    logger = logging.getLogger(__name__)
    logger.info(f"Range gating applied: {max_range_bin} range bins (up to {max_distance_m} m)")
    return gated


def interpolate_range_axis(data: np.ndarray, zoom_factor: float = 1.0, method: str = 'linear') -> np.ndarray:
    """
    Interpolate (upsample) the range axis by a zoom factor to increase range resolution.
    Args:
        data: Input array of shape (frames, range_bins, rx, doppler_bins)
        zoom_factor: Multiplicative factor for upsampling the range axis (e.g., 2.0 doubles the bins)
        method: Interpolation method ('linear', 'cubic', etc. for scipy.interpolate.interp1d)
    Returns:
        Interpolated array of shape (frames, int(range_bins * zoom_factor), rx, doppler_bins)
    """
    import scipy.ndimage # Added import
    frames, range_bins, rx, doppler_bins = data.shape

    if zoom_factor == 1.0: # No interpolation needed
        return data

    # Determine order for scipy.ndimage.zoom based on method
    if method == 'linear':
        order = 1
    elif method == 'cubic':
        order = 3
    else:
        order = 1 # Default to linear for other methods
        logging.getLogger(__name__).warning(
            f"Unsupported interpolation method '{method}' for 'interpolate_range_axis'. Defaulting to 'linear' (order=1)."
        )

    # Zoom factors for each dimension: (frames, range, rx, doppler)
    # We only want to zoom the range axis (axis 1).
    zoom_factors = (1, zoom_factor, 1, 1)

    # Perform the zoom operation
    # Note: scipy.ndimage.zoom might change the dtype depending on order and input.
    # We should ensure output dtype matches input if possible, or handle complex data appropriately.
    # For complex data, zoom real and imaginary parts separately.
    if np.iscomplexobj(data):
        real_zoomed = scipy.ndimage.zoom(data.real, zoom_factors, order=order, mode='constant', cval=0.0)
        imag_zoomed = scipy.ndimage.zoom(data.imag, zoom_factors, order=order, mode='constant', cval=0.0)
        out = real_zoomed + 1j * imag_zoomed
    else:
        out = scipy.ndimage.zoom(data, zoom_factors, order=order, mode='constant', cval=0.0)

    # target_num_bins might not be perfectly matched by zoom due to rounding,
    # but zoom calculates the output shape. Let's log the actual new shape.
    new_range_bins = out.shape[1]
    logger = logging.getLogger(__name__)
    logger.info(f"Upsampled range axis from {range_bins} to {new_range_bins} bins (target_zoom_factor={zoom_factor}) using scipy.ndimage.zoom (order={order}).")
    return out


def gaussian_smoothing(data, range_sigma, doppler_sigma):
    """Apply 2D Gaussian smoothing along Range and Doppler bins before background subtraction."""
    smoothed = np.empty_like(data)
    num_frames, _, num_rx, _ = data.shape
    for frame in range(num_frames):
        for rx in range(num_rx):
            smoothed[frame, :, rx, :] = gaussian_filter(data[frame, :, rx, :], sigma=(range_sigma, doppler_sigma))
    return smoothed


def temporal_filter(data, sigma):
    """Apply a 1D Gaussian filter along the time axis to suppress temporal noise."""
    from scipy.ndimage import gaussian_filter1d
    # data shape: (frames, range, rx, doppler)
    filtered = gaussian_filter1d(data, sigma=sigma, axis=0)
    logger = logging.getLogger(__name__)
    logger.info(f"Applied temporal Gaussian filter (sigma={sigma}) along frame axis.")
    return filtered


def iterative_background_subtraction(data, minimum, logger, alpha, max_iters, plateau_method='relative_drop', plateau_percent=0.05):
    """Iterative mean subtraction with ReLU until nonzero pixel count drops less than half of previous count, per channel.
    plateau_method: 'relative_drop' (default) or 'fixed_iters' (always run max_iters) or 'abs_drop' (absolute count drop)
    plateau_percent: threshold for relative drop (e.g., 0.05 for 5%)
    """
    num_frames, num_range_bins, num_rx, num_doppler_bins = data.shape
    history = ""
    bg_subtracted = np.zeros_like(data)
    plateau_iters = np.zeros((num_frames, num_rx), dtype=int)
    for frame in range(num_frames):
        history += "\n" if frame % 10 == 0 else " "
        history += f"F{frame:{len(str(num_frames))}}: "
        for rx in range(num_rx):
            frame_in = data[frame, :, rx, :]
            frame_proc = frame_in.copy()
            frame_proc = np.maximum(frame_proc, 0)
            nonzero_counts = []
            for it in range(max_iters):
                frame_proc = frame_proc - (frame_proc.mean() * alpha)
                frame_proc = np.maximum(frame_proc, 0)
                nonzero = np.count_nonzero(frame_proc > minimum)
                min_val = frame_proc.min()
                max_val = frame_proc.max()
                mean_val = frame_proc.mean()
                nonzero_counts.append(nonzero)
                if rx == 0 and ((frame > 5 and frame < 8) or (frame > 30 and frame < 34)):
                    logger.info(f"Frame {frame} RX {rx}: Iter {it}, min={min_val:.2f}, max={max_val:.2f}, mean={mean_val:.2f}, nonzero count={nonzero}")
                if plateau_method == 'relative_drop' and it > 0 and np.abs(nonzero_counts[-1] - nonzero_counts[-2]) < plateau_percent * nonzero_counts[-2]:
                    plateau_iters[frame, rx] = it
                    break
                elif plateau_method == 'abs_drop' and it > 0 and np.abs(nonzero_counts[-1] - nonzero_counts[-2]) < 10:
                    plateau_iters[frame, rx] = it
                    break
                elif plateau_method == 'fixed_iters':
                    continue
            else:
                plateau_iters[frame, rx] = max_iters
            bg_subtracted[frame, :, rx, :] = frame_proc
            history += f"{rx}:{plateau_iters[frame, rx]:>{2}},{np.count_nonzero(frame_proc):>{3}}"
    logger.info(f"Background subtraction: {history}")
    return bg_subtracted


def minmax_truncation(data, pmin=0, pmax=99):
    """Clip to given percentiles for each frame.
    Percentiles are calculated per frame across all its range, rx, and doppler bins.
    """
    # data shape: (frames, range_bins, num_rx, doppler_bins)
    # Calculate percentiles along axes 1, 2, and 3 (range, rx, doppler) for each frame.
    lower = np.percentile(data, pmin, axis=(1, 2, 3), keepdims=True)
    upper = np.percentile(data, pmax, axis=(1, 2, 3), keepdims=True)

    scaled_data = np.clip(data, lower, upper)
    return scaled_data


def detect_onset(scaled_data, BACKGROUND_FRAMES, logger):
    """Detect Doppler onset (first frame where energy exceeds threshold)."""
    frame_energy = np.sum(np.abs(scaled_data), axis=(1, 2, 3))
    bg_energy_mean = np.median(frame_energy[:BACKGROUND_FRAMES])
    bg_energy_std = np.std(frame_energy[:BACKGROUND_FRAMES])
    energy_threshold = bg_energy_mean + 3 * bg_energy_std
    onset_frames = np.where(frame_energy > energy_threshold)[0]
    if len(onset_frames) == 0:
        logger.warning("Warning: No gesture onset detected. Using first frame as onset.")
        onset_frame = 0
    else:
        onset_frame = onset_frames[0]
    logger.info(f"Gesture onset detected at frame {onset_frame}")
    return onset_frame


def temporal_trimming(scaled_data, onset_frame, num_output_frames, logger):
    """Trim or pad frames starting from onset to get fixed output length."""
    num_input_frames = scaled_data.shape[0]

    # Calculate how many frames are available *from* the onset_frame to the end of scaled_data
    # e.g., if num_input_frames = 10, onset_frame = 8. Frames 8, 9 are available. count = 10 - 8 = 2.
    # e.g., if num_input_frames = 10, onset_frame = 0. Frames 0-9 are available. count = 10 - 0 = 10.
    # If onset_frame >= num_input_frames, no frames are available from onset.
    frames_available_from_onset = max(0, num_input_frames - onset_frame)

    if frames_available_from_onset >= num_output_frames:
        # Enough frames are available, just take a slice
        start_index = onset_frame
        end_index = onset_frame + num_output_frames
        trimmed_data = scaled_data[start_index:end_index]
    else:
        # Not enough frames, need to copy what's available and then pad
        logger.warning(f"Warning: Only {frames_available_from_onset} frames available from onset frame {onset_frame} (total input {num_input_frames}). "
                      f"Padding to {num_output_frames} frames by repeating the last frame of input.")

        trimmed_data = np.zeros((num_output_frames, *scaled_data.shape[1:]), dtype=scaled_data.dtype)

        # Number of frames to copy directly from scaled_data
        num_to_copy_directly = frames_available_from_onset

        if num_to_copy_directly > 0:
            # Copy the available frames starting from onset_frame
            trimmed_data[:num_to_copy_directly] = scaled_data[onset_frame : onset_frame + num_to_copy_directly]

        # Determine how many frames need padding
        num_padding_frames = num_output_frames - num_to_copy_directly

        if num_padding_frames > 0:
            if num_input_frames > 0: # Ensure there is a last frame to use for padding
                # Use the global last frame of the original scaled_data for padding
                frame_to_pad_with = scaled_data[-1, np.newaxis, ...]

                # Construct the correct tile dimensions for np.tile
                # e.g., if scaled_data.ndim is 4, tile_shape = (num_padding_frames, 1, 1, 1)
                tile_shape = (num_padding_frames,) + tuple(1 for _ in range(scaled_data.ndim - 1))

                padding_block = np.tile(frame_to_pad_with, tile_shape)
                trimmed_data[num_to_copy_directly:] = padding_block
            else:
                # If scaled_data was empty, trimmed_data remains zeros, which is acceptable.
                logger.warning("Input 'scaled_data' for temporal_trimming is empty. Padding with zeros.")
    return trimmed_data


def ca_cfar_2d(data, num_train=4, num_guard=2, rate=1.5):
    """
    Apply 2D Cell-Averaging CFAR (CA-CFAR) to each frame and RX channel using convolution.
    Args:
        data: 4D numpy array (frames, range, rx, doppler), expected to be linear power.
        num_train: Number of training cells on each side of the guard band (range/doppler).
        num_guard: Number of guard cells on each side of the CUT (range/doppler).
        rate: Threshold scaling factor (multiplier for noise estimate).
    Returns:
        Binary mask of detections (same shape as data, dtype=bool).
    """
    logger = logging.getLogger(__name__)
    frames, _, _, _ = data.shape # Use _ for unused dimensions from shape
    output_mask = np.zeros_like(data, dtype=bool)

    if num_train <= 0:
        logger.warning("ca_cfar_2d: num_train is <= 0. CFAR is not meaningful and will not be applied. Returning empty mask.")
        return output_mask

    # Create 2D CFAR kernel
    # Total half-width of the window (including CUT and guard cells)
    total_half_width = num_train + num_guard
    kernel_size = 2 * total_half_width + 1

    cfar_kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)

    # Define the start and end of the guard/CUT region (center part to be zeroed out)
    # This region has half-width equal to num_guard around the CUT.
    guard_cut_half_width = num_guard
    center_start_idx = total_half_width - guard_cut_half_width # Start index of the central zeroed block
    center_end_idx = total_half_width + guard_cut_half_width + 1   # End index (exclusive) of the central zeroed block

    cfar_kernel[center_start_idx:center_end_idx, center_start_idx:center_end_idx] = 0

    num_training_cells_in_kernel = np.sum(cfar_kernel)

    if num_training_cells_in_kernel == 0:
        logger.error("ca_cfar_2d: Calculated number of training cells in kernel is 0. Check num_train and num_guard. Aborting CFAR.")
        return output_mask

    logger.info(f"Applying CA-CFAR with 2D kernel: num_train={num_train}, num_guard={num_guard}, "
                f"kernel_size={kernel_size}x{kernel_size}, num_training_cells={num_training_cells_in_kernel}, rate={rate}")

    for f_idx in range(frames):
        for rx_idx in range(data.shape[2]): # Iterate over RX channels
            current_slice = data[f_idx, :, rx_idx, :] # This is a 2D slice (range, doppler)

            # Calculate the sum of training cells around each cell using convolution
            # 'same' mode ensures output is same size as input, 'symm' handles boundaries by symmetric extension
            sum_of_training_cells = signal.convolve2d(current_slice, cfar_kernel, mode='same', boundary='symm')

            # Calculate noise estimate (average power of training cells)
            noise_estimate = sum_of_training_cells / num_training_cells_in_kernel

            # Calculate CFAR threshold
            threshold = noise_estimate * rate

            # Compare Cell Under Test (CUT) with its threshold
            output_mask[f_idx, :, rx_idx, :] = current_slice > threshold

    return output_mask


def pad_or_crop_to_32x32(img: np.ndarray) -> np.ndarray:
    """
    Pad or crop a 2D array to 32x32.
    Args:
        img: 2D numpy array (range_bins, doppler_bins)
    Returns:
        2D numpy array of shape (32, 32)
    """
    out = np.zeros((32, 32), dtype=img.dtype)
    r, d = img.shape
    r_crop = min(r, 32)
    d_crop = min(d, 32)
    out[:r_crop, :d_crop] = img[:r_crop, :d_crop]
    return out


def create_hdf5_output(processed_data: np.ndarray, output_path: str, params: Dict,
                       label: int, command_line: str) -> None:
    """
    Create HDF5 output file in Soli format.

    Args:
        processed_data: Processed radar data array
        output_path: Path for the output HDF5 file
        params: Dictionary containing radar parameters
        label: Gesture label
        command_line: Command-line string used to invoke the script
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating HDF5 output file: {output_path}")

    # Extract dimensions
    num_frames, num_range_bins, num_rx, num_doppler_bins = processed_data.shape

    with h5py.File(output_path, "w") as f:
        for rx in range(num_rx):
            # Pad/crop each frame's RDI to 32x32, then flatten
            channel_data = np.zeros((num_frames, 1024), dtype=np.float32)
            for i in range(num_frames):
                rdi = processed_data[i, :, rx, :]
                rdi_32 = pad_or_crop_to_32x32(rdi)
                channel_data[i] = rdi_32.flatten()
            f.create_dataset(f"ch{rx}", data=channel_data)

        label_data = np.full((num_frames, 1), label, dtype=np.int32)
        f.create_dataset("label", data=label_data)

        f.attrs["fps"] = float(params['derived']['fps'])
        f.attrs["range_bins"] = 32
        f.attrs["doppler_bins"] = 32

        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        comment = f"{timestamp} {command_line}"
        f.attrs["comment"] = comment

    logger.info(f"Successfully created HDF5 file with {num_frames} frames, 1024 features per channel (flattened 32x32)")


def display_range_doppler_images(processed_data: np.ndarray,
                                 raw_data: np.ndarray = None,
                                 show_interactive: bool = True,
                                 save_to_file: bool = False,
                                 debug_mode: bool = False,
                                 output_dir_for_screenshots: str = "screenshots") -> None:
    """
    Display intermediate Range-Doppler images for visualization.

    Args:
        processed_data: Processed radar data array (after full pipeline)
        raw_data: Optional, raw ADC data for plotting FFT-based RDI (for direct comparison with generator)
        show_interactive: If True, call plt.show().
        save_to_file: If True, save plot to a file.
        debug_mode: If True, save debug dumps
        output_dir_for_screenshots: Directory to save plots if save_to_file is True.
    """
    logger = logging.getLogger(__name__)
    # Create dumps directory
    dump_dir = os.path.join(os.path.dirname(__file__), "dumps")
    os.makedirs(dump_dir, exist_ok=True)

    # For processed data
    num_frames_proc, num_range_bins, num_rx, num_doppler_bins = processed_data.shape
    frame_indices_proc = [0, 9, 19, 29, 39]
    frame_indices_proc = [idx for idx in frame_indices_proc if idx < num_frames_proc]
    logger.info(f"Displaying/Saving final RDI comparison: num_frames_proc={num_frames_proc}, selected_indices={frame_indices_proc}, num_plot_columns={len(frame_indices_proc)}")

    # Plot both raw and processed RDI in the same figure, raw first, then processed
    if not (show_interactive or save_to_file):
        return # Don't bother creating plots if not showing or saving

    n_frames = len(frame_indices_proc)
    num_rows = 2 if raw_data is not None else 1
    fig, axes = plt.subplots(num_rows, n_frames, figsize=(3*n_frames, 4*num_rows), constrained_layout=True, squeeze=False)
    fig.suptitle("Range-Doppler Images (Channel 0): Top=Raw (pre-pipeline), Bottom=Processed (post-pipeline)")

    # Plot raw RDI (pre-pipeline) on top row
    if raw_data is not None:
        logger.info("Displaying Range-Doppler images after FFT/log (pre-pipeline)")
        num_frames_raw, chirps_per_frame, num_rx, adc_samples = raw_data.shape
        frame_indices_raw = frame_indices_proc  # Ensure alignment with processed plots
        im_raw_colorbar_source = None  # To store the last image object for the colorbar
        for i, frame_idx in enumerate(frame_indices_raw):
            frame = raw_data[frame_idx, :, 0, :]
            window = np.hanning(adc_samples)
            range_fft = np.fft.fft(frame * window, axis=-1)
            doppler_fft = np.fft.fft(range_fft, axis=0)
            rdi = np.abs(doppler_fft)
            rdi_log = 20 * np.log10(rdi + EPSILON) # Use EPSILON
            rdi_log_32 = pad_or_crop_to_32x32(rdi_log)
            im_raw_colorbar_source = axes[0, i].imshow(rdi_log_32.T, aspect='auto', cmap='viridis', origin='lower')
            axes[0, i].set_title(f"Frame {frame_idx}")
            axes[0, i].set_xlabel("Doppler Bin")
            axes[0, i].set_ylabel("Range Bin")
            if debug_mode:
                np.save(os.path.join(dump_dir, f"rdi_raw_frame{frame_idx}.npy"), rdi_log_32)
            logger.info(f"Saved raw RDI log data for frame {frame_idx} to {os.path.join(dump_dir, f'rdi_raw_frame{frame_idx}.npy')}")
            rdi_min = rdi_log_32.min()
            rdi_max = rdi_log_32.max()
            rdi_avg = rdi_log_32.mean()
            rdi_pow = np.mean(rdi_log_32 ** 2)
            logger.info(f"RDI raw frame {frame_idx}: min={rdi_min:.2f}, max={rdi_max:.2f}, avg={rdi_avg:.2f}, pow={rdi_pow:.2f}")
        if im_raw_colorbar_source: # Add colorbar if images were plotted
            fig.colorbar(im_raw_colorbar_source, ax=axes[0, :], orientation='horizontal', pad=0.1, fraction=0.08)

    # Plot processed RDI (post-pipeline) on bottom row
    logger.info("Displaying processed Range-Doppler images (post-pipeline)")
    im_processed_colorbar_source = None # To store the last image object for the colorbar
    for i, frame_idx in enumerate(frame_indices_proc):
        rdi_proc = processed_data[frame_idx, :, 0, :]
        rdi_proc_32 = pad_or_crop_to_32x32(rdi_proc)
        im_processed_colorbar_source = axes[1, i].imshow(rdi_proc_32.T, aspect='auto', cmap='viridis', origin='lower')
        axes[1, i].set_title(f"Frame {frame_idx}")
        axes[1, i].set_xlabel("Doppler Bin")
        axes[1, i].set_ylabel("Range Bin")
        if debug_mode:
            # Ensure dump_dir is defined relative to output, consistent with other debug dumps
            final_rdi_dump_path = os.path.join(os.path.dirname(output_dir_for_screenshots), "dumps", f"rdi_final_processed_frame{frame_idx}.npy")
            os.makedirs(os.path.dirname(final_rdi_dump_path), exist_ok=True)
            np.save(final_rdi_dump_path, rdi_proc_32)
            logger.info(f"DEBUG: Saved final processed RDI for frame {frame_idx} to {final_rdi_dump_path}")
        rdi_min = rdi_proc_32.min()
        rdi_max = rdi_proc_32.max()
        rdi_avg = rdi_proc_32.mean()
        rdi_pow = np.mean(rdi_proc_32 ** 2)
        logger.info(f"RDI gen frame {frame_idx}: min={rdi_min:.2f}, max={rdi_max:.2f}, avg={rdi_avg:.2f}, pow={rdi_pow:.2f}")
    if im_processed_colorbar_source: # Add colorbar if images were plotted
        fig.colorbar(im_processed_colorbar_source, ax=axes[num_rows-1, :], orientation='horizontal', pad=0.1, fraction=0.08)

    if save_to_file:
        os.makedirs(output_dir_for_screenshots, exist_ok=True)
        plot_filename = os.path.join(output_dir_for_screenshots, "final_rdi_comparison.png")
        plt.savefig(plot_filename)
        logger.info(f"Saved final RDI comparison plot to {plot_filename}")
        plt.close(fig)
    elif show_interactive:
        plt.show()
    else:
        plt.close(fig) # Should not happen if logic is correct, but good practice


def display_intermediate_rdi(rdi_data_to_plot: np.ndarray, figure_title_prefix: str,
                               show_interactive: bool, save_to_file: bool, debug_mode: bool,
                               output_dir_for_screenshots: str) -> None:
    """
    Display intermediate log-scaled Range-Doppler images for all 4 RX channels in the same plot.

    Args:
        rdi_data_to_plot: Log-scaled RDI data
                          Shape: (num_frames, num_range_bins, num_rx_channels, num_doppler_bins)
        figure_title_prefix: Prefix for the plot title
        show_interactive: If True, call plt.show().
        save_to_file: If True, save plot to a file.
        debug_mode: If True, add more debug information to plots.
        output_dir_for_screenshots: Directory to save plots if save_to_file is True.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Displaying {figure_title_prefix} Range-Doppler images for all RX channels")

    # Create dumps directory relative to the output file (sibling to screenshots)
    dump_dir = os.path.join(os.path.dirname(output_dir_for_screenshots), "dumps")
    os.makedirs(dump_dir, exist_ok=True)

    num_frames, num_range_bins, num_rx, num_doppler_bins = rdi_data_to_plot.shape
    frame_indices = [0, 9, 19, 29, 39]
    frame_indices = [idx for idx in frame_indices if idx < num_frames]
    logger.info(f"Displaying/Saving intermediate RDI plot '{figure_title_prefix}': num_frames_in_data={num_frames}, selected_indices={frame_indices}, num_plot_columns={len(frame_indices)}")
    n_frames = len(frame_indices)

    if not (show_interactive or save_to_file) or n_frames == 0:
        return # Don't bother creating plots if not showing or saving, or no frames to show

    # Ensure axes is always 2D, even if num_rx=1 or n_frames=1
    fig, axes = plt.subplots(num_rx, n_frames, figsize=(3*n_frames, 3*num_rx), constrained_layout=True, squeeze=False)
    # If num_rx is 1, axes might be 1D, so reshape if n_frames > 1
    # if num_rx == 1 and n_frames > 0 : axes = axes.reshape(1, n_frames) # squeeze=False handles this
    fig.suptitle(f"{figure_title_prefix} Range-Doppler Images (All RX Channels)")

    for rx in range(num_rx):
        for i, frame_idx in enumerate(frame_indices):
            rdi_slice = rdi_data_to_plot[frame_idx, :, rx, :]
            logger.info(f"Plotting RDI shape for frame {frame_idx} RX {rx}: {rdi_slice.shape}")
            current_ax = axes[rx, i]
            im_intermediate = current_ax.imshow(rdi_slice, aspect='auto', cmap='viridis', origin='lower')
            current_ax.set_title(f"Frame {frame_idx} RX {rx}")
            current_ax.set_xlabel("Doppler Bin")
            current_ax.set_ylabel("Range Bin")
            rdi_32 = pad_or_crop_to_32x32(rdi_slice)
            if debug_mode:
                intermediate_rdi_path = os.path.join(dump_dir, f"rdi_intermediate_frame{frame_idx}_rx{rx}.npy")
                np.save(intermediate_rdi_path, rdi_32)
                logger.info(f"DEBUG: Saved intermediate RDI for frame {frame_idx} RX {rx} to {intermediate_rdi_path}")
            rdi_min = rdi_32.min()
            rdi_max = rdi_32.max()
            rdi_avg = rdi_32.mean()
            rdi_pow = np.mean(rdi_32 ** 2)
            logger.info(f"RDI intermediate frame {frame_idx} RX {rx}: min={rdi_min:.2f}, max={rdi_max:.2f}, avg={rdi_avg:.2f}, pow={rdi_pow:.2f}")
        if n_frames > 0: # Add colorbar if images were plotted for this RX channel
            fig.colorbar(im_intermediate, ax=axes[rx, :], orientation='horizontal', pad=0.1, fraction=0.08)

    if save_to_file:
        os.makedirs(output_dir_for_screenshots, exist_ok=True)
        safe_title = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in figure_title_prefix).rstrip()
        plot_filename = os.path.join(output_dir_for_screenshots, f"{safe_title.replace(' ', '_').lower()}_all_rx.png")
        plt.savefig(plot_filename)
        logger.info(f"Saved intermediate RDI plot to {plot_filename}")
        plt.close(fig)
    elif show_interactive:
        plt.show()
    else:
        plt.close(fig) # Should not happen if logic is correct, but good practice


def process_radar_data(raw_data: np.ndarray, params: Dict, num_output_frames: int,
                      show_plots_in_pipeline: bool, # True if either interactive show or save to file is enabled
                      show_interactive_plots: bool, # True if interactive plt.show() is desired
                      save_plots_to_file: bool,     # True if plt.savefig() is desired
                      enable_debug_content: bool, # True if plot content should be more detailed
                      screenshots_dir_path: str,     # Path to save screenshots

                      do_dc_removal=False, #True, #
                      do_doppler_bandpass=False,
                      do_zero_doppler_suppression=True, #False, #
                      do_range_gating=True, #False, #
                      do_range_interpolation=False, #True, #
                      do_smoothing=False, #True,
                      do_temporal_filter=False, #True, #
                      do_background_subtraction= True, #False, #
                      do_minmax_truncation=False, #True, #
                      do_cfar=False, #True, #
                      do_onset_detection=True, #False, #
                      do_temporal_trimming=True) -> Tuple[np.ndarray, int]:
    """
    Process raw radar data through the signal processing pipeline, with optional stage skipping.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting signal processing pipeline")
    num_frames, chirps_per_frame, num_rx, adc_samples = raw_data.shape

    # 1. Range FFT
    range_fft_data = range_fft(raw_data, adc_samples)

    # 2. DC Removal
    if do_dc_removal:
        range_fft_data = dc_removal(range_fft_data)
    else:
        # Ensure correct shape for doppler_fft: (num_frames, adc_samples, num_rx, chirps_per_frame)
        range_fft_data = np.transpose(range_fft_data, (0, 3, 2, 1))

    # 3. Doppler FFT
    doppler_fft_data = doppler_fft(range_fft_data)

    # 4. Doppler Bandpass Filter (parameterized)
    if do_doppler_bandpass:
        doppler_fft_data = doppler_bandpass_filter(doppler_fft_data, params, low_hz=1.0, high_hz=1500.0)

    # 5. Center zero Doppler
    doppler_fft_data = center_zero_doppler(doppler_fft_data)

    # 7. Magnitude & Log scaling
    log_data = magnitude_log_scaling(doppler_fft_data, EPSILON)

    # 8. Range gating (parameterized)
    if do_range_gating:
        log_data_gated = range_gating(log_data, params, max_distance_m=0.4)
    else:
        log_data_gated = log_data

    # 8.5. Range axis interpolation (optional, after range gating, before smoothing)
    range_interp_zoom=2.0
    range_interp_method='linear'
    if do_range_interpolation and range_interp_zoom is not None and range_interp_zoom > 1.0:
        log_data_gated = interpolate_range_axis(log_data_gated, zoom_factor=range_interp_zoom, method=range_interp_method)

    # 9. Smoothing (parameterized)
    if do_smoothing:
        log_data_smoothed = gaussian_smoothing(log_data_gated, range_sigma=0.05, doppler_sigma=0.05)
    else:
        log_data_smoothed = log_data_gated
        # Temporal noise filter (parameterized)
        temporal_sigma=1.0
        if do_temporal_filter:
            log_data_smoothed = temporal_filter(log_data_smoothed, temporal_sigma)
        if show_plots_in_pipeline:
            display_intermediate_rdi(log_data_smoothed,
                                     f"After Temporal Gaussian Filter (sigma={temporal_sigma})",
                                     show_interactive=show_interactive_plots,
                                     save_to_file=save_plots_to_file,
                                     debug_mode=enable_debug_content,
                                     output_dir_for_screenshots=screenshots_dir_path)

    # 10. Background subtraction (parameterized)
    if do_background_subtraction:
        alpha = 1.0  # Background subtraction alpha
        bg_subtracted_data = iterative_background_subtraction(log_data_smoothed, EPSILON, logger, alpha, max_iters=10,
                                                              plateau_method='relative_drop', plateau_percent=0.01)
        if show_plots_in_pipeline:
            display_intermediate_rdi(bg_subtracted_data,
                                     f"After Iterative Background Subtraction (Mean+Solarized, alpha={alpha})",
                                     show_interactive=show_interactive_plots,
                                     save_to_file=save_plots_to_file,
                                     debug_mode=enable_debug_content,
                                     output_dir_for_screenshots=screenshots_dir_path)

    else:
        bg_subtracted_data = log_data_smoothed

    # 11. Min-max truncation
    if do_minmax_truncation:
        scaled_data = minmax_truncation(bg_subtracted_data, pmin=90, pmax=99)
    else:
        scaled_data = bg_subtracted_data

    # 11.5. CA-CFAR detection (optional, after min-max truncation)
    if do_cfar:
        # Convert dB scaled data to linear power for CFAR
        # CFAR needs to operate on linear power data for mean-based noise estimation
        logger.info("Converting scaled_data (dB) to linear power for CA-CFAR processing.")
        # Assuming scaled_data is P_dB = 20*log10(Amplitude) or P_dB = 10*log10(Power)
        # If P_dB = 20*log10(A), then Power_linear = A^2 = (10^(P_dB/20))^2 = 10^(P_dB/10).
        # If P_dB = 10*log10(P), then Power_linear = 10^(P_dB/10).
        # The formula 10**(scaled_data / 10.0) is correct for converting power in dB to linear power.
        scaled_data_linear = 10**(scaled_data / 10.0)

        cfar_mask = ca_cfar_2d(scaled_data_linear, num_train=4, num_guard=2, rate=1.5)

        # Apply mask to original dB-scaled data
        # Non-detections (mask is False, so ~cfar_mask is True) are zeroed out (or set to a very low dB value).
        # Setting to 0 in dB scale might be problematic if 0 is not the floor.
        # However, subsequent processing/output might handle 0 appropriately or it implies no signal.
        scaled_data[~cfar_mask] = 0
        logger.info("Applied CA-CFAR mask to original scaled_data (dB by zeroing out non-detections).")

        if show_plots_in_pipeline:
            display_intermediate_rdi(cfar_mask.astype(float), "CA-CFAR Detection Mask",
                                     show_interactive=show_interactive_plots,
                                     save_to_file=save_plots_to_file,
                                     debug_mode=enable_debug_content,
                                     output_dir_for_screenshots=screenshots_dir_path)


    # 6. Suppress zero Doppler (parameterized)
    if do_zero_doppler_suppression:
        scaled_data = suppress_zero_doppler(scaled_data, left_width=0, right_width=0)

    # 12. Onset detection
    if do_onset_detection:
        onset_frame = detect_onset(scaled_data, BACKGROUND_FRAMES, logger)
    else:
        onset_frame = 0
    # 13. Temporal trimming/padding
    if do_temporal_trimming:
        trimmed_data = temporal_trimming(scaled_data, onset_frame, num_output_frames, logger)
    else:
        trimmed_data = scaled_data
    if show_plots_in_pipeline:
        display_intermediate_rdi(trimmed_data, "Processed Data (Final Pipeline Stage)",
                                 show_interactive=show_interactive_plots,
                                 save_to_file=save_plots_to_file,
                                 debug_mode=enable_debug_content,
                                 output_dir_for_screenshots=screenshots_dir_path)

    # The shape of trimmed_data is expected to be (frames, range, rx, doppler),
    # which corresponds to axes (0, 1, 2, 3).
    # The transpose np.transpose(trimmed_data, (0, 1, 2, 3)) is redundant.
    # Ensure the second return value is the number of range bins (shape[1])
    return trimmed_data, trimmed_data.shape[1]


def main():
    """Main function to run the mmWave to Soli-format converter."""
    start_time = time.time()

    # Parse command-line arguments
    args = parse_args()

    # Set up logging
    setup_logging(args.verbosity, args.log)

    # Store command-line string for provenance
    command_line = f"--cfg {args.cfg} --bin {args.bin} --out {args.out} --label {args.label}"
    if args.frames != DEFAULT_FRAMES:
        command_line += f" --frames {args.frames}"
    if args.show: command_line += " --show"
    if args.save_plots: command_line += " --save-plots"
    if args.debug: command_line += " --debug"


    # Parse configuration file
    params = parse_config_file(args.cfg)

    # Print derived parameters
    print_derived_parameters(params)

    # Read and reshape binary data
    raw_data = read_binary_data(args.bin, params)

    # Determine plotting parameters
    debug_param = args.debug
    show_interactive_param = args.show
    save_plots_param = args.save_plots or (not show_interactive_param and debug_param)
    any_plotting_enabled = show_interactive_param or save_plots_param
    screenshots_directory = os.path.join(os.path.dirname(args.out), "screenshots")

    # Process radar data
    processed_data, num_range_bins = process_radar_data(
        raw_data, params, args.frames,
        show_plots_in_pipeline=any_plotting_enabled,
        show_interactive_plots=args.show,
        save_plots_to_file=save_plots_param,
        enable_debug_content=debug_param,
        screenshots_dir_path=screenshots_directory
    )

    # Create HDF5 output file
    create_hdf5_output(processed_data, args.out, params, args.label, command_line)

    # Display Range-Doppler images if requested
    if any_plotting_enabled:
        display_range_doppler_images(processed_data, raw_data,
                                     show_interactive=show_interactive_param, save_to_file=save_plots_param,
                                     debug_mode=debug_param, output_dir_for_screenshots=screenshots_directory)

    # Calculate and log processing time
    end_time = time.time()
    processing_time = end_time - start_time
    frames_processed = raw_data.shape[0]
    fps = frames_processed / processing_time

    logger = logging.getLogger(__name__)
    logger.info(f"Processing completed in {processing_time:.2f} seconds")
    logger.info(f"Processing speed: {fps:.2f} frames per second")

    return 0


if __name__ == "__main__":
    sys.exit(main())
