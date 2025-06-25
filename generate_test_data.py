#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Test Data for mmWave to Soli-format Converter

This script generates synthetic binary data that mimics the format of raw ADC data
from an IWR6843AOP mmWave sensor with DCA1000 capture card. The generated data can be
used for testing the mmwave_to_soli.py script without having actual hardware data.
"""

import argparse
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from mmwave_config_parser import parse_config_file

# Constants
NUM_RX = 4  # IWR6843AOP has 4 RX antennas


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic mmWave radar data for testing.'
    )

    parser.add_argument('--cfg', required=True, help='Path to mmWaveStudio TXT configuration file')
    parser.add_argument('--out', required=True, help='Path for output binary file')
    parser.add_argument('--frames', type=int, default=100, help='Number of frames to generate (default: 100)')
    parser.add_argument('--gesture', action='store_true', help='Include a synthetic gesture in the data')
    parser.add_argument('--show', action='store_true',
                        help='Display Range-Doppler images of synthetic data interactively')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save Range-Doppler images to files instead of displaying them')
    parser.add_argument('--debug', action='store_true',
                        help='Enable additional debug outputs (e.g., .npy files) when --show or --save-plots is active')
    parser.add_argument('--no-noise', action='store_true', help='Disable background noise in synthetic data')

    args = parser.parse_args()

    # Validate arguments
    if not os.path.isfile(args.cfg):
        print(f"Error: Configuration file {args.cfg} not found.")
        sys.exit(1)

    if args.frames <= 0:
        print("Error: Frames must be a positive integer.")
        sys.exit(1)

    return args


def generate_synthetic_data(params, num_frames, include_gesture=False, no_noise=False):
    """
    Generate synthetic mmWave radar data.
    If include_gesture is True, inject a gesture as a blob in the Range-Doppler (RDI) domain,
    then use inverse FFTs to generate the corresponding ADC data.
    Returns:
        data_complex: 4D complex array [frames, chirps, rx, adc]
    """
    print(f"Generating synthetic data for {num_frames} frames (RDI-domain gesture mode)")

    adc_samples = params['profile']['adc_samples']
    chirps_per_frame = params['chirps_per_frame']
    num_rx = NUM_RX

    print(f"  Data shape will be: [frames={num_frames}, chirps/frame={chirps_per_frame}, RX={num_rx}, ADC samples/chirp={adc_samples}]")

    # Print parsed config for debugging (like the consumer)
    print("Parsed configuration parameters:")
    print(f"  ADC samples per chirp: {params['profile']['adc_samples']}")
    print(f"  Chirps per frame: {params['chirps_per_frame']}")
    print(f"  RX antennas: {NUM_RX}")
    print(f"  Frames to generate: {params['frame']['num_frames']}")
    print(f"  Profile: {params['profile']}")
    print(f"  Chirp: {params['chirp']}")
    print(f"  Frame: {params['frame']}")

    # Set noise standard deviation (much lower for less noise)
    noise_std = 0.005

    # Start with noise or zeros
    if not no_noise:
        data = np.random.normal(0, noise_std, (num_frames, chirps_per_frame, num_rx, adc_samples)).astype(np.complex64)
    else:
        data = np.zeros((num_frames, chirps_per_frame, num_rx, adc_samples), dtype=np.complex64)

    # --- Patch: Keep gesture blob within first 32 range bins ---
    RDI_RANGE_BINS = 32

    if include_gesture:
        print("Injecting gesture as a bright, strictly Gaussian blob in the Range-Doppler domain (RDI)...")
        gesture_start = 5
        gesture_duration = max(10, num_frames // 4)
        gesture_end = min(num_frames, gesture_start + gesture_duration)
        patch_size_range = max(2, RDI_RANGE_BINS // 16)  # Reduced patch size
        patch_size_doppler = max(2, chirps_per_frame // 16) # Reduced patch size
        sigma_range = patch_size_range / 2.0
        sigma_doppler = patch_size_doppler / 2.0
        blob_amplitude = 5000.0  # Much brighter blob
        for frame in range(gesture_start, gesture_end):
            # Center moves slightly with progress, but amplitude is fixed and high
            progress = (frame - gesture_start) / gesture_duration
            if progress > 0.5:
                progress = 1 - progress
            center_range = int(RDI_RANGE_BINS * (0.2 + 0.5 * progress))
            center_doppler = int(chirps_per_frame * (0.2 + 0.5 * progress))
            for rx in range(num_rx):
                rdi = np.zeros((adc_samples, chirps_per_frame), dtype=np.complex64)
                for dr in range(-patch_size_doppler//2, patch_size_doppler//2+1):
                    for rr in range(-patch_size_range//2, patch_size_range//2+1):
                        d_idx = center_doppler + dr
                        r_idx = center_range + rr
                        if 0 <= d_idx < chirps_per_frame and 0 <= r_idx < RDI_RANGE_BINS:
                            # Strict 2D Gaussian
                            weight = np.exp(-0.5 * (dr/sigma_doppler)**2 - 0.5 * (rr/sigma_range)**2)
                            rdi[r_idx, d_idx] += blob_amplitude * weight
                time_data = np.fft.ifft2(rdi, s=(adc_samples, chirps_per_frame))
                data[frame, :, rx, :] += time_data.T
        gesture_slice = data[gesture_start:gesture_end].real
        non_gesture_slice = data[:gesture_start].real
        print(f"Gesture frames: {gesture_start}-{gesture_end-1}")
        print(f"  Gesture frame max: {gesture_slice.max()}, min: {gesture_slice.min()}, mean: {gesture_slice.mean():.1f}")
        print(f"  Non-gesture frame max: {non_gesture_slice.max()}, min: {non_gesture_slice.min()}, mean: {non_gesture_slice.mean():.1f}")
    print(f"Synthetic data generated. Final shape: {data.shape}, dtype: {data.dtype}")
    # Return only the 4D complex array for downstream use
    return data


def save_binary_data(data, output_path):
    """
    Save synthetic data to binary file.

    Args:
        data: Numpy array containing synthetic data
        output_path: Path for output binary file
    """
    print(f"Saving synthetic data to: {output_path}")

    # Flatten the array and save as little-endian 16-bit signed integers
    data.astype(np.int16).tofile(output_path)

    file_size = os.path.getsize(output_path)
    print(f"Generated binary file size: {file_size} bytes")


def main():
    """Main function to generate test data."""
    # Parse command-line arguments
    args = parse_args()

    # Parse configuration file
    params = parse_config_file(args.cfg)

    # Generate synthetic data
    data_complex = generate_synthetic_data(params, args.frames, args.gesture, args.no_noise)

    # Convert to int16 IQ for all downstream use
    num_frames, chirps_per_frame, num_rx, adc_samples = data_complex.shape

    # Apply scale factor before quantization to int16
    SCALE_FACTOR = 2000.0  # Empirically chosen to ensure gesture survives quantization
    print(f"[DEBUG] Applying scale factor {SCALE_FACTOR} before int16 quantization.")
    data_scaled = data_complex * SCALE_FACTOR
    print(f"[DEBUG] After scaling: max={data_scaled.max()}, min={data_scaled.min()}, mean={data_scaled.mean():.1f}")

    data_iq = np.empty(data_scaled.shape + (2,), dtype=np.int16)
    data_iq[..., 0] = np.round(data_scaled.real).astype(np.int16)
    data_iq[..., 1] = np.round(data_scaled.imag).astype(np.int16)
    print(f"[DEBUG] After int16 quantization: max={data_iq.max()}, min={data_iq.min()}, mean={data_iq.mean():.1f}")

    # Save binary data (interleaved IQ)
    save_binary_data(data_iq.flatten(), args.out)

    # Create dumps directory for all debug output
    dump_dir = os.path.join(os.path.dirname(args.out), "dumps")
    os.makedirs(dump_dir, exist_ok=True)

    adc_dump_indices = [0, 9, 19, 29, 39]
    data_iq_reshaped = data_iq.reshape(num_frames, chirps_per_frame, num_rx, adc_samples, 2)
    for idx in adc_dump_indices:
        if idx < num_frames:
            data_pow = np.mean(data_iq[idx] ** 2)
            print(f"[DEBUG] ADC export stats for frame {idx}: max={data_iq[idx].max()}, min={data_iq[idx].min()}, mean={data_iq[idx].mean()}, pow={data_pow:.1f}")
            np.save(os.path.join(dump_dir, f"adc_exported_frame{idx}.npy"), data_iq_reshaped[idx])
            print(f"Saved exported ADC data for frame {idx} to {os.path.join(dump_dir, f'adc_exported_frame{idx}.npy')}")

    # Optional: Show or save Range-Doppler images for visual comparison
    if args.show or args.save_plots or args.debug:
        print("Displaying/Saving synthetic gesture (RDI-domain, pre-inverse-FFT) and resulting ADC data...")
        N = 40  # Hardcode to match --frames 40 in run_test.bat for exact alignment
        N = min(N, num_frames)
        frame_indices = adc_dump_indices    # np.linspace(0, N - 1, 5, dtype=int)
        print(f"Plotting frame indices: {frame_indices}")

        # --- Plot 1: The actual RDI blob injected (pre-inverse-FFT) ---
        # Only plot the first 32 range bins for verification
        gesture_start = 5
        gesture_duration = max(10, num_frames // 4)
        gesture_end = min(num_frames, gesture_start + gesture_duration)
        patch_size_range = max(2, 32 // 16) # Reduced patch size for plotting
        patch_size_doppler = max(2, chirps_per_frame // 16) # Reduced patch size for plotting
        sigma_range = patch_size_range / 2.0
        sigma_doppler = patch_size_doppler / 2.0
        fig, axes = plt.subplots(3, 5, figsize=(15, 12), constrained_layout=True)
        fig.suptitle("Synthetic Gesture: Injected RDI, ADC Time Domain, and FFT-based RDI (Channel 0)")
        for i, frame_idx in enumerate(frame_indices):
            # Row 1: Injected RDI (cropped to 32x32)
            if gesture_start <= frame_idx < gesture_end:
                progress = (frame_idx - gesture_start) / gesture_duration
                if progress > 0.5:
                    progress = 1 - progress
                amplitude = 200.0 * progress + 50.0
                center_range = int(32 * (0.2 + 0.5 * progress))
                center_doppler = int(chirps_per_frame * (0.2 + 0.5 * progress))
                rdi = np.zeros((32, chirps_per_frame), dtype=np.float32)
                for dr in range(-patch_size_doppler//2, patch_size_doppler//2+1):
                    for rr in range(-patch_size_range//2, patch_size_range//2+1):
                        d_idx = center_doppler + dr
                        r_idx = center_range + rr
                        if 0 <= d_idx < chirps_per_frame and 0 <= r_idx < 32:
                            weight = np.exp(-0.5 * (dr/sigma_doppler)**2 - 0.5 * (rr/sigma_range)**2)
                            rdi[r_idx, d_idx] = amplitude * weight
            else:
                rdi = np.zeros((32, chirps_per_frame), dtype=np.float32)
            # Crop/pad to 32x32 for plotting
            rdi_32 = np.zeros((32, 32), dtype=np.float32)
            d_crop = min(32, chirps_per_frame)
            rdi_32[:, :d_crop] = rdi[:, :d_crop]
            im_rdi = axes[0, i].imshow(rdi_32.T, aspect='auto', cmap='viridis', origin='lower')
            axes[0, i].set_title(f"Frame {frame_idx}")
            axes[0, i].set_xlabel("Doppler Bin")
            axes[0, i].set_ylabel("Range Bin")
            if args.debug:
                # dump_dir is already created
                debug_rdi_path = os.path.join(dump_dir, f"rdi_gen_frame{frame_idx}.npy")
                np.save(debug_rdi_path, rdi_32)
                print(f"DEBUG: Saved generated RDI for frame {frame_idx} to {debug_rdi_path}")
        fig.colorbar(im_rdi, ax=axes[0, :], orientation='horizontal', pad=0.2)

        # Row 2: ADC time domain waveform (IQ)
        for i, frame_idx in enumerate(frame_indices):
            # Show I and Q as separate lines for the first chirp, channel 0
            adc_waveform = data_iq[frame_idx, 0, 0, :]
            axes[1, i].plot(adc_waveform[:, 0], label='I')
            axes[1, i].plot(adc_waveform[:, 1], label='Q')
            axes[1, i].set_title(f"Frame {frame_idx}")
            axes[1, i].set_xlabel("ADC Sample")
            axes[1, i].set_ylabel("Amplitude")
            axes[1, i].legend()

        # Row 3: FFT-based RDI (cropped to 32x32)
        for i, frame_idx in enumerate(frame_indices):
            # Reconstruct complex [chirps, adc] for channel 0
            frame_iq = data_iq[frame_idx, :, 0, :, :]  # shape: (chirps_per_frame, adc_samples, 2)
            frame_complex = frame_iq[..., 0].astype(np.float32) + 1j * frame_iq[..., 1].astype(np.float32)  # shape: (chirps_per_frame, adc_samples)
            window = np.hanning(adc_samples)
            range_fft = np.fft.fft(frame_complex * window, axis=-1)
            doppler_fft = np.fft.fft(range_fft, axis=0)
            rdi = np.abs(doppler_fft)
            rdi_log = 20 * np.log10(rdi + 1e-6)
            rdi_log_32 = np.zeros((32, 32), dtype=np.float32)
            r_crop = min(32, rdi_log.shape[0])
            d_crop = min(32, rdi_log.shape[1])
            rdi_log_32[:r_crop, :d_crop] = rdi_log[:r_crop, :d_crop]
            im2 = axes[2, i].imshow(rdi_log_32.T, aspect='auto', cmap='viridis', origin='lower')
            axes[2, i].set_title(f"Frame {frame_idx}")
            axes[2, i].set_xlabel("Doppler Bin")
            axes[2, i].set_ylabel("Range Bin")
        fig.colorbar(im2, ax=axes[2, :], orientation='horizontal', pad=0.2)

        if args.save_plots or args.debug:
            screenshots_dir = os.path.join(os.path.dirname(args.out), "screenshots")
            os.makedirs(screenshots_dir, exist_ok=True)
            plot_filename = os.path.join(screenshots_dir, "0_synthetic_gesture_rdi_comparison.png")
            plt.savefig(plot_filename)
            print(f"Saved synthetic gesture plot to {plot_filename}")
            plt.close(fig)
        elif args.show:
            plt.show()


    print("Test data generation completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
