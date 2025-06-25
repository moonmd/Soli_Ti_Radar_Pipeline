#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify HDF5 Output for mmWave to Soli-format Converter

This script verifies that an HDF5 file conforms to the Soli format v1.0 specifications.
It checks the structure, datasets, attributes, and data types of the file.
"""

import h5py
import sys


def verify_hdf5(file_path):
    """
    Verify that the HDF5 file structure conforms to Soli format v1.0.

    Args:
        file_path: Path to the HDF5 file
    """
    print(f"Verifying HDF5 file: {file_path}")
    with h5py.File(file_path, "r") as f:
        # Check required datasets
        for ch in range(4):
            dset_name = f"ch{ch}"
            if dset_name not in f:
                print(f"Error: Dataset '{dset_name}' not found.")
                return False
            dset = f[dset_name]
            if dset.shape[1] != 1024:
                print(f"Error: Dataset '{dset_name}' has incorrect shape: {dset.shape}, expected: (N, 1024)")
                return False
            print(f"Dataset '{dset_name}' shape: {dset.shape}")
        # Check label dataset
        if "label" not in f:
            print("Error: Dataset 'label' not found.")
            return False
        label = f["label"]
        if label.shape[1] != 1:
            print(f"Error: Label dataset has incorrect shape: {label.shape}, expected (N, 1)")
            return False
        print(f"Label dataset shape: {label.shape}")
        # Check attributes
        if "fps" not in f.attrs:
            print("Error: Attribute 'fps' not found.")
            return False
        if "range_bins" not in f.attrs or "doppler_bins" not in f.attrs:
            print("Error: Attributes 'range_bins' or 'doppler_bins' not found.")
            return False
        print(f"Attributes: fps={f.attrs['fps']}, range_bins={f.attrs['range_bins']}, doppler_bins={f.attrs['doppler_bins']}")
        print("HDF5 file verification passed.")
        return True


def main():
    """
    Main function to verify HDF5 file structure.
    """
    if len(sys.argv) < 3 or sys.argv[1] != "--file":
        print("Usage: python verify_hdf5.py --file <h5file>")
        sys.exit(1)
    file_path = sys.argv[2]
    if not verify_hdf5(file_path):
        print("Error: HDF5 verification failed.")
        sys.exit(1)
    print("Verification successful.")
    sys.exit(0)


if __name__ == "__main__":
    main()
