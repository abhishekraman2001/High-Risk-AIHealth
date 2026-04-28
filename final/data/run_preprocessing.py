#!/usr/bin/env python3
"""
QUICK START GUIDE - PET Preprocessing Pipeline

Run this script to preprocess all patients from raw DICOM to HDF5 format.
"""

from data.preprocessing_pipeline import PETPreprocessor
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input and output paths
RAW_DATA_FOLDER = "uExplorerPART16 (1)"           # Raw DICOM data
OUTPUT_FOLDER = "train_sample"                     # Output HDF5 files

# Preprocessing parameters
TARGET_SIZE = (640, 320)                           # Spatial dimensions
NORMALIZE = True                                   # Apply per-slice normalization
NORMALIZATION_RANGE = (-1, 1)                     # Normalize to [-1, 1]

# ============================================================================
# PREPROCESSING
# ============================================================================

def main():
    """Main preprocessing script."""
    
    print("\n" + "="*70)
    print("PET IMAGE PREPROCESSING PIPELINE")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Raw data folder: {RAW_DATA_FOLDER}")
    print(f"  Output folder: {OUTPUT_FOLDER}")
    print(f"  Target size: {TARGET_SIZE}")
    print(f"  Normalization: {NORMALIZE} to range {NORMALIZATION_RANGE}")
    print("\n" + "="*70)
    
    # Create preprocessor with specified configuration
    preprocessor = PETPreprocessor(
        target_size=TARGET_SIZE,
        normalize=NORMALIZE,
        normalization_range=NORMALIZATION_RANGE
    )
    
    # Process all patients
    preprocessor.process_all_patients(
        raw_data_folder=RAW_DATA_FOLDER,
        output_folder=OUTPUT_FOLDER
    )
    
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"\nOutput files saved to: {os.path.abspath(OUTPUT_FOLDER)}")
    print("\nNext steps:")
    print("  1. Verify output HDF5 files")
    print("  2. Split data into train/validation/test sets")
    print("  3. Train deep learning model on preprocessed data")
    print()


if __name__ == "__main__":
    main()
