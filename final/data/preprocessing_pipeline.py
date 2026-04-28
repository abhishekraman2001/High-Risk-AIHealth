"""
PET Image Preprocessing Pipeline for UDPET Dataset
"""

import os
import numpy as np
import h5py
import pydicom
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import zoom
from skimage.util import crop
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class PETPreprocessor:
    """Preprocesses PET DICOM images and saves as HDF5 format."""
    
    # Dose reduction factor mapping from folder names to standardized keys
    DOSE_FACTORS = {
        'D2': '1_2',
        'D4': '1_4', 
        'D5': '1_5',
        'D10': '1_10',
        'D20': '1_20',
        'D50': '1_50',
        'D100': '1_100',
        'NORMAL': 'full'
    }

    TARGET_SIZE = (640, 320)  
    
    def __init__(self, target_size: Optional[Tuple[int, int]] = None, 
                 normalize: bool = True, 
                 normalization_range: Tuple[float, float] = (-1, 1)):
        """
        Initialize preprocessor.
        
        Args:
            target_size: Target output size (H, W). If None, uses largest dimension.
            normalize: Whether to apply per-slice min-max normalization
            normalization_range: Range for normalization (default: [-1, 1])
        """
        self.target_size = target_size
        self.normalize = normalize
        self.norm_range = normalization_range
        self.collected_dimensions = []
    
    def load_dicom_series(self, folder_path: str) -> np.ndarray:
        """
        Load a series of DICOM files from a folder and stack them.
        
        Args:
            folder_path: Path to folder containing DICOM files
            
        Returns:
            3D numpy array of shape (num_slices, height, width)
        """
        dicom_files = sorted(Path(folder_path).glob('*.dcm'))
        
        if not dicom_files:
            raise FileNotFoundError(f"No DICOM files found in {folder_path}")
        
        slices = []
        for dcm_file in dicom_files:
            try:
                dcm = pydicom.dcmread(str(dcm_file))
                pixel_array = dcm.pixel_array.astype(np.float32)
                slices.append(pixel_array)
            except Exception as e:
                print(f"Warning: Failed to read {dcm_file}: {e}")
                continue
        
        if not slices:
            raise ValueError(f"Could not load any DICOM files from {folder_path}")

        volume = np.stack(slices, axis=0)
        return volume
    
    def normalize_slice(self, slice_2d: np.ndarray) -> np.ndarray:
        """
        Apply per-slice min-max normalization to [-1, 1] or specified range.
        
        Args:
            slice_2d: 2D array representing a single slice
            
        Returns:
            Normalized 2D array
        """
        if not self.normalize:
            return slice_2d
        
        slice_min = slice_2d.min()
        slice_max = slice_2d.max()
        
        if slice_max == slice_min:
            return np.zeros_like(slice_2d)
        
        # Normalize to [0, 1]
        normalized = (slice_2d - slice_min) / (slice_max - slice_min)
        
        # Scale to target range
        range_min, range_max = self.norm_range
        scaled = normalized * (range_max - range_min) + range_min
        
        return scaled.astype(np.float32)
    
    def spatial_standardization(self, slice_2d: np.ndarray, 
                                target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Apply spatial standardization: center crop or zero-pad to target size.
        
        Args:
            slice_2d: 2D input image
            target_size: Target (height, width). If None, uses self.target_size
            
        Returns:
            Standardized 2D image
        """
        if target_size is None:
            target_size = self.target_size
        
        h, w = slice_2d.shape
        target_h, target_w = target_size
        
        # Center crop if image is larger than target
        if h > target_h or w > target_w:
            crop_h = min(h, target_h)
            crop_w = min(w, target_w)
            
            # Calculate crop margins
            start_h = (h - crop_h) // 2
            start_w = (w - crop_w) // 2
            
            slice_2d = slice_2d[start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        # Zero-pad if image is smaller than target
        h, w = slice_2d.shape
        if h < target_h or w < target_w:
            pad_h = max(0, target_h - h)
            pad_w = max(0, target_w - w)
            
            # Center padding
            pad_h_before = pad_h // 2
            pad_h_after = pad_h - pad_h_before
            pad_w_before = pad_w // 2
            pad_w_after = pad_w - pad_w_before
            
            slice_2d = np.pad(slice_2d, 
                            ((pad_h_before, pad_h_after), 
                             (pad_w_before, pad_w_after)),
                            mode='constant', constant_values=0)
        
        return slice_2d.astype(np.float32)
    
    def preprocess_volume(self, volume_3d: np.ndarray) -> np.ndarray:
        """
        Preprocess a 3D volume: normalize and spatially standardize each slice.
        
        Args:
            volume_3d: 3D array of shape (num_slices, height, width)
            
        Returns:
            Preprocessed 3D array
        """
        num_slices = volume_3d.shape[0]
        
        # Determine target size from first slice if not set
        if self.target_size is None:
            h, w = volume_3d[0].shape
            # Use maximum dimension or set to common size
            self.target_size = (max(h, w), max(h, w))
        
        processed_volume = []
        
        for i in range(num_slices):
            slice_2d = volume_3d[i]
            
            # Normalize per-slice
            normalized = self.normalize_slice(slice_2d)
            
            # Spatial standardization
            standardized = self.spatial_standardization(normalized)
            
            processed_volume.append(standardized)
        
        # Stack back to 3D
        processed_3d = np.stack(processed_volume, axis=0)
        return processed_3d
    
    def extract_dose_folder(self, folder_path: str) -> Optional[str]:
        """
        Extract dose folder name from parent path.
        Looks for folders like "2.886 x 600 WB D10", "2.886 x 600 WB NORMAL", etc.
        
        Args:
            folder_path: Path to DICOM folder
            
        Returns:
            Dose key or None if not recognized
        """
        folder_name = os.path.basename(folder_path.rstrip('/'))
        
        for dose_key, mapped_key in self.DOSE_FACTORS.items():
            if dose_key in folder_name:
                return mapped_key
        
        return None
    
    def process_patient_folder(self, patient_folder_path: str, 
                              output_h5_path: str) -> bool:
        """
        Process all dose levels for a single patient and save to HDF5.
        
        Args:
            patient_folder_path: Path to patient folder containing dose subfolders
            output_h5_path: Output HDF5 file path
            
        Returns:
            True if successful, False otherwise
        """
        hdf5_data = {}
        
        # Iterate through all dose folders
        dose_folders = sorted(Path(patient_folder_path).glob('*/'))
        
        print(f"\nProcessing patient: {os.path.basename(patient_folder_path)}")
        print(f"Found {len(dose_folders)} dose folders")
        
        for dose_folder in dose_folders:
            dose_key = self.extract_dose_folder(str(dose_folder))
            
            if dose_key is None:
                print(f"  Skipping {dose_folder.name} (unrecognized dose)")
                continue
            
            try:
                # Load DICOM series
                print(f"  Loading {dose_key}...", end=" ", flush=True)
                volume = self.load_dicom_series(str(dose_folder))
                print(f"Loaded shape {volume.shape}", end=" ", flush=True)
                
                # Preprocess
                processed = self.preprocess_volume(volume)
                print(f"Processed shape {processed.shape}")
                
                hdf5_data[dose_key] = processed
                
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        if not hdf5_data:
            print(f"No valid dose data found for {patient_folder_path}")
            return False
        
        # Save to HDF5
        try:
            with h5py.File(output_h5_path, 'w') as h5f:
                for key, data in hdf5_data.items():
                    h5f.create_dataset(key, data=data, compression='gzip', 
                                      compression_opts=4)
            
            print(f"Saved to {output_h5_path}")
            return True
            
        except Exception as e:
            print(f"Error saving HDF5: {e}")
            return False
    
    def process_all_patients(self, raw_data_folder: str, 
                            output_folder: str) -> None:
        """
        Process all patients in the raw data folder.
        
        Args:
            raw_data_folder: Path to folder containing patient folders
            output_folder: Output folder for preprocessed HDF5 files
        """
        os.makedirs(output_folder, exist_ok=True)
        
        patient_folders = sorted(Path(raw_data_folder).glob('*/'))
        
        print(f"Found {len(patient_folders)} patient folders")
        print(f"Processing...")
        
        success_count = 0
        
        for patient_folder in tqdm(patient_folders, desc="Processing patients"):
            # Generate output filename based on patient folder
            patient_name = patient_folder.name
            output_filename = f"{patient_name}.h5"
            output_path = os.path.join(output_folder, output_filename)
            
            success = self.process_patient_folder(str(patient_folder), output_path)
            if success:
                success_count += 1
        
        print(f"\n{'='*50}")
        print(f"Preprocessing complete!")
        print(f"Successfully processed: {success_count}/{len(patient_folders)} patients")
        print(f"Output folder: {output_folder}")
