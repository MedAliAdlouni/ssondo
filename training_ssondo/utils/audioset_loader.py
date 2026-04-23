"""
Minimal AudioSet class to replace adasp_data_management dependency.
Only includes the functionality used in the project.
"""

import os
import pandas as pd
from typing import Optional

from training_ssondo import DATA


class AudioSet:
    """
    AudioSet class for loading metadata and downloading audio files.
    """
    def __init__(self, root_dir: Optional[str] = None, load_metadata: bool = True):
        """
        Initialize AudioSet loader.
        
        Parameters
        ----------
        root_dir : str, optional
            Root directory of AudioSet dataset. If None, defaults to
            $DATA/AudioSet (or training_ssondo/data/AudioSet if DATA is unset).
        load_metadata : bool, optional
            Whether to load metadata file immediately, by default True
        """
        if root_dir is None:
            root_dir = os.path.join(DATA, 'AudioSet')
        if not os.path.isdir(root_dir):
            raise ValueError(f"Root directory does not exist: {root_dir}")
        
        self.root_dir = root_dir
        self.metadata_file_path = os.path.join(root_dir, 'metadata.csv')
        self._pdf_metadata = None
        if load_metadata:
            self.load_metadata_file()
    
    @property
    def pdf_metadata(self):
        """Get metadata dataframe (alias for _pdf_metadata for compatibility)."""
        if self._pdf_metadata is None:
            self.load_metadata_file()
        return self._pdf_metadata
    
    def load_metadata_file(self):
        """
        Load the metadata CSV file.
        The metadata file should already exist (generated separately).
        This method loads it and converts relative paths to absolute paths.
        """
        if not os.path.exists(self.metadata_file_path):
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_file_path}\n"
                "Please ensure the metadata.csv file exists in the AudioSet root directory."
            )
        
        print(f"AudioSet: loading metadata file {self.metadata_file_path}")
        
        # Load metadata file
        self._pdf_metadata = pd.read_csv(
            self.metadata_file_path,
            index_col=0,
            keep_default_na=True,
            low_memory=False
        )
        
        # Convert relative paths to absolute paths for 'file_path' column
        # This matches the behavior of event.AudioSet from adasp-data-management
        if 'file_path' in self._pdf_metadata.columns:
            root_dir = os.path.dirname(self.metadata_file_path)
            self._pdf_metadata['file_path'] = self._pdf_metadata['file_path'].apply(
                lambda x: os.path.join(root_dir, x) if x and not os.path.isabs(x) else x
            )
    
    def get_abs_path(self, file_rel_path: str) -> str:
        """
        Convert relative path to absolute path based on root directory.
        
        Parameters
        ----------
        file_rel_path : str
            Relative file path
            
        Returns
        -------
        str
            Absolute file path
        """
        if file_rel_path.startswith('./'):
            file_rel_path = file_rel_path[2:]
        if file_rel_path.startswith('.'):
            file_rel_path = file_rel_path[1:]
        return os.path.join(self.root_dir, file_rel_path)
