"""
Purged and Embargoed Time-Series Cross-Validation for Sliding-Window Panel Data.

Inspired by Marcos López de Prado (Advances in Financial Machine Learning, 2018).
Customized for forward risk forecasting to eliminate:
1. Target overlap leakage (where test forecast origins occur before training target windows conclude).
2. Contemporaneous cross-sectional contamination (where portfolios on the same calendar date are split across folds).
3. Autoregressive persistence via configurable embargo buffers.
"""

import pandas as pd
import numpy as np
import logging
from typing import Generator, Tuple, Optional

logger = logging.getLogger(__name__)

class PurgedGroupTimeSeriesSplit:
    """
    Time-Series Cross-Validation with boundary purging and embargo for overlapping window datasets.
    
    Purging Rule (Forward Targets):
    For any fold, let max_train_target_end be the latest actual date on which any training sample's
    forward target window concluded (max Target_End in train).
    To eliminate overlapping target evaluation and information leakage:
    Any test sample whose forecast origin (Date_t) occurs on or before:
        max_train_target_end + embargo_days
    is purged from the test set.
    
    Attributes:
        n_splits (int): Number of forward-chaining splits.
        embargo_days (int): Number of calendar buffer days after max training target date to account for
                            autoregressive persistence (e.g., rolling volatility clustering).
    """
    def __init__(self, n_splits: int = 5, embargo_days: int = 0):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        self.n_splits = n_splits
        self.embargo_days = embargo_days

    def split(
        self, 
        df: pd.DataFrame, 
        y: Optional[pd.DataFrame] = None, 
        groups: Optional[pd.Series] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generates integer index positions for train and test splits.
        
        Args:
            df (pd.DataFrame): Sorted panel dataset containing either:
                               - 'Date_t' and 'Target_End' (forward forecasting schema), or
                               - 'Window_Start' and 'Window_End' (legacy window schema).
            y: Optional target vector/matrix for scikit-learn API compatibility.
            groups: Optional group identifier for scikit-learn API compatibility.
            
        Yields:
            Tuple[np.ndarray, np.ndarray]: (train_indices, test_indices)
        """
        if "Date_t" in df.columns and "Target_End" in df.columns:
            date_t = pd.to_datetime(df["Date_t"])
            target_end = pd.to_datetime(df["Target_End"])
            
            # Group by unique chronological forecast origin dates
            unique_dates = date_t.drop_duplicates().sort_values().reset_index(drop=True)
            n_dates = len(unique_dates)
            
            if n_dates < self.n_splits + 1:
                raise ValueError(f"Not enough unique dates ({n_dates}) for {self.n_splits} splits.")
                
            chunk_size = n_dates // (self.n_splits + 1)
            
            for fold in range(self.n_splits):
                train_end_idx = (fold + 1) * chunk_size
                train_cutoff_date = unique_dates.iloc[train_end_idx]
                
                if fold == self.n_splits - 1:
                    test_cutoff_date = unique_dates.iloc[-1]
                else:
                    test_cutoff_date = unique_dates.iloc[(fold + 2) * chunk_size]
                    
                train_mask = date_t <= train_cutoff_date
                train_indices = np.where(train_mask)[0]
                
                if len(train_indices) == 0:
                    continue
                    
                # Exact forward target end date across all training samples
                max_train_target_end = target_end.iloc[train_indices].max()
                
                # PURGE & EMBARGO:
                # Discard any test observation whose Date_t occurs while training targets are still unfolding
                # or within embargo_days of the latest training target end.
                embargo_boundary = max_train_target_end + pd.Timedelta(days=self.embargo_days)
                
                test_mask = (date_t > train_cutoff_date) & \
                            (date_t <= test_cutoff_date) & \
                            (date_t > embargo_boundary)
                            
                test_indices = np.where(test_mask)[0]
                
                yield train_indices, test_indices

        elif "Window_Start" in df.columns and "Window_End" in df.columns:
            # Legacy branch for backward compatibility
            window_start = pd.to_datetime(df["Window_Start"])
            window_end = pd.to_datetime(df["Window_End"])
            
            unique_dates = window_end.drop_duplicates().sort_values().reset_index(drop=True)
            n_dates = len(unique_dates)
            
            if n_dates < self.n_splits + 1:
                raise ValueError(f"Not enough unique dates ({n_dates}) for {self.n_splits} splits.")
                
            chunk_size = n_dates // (self.n_splits + 1)
            
            for fold in range(self.n_splits):
                train_end_idx = (fold + 1) * chunk_size
                train_cutoff_date = unique_dates.iloc[train_end_idx]
                
                if fold == self.n_splits - 1:
                    test_cutoff_date = unique_dates.iloc[-1]
                else:
                    test_cutoff_date = unique_dates.iloc[(fold + 2) * chunk_size]
                    
                train_mask = window_end <= train_cutoff_date
                train_indices = np.where(train_mask)[0]
                
                embargo_boundary = train_cutoff_date + pd.Timedelta(days=self.embargo_days)
                
                test_mask = (window_end > train_cutoff_date) & \
                            (window_end <= test_cutoff_date) & \
                            (window_start > embargo_boundary)
                            
                test_indices = np.where(test_mask)[0]
                
                yield train_indices, test_indices
        else:
            raise ValueError(
                "Dataset must contain ('Date_t' and 'Target_End') or ('Window_Start' and 'Window_End') columns."
            )
