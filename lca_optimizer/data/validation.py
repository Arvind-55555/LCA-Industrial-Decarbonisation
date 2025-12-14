"""
Data Validation Utilities
Validates and cleans LCA data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate and clean LCA data"""
    
    def __init__(self):
        """Initialize validator"""
        self.validation_errors = []
        logger.info("Data validator initialized")
    
    def validate_grid_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[bool, List[str]]:
        """
        Validate grid carbon intensity data.
        
        Args:
            df: DataFrame with grid data
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required columns
        required_cols = ['timestamp', 'carbon_intensity']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            errors.append(f"Missing required columns: {missing}")
        
        # Check data types
        if 'carbon_intensity' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['carbon_intensity']):
                errors.append("carbon_intensity must be numeric")
        
        # Check for negative values
        if 'carbon_intensity' in df.columns:
            negative = (df['carbon_intensity'] < 0).sum()
            if negative > 0:
                errors.append(f"Found {negative} negative carbon intensity values")
        
        # Check for unrealistic values
        if 'carbon_intensity' in df.columns:
            unrealistic = ((df['carbon_intensity'] > 2000) | 
                          (df['carbon_intensity'] < 0)).sum()
            if unrealistic > 0:
                errors.append(f"Found {unrealistic} unrealistic values (>2000 or <0)")
        
        # Check for missing values
        if 'carbon_intensity' in df.columns:
            missing = df['carbon_intensity'].isna().sum()
            if missing > 0:
                errors.append(f"Found {missing} missing carbon intensity values")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def clean_grid_data(
        self,
        df: pd.DataFrame,
        remove_negative: bool = True,
        remove_outliers: bool = True,
        fill_missing: str = "forward"
    ) -> pd.DataFrame:
        """
        Clean grid carbon intensity data.
        
        Args:
            df: DataFrame to clean
            remove_negative: Remove negative values
            remove_outliers: Remove outliers (>2000 g CO2eq/kWh)
            fill_missing: Method to fill missing values ("forward", "backward", "interpolate")
        
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        if 'carbon_intensity' in df_clean.columns:
            # Remove negative values
            if remove_negative:
                df_clean = df_clean[df_clean['carbon_intensity'] >= 0]
            
            # Remove outliers
            if remove_outliers:
                df_clean = df_clean[df_clean['carbon_intensity'] <= 2000]
            
            # Fill missing values
            if fill_missing == "forward":
                df_clean['carbon_intensity'] = df_clean['carbon_intensity'].fillna(method='ffill')
            elif fill_missing == "backward":
                df_clean['carbon_intensity'] = df_clean['carbon_intensity'].fillna(method='bfill')
            elif fill_missing == "interpolate":
                df_clean['carbon_intensity'] = df_clean['carbon_intensity'].interpolate()
        
        logger.info(f"Cleaned data: {len(df)} -> {len(df_clean)} records")
        return df_clean
    
    def validate_lca_result(
        self,
        result: Dict[str, float]
    ) -> Tuple[bool, List[str]]:
        """
        Validate LCA calculation result.
        
        Args:
            result: LCA result dictionary
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Check for required keys
        required_keys = ['total_emissions']
        missing = [key for key in required_keys if key not in result]
        if missing:
            errors.append(f"Missing required keys: {missing}")
        
        # Check for negative emissions
        if 'total_emissions' in result:
            if result['total_emissions'] < 0:
                errors.append("Total emissions cannot be negative")
        
        # Check breakdown sums to total
        if 'breakdown' in result and 'total_emissions' in result:
            breakdown_sum = sum(result['breakdown'].values())
            total = result['total_emissions']
            if abs(breakdown_sum - total) > 0.01:  # Allow small floating point errors
                errors.append(f"Breakdown sum ({breakdown_sum}) != total ({total})")
        
        is_valid = len(errors) == 0
        return is_valid, errors

