"""
Data Handler Module
Handles data loading, validation, and preprocessing
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
import streamlit as st


class DataHandler:
    """Manages data loading, validation, and preprocessing for experiments"""
    
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []
        
    def load_data(self, uploaded_file) -> pd.DataFrame:
        """
        Load data from uploaded file (CSV or Parquet)
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        try:
            if uploaded_file.name.endswith('.csv'):
                self.data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.parquet'):
                self.data = pd.read_parquet(uploaded_file)
            else:
                raise ValueError("Unsupported file format. Please upload CSV or Parquet.")
            
            self._identify_column_types()
            return self.data
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            raise
    
    def _identify_column_types(self):
        """Identify numeric and categorical columns"""
        if self.data is not None:
            self.numeric_columns = self.data.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            self.categorical_columns = self.data.select_dtypes(
                include=['object', 'category', 'bool']
            ).columns.tolist()
    
    def get_summary_stats(self) -> pd.DataFrame:
        """Get summary statistics for the dataset"""
        if self.data is None:
            return pd.DataFrame()
        
        summary = pd.DataFrame({
            'Column': self.data.columns,
            'Type': self.data.dtypes.astype(str),
            'Non-Null Count': self.data.count(),
            'Null Count': self.data.isnull().sum(),
            'Unique Values': self.data.nunique()
        })
        
        return summary
    
    def validate_ab_test_columns(
        self, 
        group_col: str, 
        metric_col: str
    ) -> Tuple[bool, str]:
        """
        Validate columns for A/B testing
        
        Args:
            group_col: Column containing group assignments
            metric_col: Column containing the metric to analyze
            
        Returns:
            Tuple of (is_valid, message)
        """
        if self.data is None:
            return False, "No data loaded"
        
        if group_col not in self.data.columns:
            return False, f"Group column '{group_col}' not found"
        
        if metric_col not in self.data.columns:
            return False, f"Metric column '{metric_col}' not found"
        
        # Check for at least 2 groups
        unique_groups = self.data[group_col].nunique()
        if unique_groups < 2:
            return False, f"Need at least 2 groups, found {unique_groups}"
        
        # Check if metric is numeric
        if metric_col not in self.numeric_columns:
            return False, f"Metric column '{metric_col}' must be numeric"
        
        return True, "Validation passed"
    
    def prepare_ab_data(
        self, 
        group_col: str, 
        metric_col: str,
        filter_nulls: bool = True
    ) -> pd.DataFrame:
        """
        Prepare data for A/B testing
        
        Args:
            group_col: Group assignment column
            metric_col: Metric column
            filter_nulls: Whether to filter null values
            
        Returns:
            Cleaned DataFrame
        """
        df = self.data[[group_col, metric_col]].copy()
        
        if filter_nulls:
            df = df.dropna()
        
        return df
    
    def get_group_stats(
        self, 
        group_col: str, 
        metric_col: str
    ) -> pd.DataFrame:
        """
        Get descriptive statistics by group
        
        Args:
            group_col: Group assignment column
            metric_col: Metric column
            
        Returns:
            DataFrame with statistics by group
        """
        df = self.prepare_ab_data(group_col, metric_col)
        
        stats = df.groupby(group_col)[metric_col].agg([
            ('count', 'count'),
            ('mean', 'mean'),
            ('std', 'std'),
            ('median', 'median'),
            ('min', 'min'),
            ('max', 'max'),
            ('25%', lambda x: x.quantile(0.25)),
            ('75%', lambda x: x.quantile(0.75))
        ]).reset_index()
        
        return stats
