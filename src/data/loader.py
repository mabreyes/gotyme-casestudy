"""Data loading and preprocessing module."""
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataLoader:
    """Handles data loading and preprocessing operations."""

    def __init__(self, data_path: str | Path):
        """Initialize the DataLoader.

        Args:
            data_path: Path to the data file
        """
        self.data_path = Path(data_path)
        self.categorical_features = [
            'Feature_jd_10', 'Feature_md_11', 'Feature_ed_12',
            'Feature_dd_13', 'Feature_hd_14', 'Feature_ld_15',
            'Feature_cd_16', 'Feature_md_17', 'Feature_dd_18',
            'Feature_pd_19'
        ]
        self.numerical_features = [
            'Feature_ae_0', 'Feature_dn_1', 'Feature_cn_2',
            'Feature_ps_3', 'Feature_ps_4', 'Feature_ee_5',
            'Feature_cx_6', 'Feature_cx_7', 'Feature_em_8',
            'Feature_nd_9'
        ]
        self.target = 'Response'
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def load_data(self) -> pd.DataFrame:
        """Load the dataset from file.

        Returns:
            pd.DataFrame: The loaded dataset
        """
        return pd.read_csv(self.data_path, delimiter='|')

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess the data for model training.

        Args:
            df: Input DataFrame

        Returns:
            Tuple containing features DataFrame and target Series
        """
        # Handle missing values
        df[self.numerical_features] = df[self.numerical_features].fillna(
            df[self.numerical_features].mean()
        )
        df[self.categorical_features] = df[self.categorical_features].fillna('Unknown')

        # Encode categorical variables
        for col in self.categorical_features:
            self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col])

        # Scale numerical features
        df[self.numerical_features] = self.scaler.fit_transform(
            df[self.numerical_features]
        )

        X = df[self.numerical_features + self.categorical_features]
        y = df[self.target]

        return X, y

    def get_train_test_split(
        self,
        test_size: float = 0.4,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load and split the data into training and test sets.

        Args:
            test_size: Proportion of the dataset to include in the test split
            random_state: Random state for reproducibility

        Returns:
            Tuple containing (X_train, X_test, y_train, y_test)
        """
        df = self.load_data()
        X, y = self.preprocess_data(df)
        
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

    def preprocess_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess new data using fitted transformers.

        Args:
            df: Input DataFrame

        Returns:
            Preprocessed DataFrame
        """
        # Handle missing values
        df[self.numerical_features] = df[self.numerical_features].fillna(
            df[self.numerical_features].mean()
        )
        df[self.categorical_features] = df[self.categorical_features].fillna('Unknown')

        # Encode categorical variables
        for col in self.categorical_features:
            df[col] = self.label_encoders[col].transform(df[col])

        # Scale numerical features
        df[self.numerical_features] = self.scaler.transform(
            df[self.numerical_features]
        )

        return df[self.numerical_features + self.categorical_features] 