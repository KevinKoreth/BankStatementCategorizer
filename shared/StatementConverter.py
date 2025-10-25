from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd


class BaseBankStatementConverter(ABC):
    """
    Base class for bank statement converters with common functionality.

    This abstract class provides common methods for reading Excel files,
    validating columns, and saving to CSV format.
    """

    def __init__(self, required_columns: List[str]):
        """
        Initialize the base converter.

        Args:
            required_columns (List[str]): List of required column names for the bank statement.
        """
        self.raw_data: Optional[pd.DataFrame] = None
        self.cleaned_data: Optional[pd.DataFrame] = None
        self.required_columns = required_columns

    def read_excel(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Reads an Excel file (.xls or .xlsx) and stores it as raw data.

        Args:
            file_path (Union[str, Path]): The path to the Excel file.

        Returns:
            pd.DataFrame: The DataFrame containing the data from the Excel file.

        Raises:
            FileNotFoundError: If the specified file doesn't exist.
            ValueError: If the file cannot be read as an Excel file.
        """
        try:
            self.raw_data = pd.read_excel(file_path)
            return self.raw_data
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Excel file not found: {file_path}") from exc
        except Exception as e:
            raise ValueError(f"Failed to read Excel file: {e}") from e

    def save_to_csv(
        self, file_path: Union[str, Path], df: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Saves a DataFrame to a CSV file.

        Args:
            file_path (Union[str, Path]): The path where the CSV file will be saved.
            df (Optional[pd.DataFrame]): The DataFrame to save.
                                       If None, uses the stored cleaned_data.

        Raises:
            ValueError: If no data is available to save.
        """
        # Use provided DataFrame or stored cleaned data
        if df is None:
            if self.cleaned_data is None:
                raise ValueError(
                    "No cleaned data available. Please clean the data first."
                )
            df = self.cleaned_data

        try:
            df.to_csv(file_path, index=False)
        except Exception as e:
            raise ValueError(f"Failed to save CSV file: {e}") from e

    def _validate_and_select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate that required columns exist and select them.

        Args:
            df (pd.DataFrame): The DataFrame to validate.

        Returns:
            pd.DataFrame: DataFrame with only the required columns.

        Raises:
            ValueError: If required columns are missing.
        """
        # Check if all required columns exist
        available_columns = [col for col in self.required_columns if col in df.columns]

        if len(available_columns) != len(self.required_columns):
            missing_columns = set(self.required_columns) - set(available_columns)
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Select only the required columns
        return df[self.required_columns].copy()

    def get_summary(self) -> dict:
        """
        Get a summary of the current data state.

        Returns:
            dict: Summary information about loaded and cleaned data.
        """
        summary = {
            "raw_data_loaded": self.raw_data is not None,
            "cleaned_data_available": self.cleaned_data is not None,
            "raw_data_shape": (
                self.raw_data.shape if self.raw_data is not None else None
            ),
            "cleaned_data_shape": (
                self.cleaned_data.shape if self.cleaned_data is not None else None
            ),
            "required_columns": self.required_columns,
        }
        return summary

    @abstractmethod
    def clean_bank_statement(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Abstract method for cleaning bank statement data.
        Must be implemented by subclasses for bank-specific logic.

        Args:
            df (Optional[pd.DataFrame]): The raw bank statement DataFrame.

        Returns:
            pd.DataFrame: A cleaned DataFrame with relevant transaction data.
        """

    def convert(
        self, input_path: Union[str, Path], output_path: Union[str, Path]
    ) -> pd.DataFrame:
        """
        Complete conversion process: read Excel file, clean data, and save to CSV.

        Args:
            input_path (Union[str, Path]): Path to the input Excel file.
            output_path (Union[str, Path]): Path where the CSV file will be saved.

        Returns:
            pd.DataFrame: The cleaned DataFrame that was saved.
        """
        # Read the Excel file
        self.read_excel(input_path)

        # Clean the data
        cleaned_df = self.clean_bank_statement()

        # Save to CSV
        self.save_to_csv(output_path)

        return cleaned_df