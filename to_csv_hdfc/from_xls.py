'''HDFC Bank specific implementation for converting XLS/XLSX bank statements to CSV format.'''
from typing import Optional, cast

import pandas as pd

from shared.StatementConverter import BaseBankStatementConverter


class HDFCBankStatementConverter(BaseBankStatementConverter):
    """
    HDFC Bank specific implementation for converting XLS/XLSX bank statements to CSV format.

    This class handles reading Excel files, cleaning HDFC bank statement data,
    and saving the processed data to CSV format.
    """

    def __init__(self):
        """Initialize the HDFC Bank Statement Converter with required columns."""
        required_columns = [
            "Date",
            "Narration",
            "Chq./Ref.No.",
            "Value Dt",
            "Withdrawal Amt.",
            "Deposit Amt.",
            "Closing Balance",
        ]
        super().__init__(required_columns)

    def clean_bank_statement(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Cleans an HDFC bank statement DataFrame and extracts relevant columns.

        This method filters the DataFrame to only include the transaction data
        and selects the required columns for HDFC bank statement processing.

        Args:
            df (Optional[pd.DataFrame]): The raw bank statement DataFrame.
                                       If None, uses the stored raw_data.

        Returns:
            pd.DataFrame: A cleaned DataFrame with only the relevant transaction data.

        Raises:
            ValueError: If no data is available or required columns are missing.
        """
        # Use provided DataFrame or stored raw data
        if df is None:
            if self.raw_data is None:
                raise ValueError("No data available. Please read an Excel file first.")
            df = self.raw_data.copy()

        # Find the row where actual transaction data starts
        header_row = self._find_header_row(df)

        if header_row is None:
            raise ValueError("Could not find the transaction data header row")

        # Set the column names from the header row
        df.columns = df.iloc[header_row]

        # Get data starting from the row after the header
        # HDFC Bank statements have a row of '*' after the header row
        data_start = header_row + 2

        # Find where the transaction data ends
        data_end = self._find_data_end(df, data_start)

        # Extract the transaction data
        cleaned_df = df.iloc[data_start:data_end].copy()

        # Validate and select required columns
        cleaned_df = self._validate_and_select_columns(cleaned_df)

        # Clean up the data
        cleaned_df = self._cleanup_data(cleaned_df)

        # Store cleaned data
        self.cleaned_data = cleaned_df

        return cleaned_df

    def _find_header_row(self, df: pd.DataFrame) -> Optional[int]:
        """
        Find the row containing the column headers.

        Args:
            df (pd.DataFrame): The DataFrame to search.

        Returns:
            Optional[int]: The index of the header row, or None if not found.
        """
        for idx, row in df.iterrows():
            if "Date" in str(row.iloc[0]) and "Narration" in str(row.iloc[1]):
                # Cast index to int since DataFrame index is typically int but typed as Hashable
                return cast(int, idx)
        return None

    def _find_data_end(self, df: pd.DataFrame, data_start: int) -> int:
        """
        Find where the transaction data ends.

        Args:
            df (pd.DataFrame): The DataFrame to search.
            data_start (int): The starting index of transaction data.

        Returns:
            int: The index where transaction data ends.
        """
        for idx in range(data_start, len(df)):
            if (
                pd.isna(df.iloc[idx, 0])
                or str(df.iloc[idx, 0]).startswith("*")
                or "STATEMENT SUMMARY" in str(df.iloc[idx, 0])
            ):
                return idx
        return len(df)

    def _cleanup_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean up the DataFrame by removing invalid rows and resetting index.

        Args:
            df (pd.DataFrame): The DataFrame to clean up.

        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """
        # Remove rows where Date is NaN or empty
        df = df.dropna(subset=["Date"])
        df = df[df["Date"].astype(str).str.strip() != ""]

        # Reset index
        df.reset_index(drop=True, inplace=True)

        return df


# Example usage:
if __name__ == "__main__":
    # Create an instance of the converter
    converter = HDFCBankStatementConverter()

    # Method 1: Complete conversion in one step
    cleaned_data = converter.convert("sample_statements/HDFC/sample_1.xls", "output_statement.csv")

    # Method 2: Step-by-step conversion
    # converter.read_excel("to_csv_hdfc/sample_1.xls")
    # cleaned_data = converter.clean_bank_statement()
    # converter.save_to_csv("output_statement.csv")

    # Get summary of current state
    # print(converter.get_summary())
