"""
Bank Statement PDF Parser

This module extracts transaction data from bank statement PDFs and converts them to CSV format.
It handles multi-page statements with continuation rows in narration fields.
"""

import os
import sys
from typing import List, Optional
from uuid import uuid4

import pandas as pd
import tabula.io as tabula

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.custom_logger import get_logger

logger = get_logger(__name__)

# Constants
uuid = uuid4().hex[:8]
DEFAULT_PDF_PATH = "sample_statements/HDFC/pdf_sample_1.pdf"
OUTPUT_CSV_PATH = f"bank_statement_transactions_{uuid}.csv"
EXPECTED_HEADERS = ["Date", "Narration", "Value Dt"]
HEADER_PATTERNS = ["Date", "date"]


class BankStatementParser:
    """Handles parsing of bank statement PDFs into structured CSV format."""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.expected_columns = None

    def parse(self) -> Optional[pd.DataFrame]:
        """Parse the PDF and return a DataFrame of transactions."""
        tables = self._extract_tables_from_pdf()

        if not tables:
            logger.warning("No tables found in the PDF")
            return None

        self._print_table_summary(tables)
        transaction_tables = self._collect_transaction_tables(tables)

        if not transaction_tables:
            logger.warning("No transaction tables found!")
            return self._try_fallback(tables)

        return self._combine_and_clean_tables(transaction_tables)

    def _extract_tables_from_pdf(self) -> List:
        """Extract all tables from the PDF."""
        return tabula.read_pdf(
            self.pdf_path, stream=True, pages="all", multiple_tables=True
        )

    def _print_table_summary(self, tables: List) -> None:
        """Print diagnostic information about extracted tables."""
        logger.info(f"Found {len(tables)} tables in the PDF")

        for i, table in enumerate(tables):
            if isinstance(table, pd.DataFrame):
                # logger.info(f"Table {i}: {len(table)} rows, columns: {list(table.columns)}")
                pass
            else:
                logger.warning(f"Table {i}: Not a DataFrame - {type(table)}")

    def _collect_transaction_tables(self, tables: List) -> List[pd.DataFrame]:
        """Collect and process all transaction tables from the PDF."""
        transaction_tables = []

        for i, table in enumerate(tables):
            if not isinstance(table, pd.DataFrame) or table.empty:
                continue

            if i == 0:
                processed_table = self._process_first_table(table, i)
            else:
                processed_table = self._process_continuation_table(table, i)

            if processed_table is not None:
                transaction_tables.append(processed_table)

        return transaction_tables

    def _process_first_table(
        self, table: pd.DataFrame, index: int
    ) -> Optional[pd.DataFrame]:
        """Process the first table which should contain headers."""
        if not any(col in table.columns for col in EXPECTED_HEADERS):
            logger.warning(
                f"First table doesn't have expected headers - columns: {list(table.columns)}"
            )
            return None

        self.expected_columns = table.columns.tolist()
        logger.info(
            f"Adding table {index} as first transaction table with headers ({len(table)} rows)"
        )
        logger.info(f"Expected columns: {self.expected_columns}")
        return table

    def _process_continuation_table(
        self, table: pd.DataFrame, index: int
    ) -> Optional[pd.DataFrame]:
        """Process continuation tables that don't have headers."""
        if not self.expected_columns or len(table.columns) != len(
            self.expected_columns
        ):
            logger.warning(
                f"Table {index} has different structure - columns: {len(table.columns)} "
                f"vs expected {len(self.expected_columns) if self.expected_columns else 'N/A'}"
            )
            return None

        table.columns = self.expected_columns
        table = self._remove_header_rows(table)

        if table.empty:
            logger.warning(f"Table {index} is empty after filtering header rows")
            return None

        logger.info(f"Adding table {index} as continuation table ({len(table)} rows)")
        return table

    def _remove_header_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows that contain header text."""
        pattern = "|".join(HEADER_PATTERNS)
        return df[~df.iloc[:, 0].astype(str).str.contains(pattern, na=False)]

    def _combine_and_clean_tables(self, tables: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple tables and clean the result."""
        combined_df = pd.concat(tables, ignore_index=True)
        combined_df = combined_df.dropna(how="all")
        combined_df = merge_continuation_rows(combined_df)
        return combined_df

    def _try_fallback(self, tables: List) -> Optional[pd.DataFrame]:
        """Try to use the first table as a fallback if no transaction tables found."""
        if not tables or len(tables) == 0:
            logger.warning("No tables available for fallback")
            return None

        first_table = tables[0]
        if not isinstance(first_table, pd.DataFrame):
            logger.warning("First table is not a DataFrame")
            return None

        logger.info("Using first table as fallback")
        return first_table

    def save_to_csv(self, df: pd.DataFrame, output_path: str = OUTPUT_CSV_PATH) -> None:
        """Save the DataFrame to a CSV file."""
        df.to_csv(output_path, index=False)
        logger.info(f"Bank statement converted to CSV: {output_path}")
        logger.info(f"Total transactions extracted: {len(df)}")


def merge_continuation_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge continuation rows where narration/cheq_ref extends across multiple rows.

    Continuation rows have only the narration and/or cheq_ref columns filled
    and rest are empty/NaN. This function can handle cases where content
    spans more than 3 rows.

    Args:
        df: DataFrame containing bank transactions

    Returns:
        DataFrame with continuation rows merged
    """
    if df.empty:
        return df

    df = df.copy()
    narration_col, cheq_ref_col = _get_column_references(df)
    other_cols = _get_other_columns(df, narration_col, cheq_ref_col)

    rows_to_remove = []
    i = 0

    while i < len(df):
        if i in rows_to_remove:
            i += 1
            continue

        continuation_count = _process_continuation_rows(
            df, i, rows_to_remove, narration_col, cheq_ref_col, other_cols
        )

        if continuation_count > 0:
            # _print_merge_summary(df, i, continuation_count, narration_col, cheq_ref_col)
            pass

        i += 1

    return _remove_marked_rows(df, rows_to_remove)


def _get_column_references(df: pd.DataFrame) -> tuple:
    """Get the narration and cheq_ref column references."""
    narration_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    cheq_ref_col = df.columns[2] if len(df.columns) > 2 else None
    return narration_col, cheq_ref_col


def _get_other_columns(df: pd.DataFrame, narration_col, cheq_ref_col) -> List:
    """Get list of columns excluding narration and cheq_ref."""
    return [col for col in df.columns if col != narration_col and col != cheq_ref_col]


def _process_continuation_rows(
    df: pd.DataFrame,
    base_row_idx: int,
    rows_to_remove: List[int],
    narration_col,
    cheq_ref_col,
    other_cols: List,
) -> int:
    """Process all continuation rows for a given base row."""
    continuation_count = 0
    j = base_row_idx + 1

    while j < len(df) and j not in rows_to_remove:
        if not _is_continuation_row(
            df.iloc[j], narration_col, cheq_ref_col, other_cols
        ):
            break

        _merge_continuation_content(df, base_row_idx, j, narration_col, cheq_ref_col)
        rows_to_remove.append(j)
        continuation_count += 1

        _print_merge_details(
            df.iloc[j], j, base_row_idx, continuation_count, narration_col, cheq_ref_col
        )
        j += 1

    return continuation_count


def _is_continuation_row(
    row: pd.Series, narration_col, cheq_ref_col, other_cols: List
) -> bool:
    """Check if a row is a continuation row."""
    non_null_other_cols = row[other_cols].notna().sum()

    has_narration = (
        pd.notna(row[narration_col]) and str(row[narration_col]).strip() != ""
    )
    has_cheq_ref = (
        cheq_ref_col
        and pd.notna(row[cheq_ref_col])
        and str(row[cheq_ref_col]).strip() != ""
    )

    return (has_narration or has_cheq_ref) and non_null_other_cols == 0


def _merge_continuation_content(
    df: pd.DataFrame, base_idx: int, continuation_idx: int, narration_col, cheq_ref_col
) -> None:
    """Merge content from continuation row into base row."""
    continuation_row = df.iloc[continuation_idx]

    # Merge narration
    if (
        pd.notna(continuation_row[narration_col])
        and str(continuation_row[narration_col]).strip()
    ):
        current = (
            str(df.at[base_idx, narration_col])
            if pd.notna(df.at[base_idx, narration_col])
            else ""
        )
        df.at[base_idx, narration_col] = current + str(continuation_row[narration_col])

    # Merge cheq_ref
    if (
        cheq_ref_col
        and pd.notna(continuation_row[cheq_ref_col])
        and str(continuation_row[cheq_ref_col]).strip()
    ):
        current = (
            str(df.at[base_idx, cheq_ref_col])
            if pd.notna(df.at[base_idx, cheq_ref_col])
            else ""
        )
        df.at[base_idx, cheq_ref_col] = current + str(continuation_row[cheq_ref_col])


def _print_merge_details(
    row: pd.Series,
    row_idx: int,
    base_idx: int,
    part_num: int,
    narration_col,
    cheq_ref_col,
) -> None:
    """Print details about a merged continuation row."""

    if pd.notna(row[narration_col]) and str(row[narration_col]).strip():
        # logger.info(f"  Narration part: '{str(row[narration_col])}'")
        pass

    if cheq_ref_col and pd.notna(row[cheq_ref_col]) and str(row[cheq_ref_col]).strip():
        # logger.info(f"  Cheq/Ref part: '{str(row[cheq_ref_col])}'")
        pass


def _print_merge_summary(
    df: pd.DataFrame, row_idx: int, count: int, narration_col, cheq_ref_col
) -> None:
    """Print summary of all merged continuation rows."""
    logger.info(f"Total: Merged {count} continuation rows into row {row_idx}")
    logger.info(f"  Final narration: '{str(df.at[row_idx, narration_col])}'")

    if cheq_ref_col:
        logger.info(f"  Final cheq/ref: '{str(df.at[row_idx, cheq_ref_col])}'")


def _remove_marked_rows(df: pd.DataFrame, rows_to_remove: List[int]) -> pd.DataFrame:
    """Remove all rows marked for deletion."""
    if not rows_to_remove:
        return df

    df = df.drop(rows_to_remove).reset_index(drop=True)
    logger.info(f"Removed {len(rows_to_remove)} continuation rows total")
    return df


def convert_pdf_to_csv(path: str = DEFAULT_PDF_PATH) -> str:
    """Main entry point for the script."""
    parser = BankStatementParser(path)
    transactions_df = parser.parse()

    if transactions_df is not None:
        parser.save_to_csv(transactions_df)
    else:
        logger.error("Failed to extract transactions from PDF")

    return OUTPUT_CSV_PATH


if __name__ == "__main__":
    convert_pdf_to_csv()
