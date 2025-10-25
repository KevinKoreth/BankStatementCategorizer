import tabula.io as tabula
import pandas as pd


def merge_continuation_rows(df):
    """
    Merge continuation rows where narration/cheq_ref extends across multiple rows.
    Continuation rows have only the narration and/or cheq_ref columns filled and rest are empty/NaN.
    This function can handle cases where content spans more than 3 rows.
    """
    if df.empty:
        return df

    # Make a copy to avoid modifying the original
    df = df.copy()

    # Get column names - assuming narration is the second column (index 1)
    narration_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    cheq_ref_col = df.columns[2] if len(df.columns) > 2 else None

    # Define columns that should be empty in continuation rows
    other_cols = [
        col for col in df.columns if col != narration_col and col != cheq_ref_col
    ]

    rows_to_remove = []
    i = 0

    while i < len(df):
        # Skip if this row is already marked for removal
        if i in rows_to_remove:
            i += 1
            continue

        # Look ahead for continuation rows
        continuation_count = 0
        j = i + 1

        # Keep looking for consecutive continuation rows
        while j < len(df) and j not in rows_to_remove:
            next_row = df.iloc[j]

            # Count non-null values in other columns (excluding narration and cheq_ref)
            non_null_other_cols = next_row[other_cols].notna().sum()

            # Check if this row has content in narration or cheq_ref but empty other columns
            has_narration = (
                pd.notna(next_row[narration_col])
                and str(next_row[narration_col]).strip() != ""
            )
            has_cheq_ref = (
                cheq_ref_col
                and pd.notna(next_row[cheq_ref_col])
                and str(next_row[cheq_ref_col]).strip() != ""
            )

            # This is a continuation row if it has narration/cheq_ref content but other important columns are empty
            if (has_narration or has_cheq_ref) and non_null_other_cols == 0:
                continuation_count += 1

                # Merge narration content
                if has_narration:
                    current_narration = (
                        str(df.at[i, narration_col])
                        if pd.notna(df.at[i, narration_col])
                        else ""
                    )
                    continuation_narration = str(next_row[narration_col])
                    merged_narration = current_narration + continuation_narration
                    df.at[i, narration_col] = merged_narration

                # Merge cheq_ref content
                if has_cheq_ref:
                    current_cheq_ref = (
                        str(df.at[i, cheq_ref_col])
                        if pd.notna(df.at[i, cheq_ref_col])
                        else ""
                    )
                    continuation_cheq_ref = str(next_row[cheq_ref_col])
                    merged_cheq_ref = current_cheq_ref + continuation_cheq_ref
                    df.at[i, cheq_ref_col] = merged_cheq_ref

                # Mark this continuation row for removal
                rows_to_remove.append(j)

                print(
                    f"Merged continuation row {j} into row {i} (part {continuation_count})"
                )
                if has_narration:
                    print(f"  Narration part: '{str(next_row[narration_col])}'")
                if has_cheq_ref:
                    print(f"  Cheq/Ref part: '{str(next_row[cheq_ref_col])}'")

                j += 1
            else:
                # Not a continuation row, stop looking
                break

        if continuation_count > 0:
            print(f"Total: Merged {continuation_count} continuation rows into row {i}")
            print(f"  Final narration: '{str(df.at[i, narration_col])}'")
            if cheq_ref_col:
                print(f"  Final cheq/ref: '{str(df.at[i, cheq_ref_col])}'")

        i += 1

    # Remove all continuation rows
    if rows_to_remove:
        df = df.drop(rows_to_remove).reset_index(drop=True)
        print(f"Removed {len(rows_to_remove)} continuation rows total")

    return df


pdf_path = "sample_statements/HDFC/pdf_sample_1.pdf"

# Read all pages and all tables
df_list = tabula.read_pdf(pdf_path, stream=True, pages="all", multiple_tables=True)

if df_list:
    print(f"Found {len(df_list)} tables in the PDF")

    # Print info about each table to help debug
    for i, df in enumerate(df_list):
        if isinstance(df, pd.DataFrame):
            print(f"Table {i}: {len(df)} rows, columns: {list(df.columns)}")
        else:
            print(f"Table {i}: Not a DataFrame - {type(df)}")

    # Combine all tables - only first table has headers
    transaction_tables = []
    expected_columns = None

    for i, df in enumerate(df_list):
        # Check if this is a DataFrame and looks like a transaction table
        if isinstance(df, pd.DataFrame) and not df.empty:
            if i == 0:
                # First table should have headers: Date, Narration, Chq./Ref.No., Value Dt, Withdrawal Amt., Deposit Amt., Closing Balance
                if any(col in df.columns for col in ["Date", "Narration", "Value Dt"]):
                    expected_columns = df.columns.tolist()
                    print(
                        f"Adding table {i} as first transaction table with headers ({len(df)} rows)"
                    )
                    print(f"Expected columns: {expected_columns}")
                    transaction_tables.append(df)
                else:
                    print(
                        f"First table doesn't have expected headers - columns: {list(df.columns)}"
                    )
            else:
                # Subsequent tables have same structure but no headers
                if expected_columns and len(df.columns) == len(expected_columns):
                    # Assign the same column names from first table
                    df.columns = expected_columns

                    # Remove any rows that might be header repetitions
                    # (rows containing header text like 'Date', 'Narration', etc.)
                    df = df[
                        ~df.iloc[:, 0].astype(str).str.contains("Date|date", na=False)
                    ]

                    if not df.empty:
                        print(
                            f"Adding table {i} as continuation table ({len(df)} rows)"
                        )
                        transaction_tables.append(df)
                    else:
                        print(f"Table {i} is empty after filtering header rows")
                else:
                    print(
                        f"Table {i} has different structure - columns: {len(df.columns)} vs expected {len(expected_columns) if expected_columns else 'N/A'}"
                    )

    if transaction_tables:
        # Combine all transaction tables
        transactions_df = pd.concat(transaction_tables, ignore_index=True)

        # Remove any completely empty rows
        transactions_df = transactions_df.dropna(how="all")

        # Handle continuation rows where narration extends to next row
        transactions_df = merge_continuation_rows(transactions_df)

        output_csv_path = "bank_statement_transactions.csv"
        transactions_df.to_csv(output_csv_path, index=False)
        print(f"Bank statement converted to CSV: {output_csv_path}")
        print(f"Total transactions extracted: {len(transactions_df)}")
    else:
        print("No transaction tables found!")
        # Fallback to using the first table if it's a DataFrame
        if df_list and len(df_list) > 0:
            first_table = df_list[0]
            if isinstance(first_table, pd.DataFrame):
                transactions_df = first_table
                output_csv_path = "bank_statement_transactions.csv"
                transactions_df.to_csv(output_csv_path, index=False)
                print(f"Using first table as fallback: {output_csv_path}")
            else:
                print("First table is not a DataFrame")
        else:
            print("No tables available for fallback")
else:
    print("No tables found in the PDF")
