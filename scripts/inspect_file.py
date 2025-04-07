import pandas as pd
import sys
import os

def inspect_file(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    if file_path.endswith(".csv"):
        print(f"Inspecting CSV file: {file_path}")
        df = pd.read_csv(file_path)
        print("\nFirst 5 rows:\n")
        print(df.head(10))
        print("\nColumn names:\n")
        print(df.columns.tolist())

    elif file_path.endswith(".xls") or file_path.endswith(".xlsx"):
        print(f"Inspecting Excel file: {file_path}")
        xls = pd.ExcelFile(file_path)
        print("\nAvailable sheet names:\n")
        print(xls.sheet_names)
        df = xls.parse(xls.sheet_names[0])
        print(f"\nPreview of first sheet ({xls.sheet_names[0]}):\n")
        print(df.head(10))
        print("\nColumn names:\n")
        print(df.columns.tolist())

    else:
        print("Unsupported file type.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_file.py <path-to-file>")
    else:
        inspect_file(sys.argv[1])
