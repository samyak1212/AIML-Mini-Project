import pandas as pd


# Columns required in the raw transaction CSV file.
REQUIRED_COLUMNS = [
    "Invoice",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
]


# Load raw transaction data directly from the CSV file.
def load_raw_data(csv_file_path):
    # Read the CSV without altering data types or dropping any rows.
    raw_df = pd.read_csv(csv_file_path, low_memory=False)
    raw_df.columns = raw_df.columns.str.strip()
    return raw_df


# Clean invalid rows from the raw transaction data.
def clean_transactions(raw_df):
    # Make a copy to keep the raw dataset untouched.
    df = raw_df.copy()

    # Convert numeric and date fields into proper types.
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Customer ID"] = pd.to_numeric(df["Customer ID"], errors="coerce")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    # Filter: Quantity > 0, Price > 0, valid Customer ID, valid Invoice and Date.
    valid_mask = (
        (df["Quantity"] > 0)
        & (df["Price"] > 0)
        & (df["Customer ID"].notna())
        & (df["InvoiceDate"].notna())
        & (df["Invoice"].notna())
    )
    df = df[valid_mask].copy()

    # Clean product description and convert Customer ID to integer.
    df["Customer ID"] = df["Customer ID"].astype("int64")
    df["Description"] = df["Description"].fillna("Unknown Product").astype(str).str.strip()

    # Calculate Revenue (Price * Quantity) and add date-only column.
    df["Revenue"] = df["Quantity"] * df["Price"]
    df["Date"] = df["InvoiceDate"].dt.normalize()

    return df
