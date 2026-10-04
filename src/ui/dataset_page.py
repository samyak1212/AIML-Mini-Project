import pandas as pd
import streamlit as st


# Display raw dataset table first, followed by dataset metadata and summary statistics.
def show_dataset_page(raw_transactions):
    st.header("Dataset Information")
    st.caption(
        "Raw transactional dataset from Online Retail II. This page contains only dataset "
        "information and has no customer segmentation or cluster-derived data."
    )

    # 1. Starting section: Raw dataset preview
    st.subheader("Raw Dataset")
    st.write("Transactions as loaded directly from the source CSV file before cleaning.")
    row_count = st.selectbox(
        "Rows to display",
        options=[100, 250, 500, 1000],
        index=0,
        help="Select the number of rows from the raw dataset to display in the table.",
    )
    st.dataframe(
        raw_transactions.head(row_count),
        use_container_width=True,
        hide_index=True,
    )

    # 2. Key dataset summary metrics
    st.subheader("Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records (Rows)", f"{len(raw_transactions):,}")
    col2.metric("Total Columns", len(raw_transactions.columns))
    col3.metric("Unique Invoices", f"{raw_transactions['Invoice'].nunique():,}")
    col4.metric("Unique Products", f"{raw_transactions['Description'].nunique():,}")

    col5, col6, col7, col8 = st.columns(4)
    raw_customers = raw_transactions["Customer ID"].dropna().nunique()
    total_missing = raw_transactions.isna().sum().sum()
    date_series = pd.to_datetime(raw_transactions["InvoiceDate"], errors="coerce")
    min_date = date_series.min().strftime("%Y-%m-%d") if date_series.notna().any() else "N/A"
    max_date = date_series.max().strftime("%Y-%m-%d") if date_series.notna().any() else "N/A"

    col5.metric("Unique Customers (Raw)", f"{raw_customers:,}")
    col6.metric("Total Missing Values", f"{total_missing:,}")
    col7.metric("Start Date", min_date)
    col8.metric("End Date", max_date)

    # 3. Column metadata and missing value breakdown
    st.subheader("Column Metadata and Missing Values")
    column_info = []
    total_rows = len(raw_transactions)
    for col in raw_transactions.columns:
        null_count = int(raw_transactions[col].isna().sum())
        non_null_count = total_rows - null_count
        null_pct = round((null_count / total_rows) * 100, 2)
        sample_val = str(raw_transactions[col].dropna().iloc[0]) if non_null_count > 0 else "None"
        column_info.append(
            {
                "Column Name": col,
                "Data Type": str(raw_transactions[col].dtype),
                "Non-Null Count": f"{non_null_count:,}",
                "Missing Count": f"{null_count:,}",
                "Missing (%)": f"{null_pct}%",
                "Sample Value": sample_val,
            }
        )
    st.dataframe(pd.DataFrame(column_info), use_container_width=True, hide_index=True)

    # 4. Descriptive statistics for numeric features (Quantity and Price)
    st.subheader("Descriptive Statistics (Raw Numerical Features)")
    st.caption("Summary statistics for Quantity and Price before cleaning invalid transactions.")
    numeric_cols = raw_transactions[["Quantity", "Price"]].apply(pd.to_numeric, errors="coerce")
    stats_df = numeric_cols.describe().T.reset_index().rename(columns={"index": "Feature"})
    stats_df = stats_df.round(2)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)
