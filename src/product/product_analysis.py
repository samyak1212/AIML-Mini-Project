import numpy as np
import pandas as pd


# Calculate product-level metrics separately for each customer segment.
def build_product_summary(transactions_df, customers_df):
    if transactions_df.empty:
        return pd.DataFrame()

    # Find total unique customers in each cluster to calculate reach percentage.
    segment_sizes = customers_df.groupby("Cluster")["Customer ID"].nunique()

    # Aggregate quantity, unique customers, and revenue by Cluster and Description.
    summary = transactions_df.groupby(["Cluster", "Description"], as_index=False).agg(
        **{
            "Total Quantity": ("Quantity", "sum"),
            "Purchasing Customers": ("Customer ID", "nunique"),
            "Sales / Revenue": ("Revenue", "sum"),
        }
    )

    # Average quantity bought per customer for this product.
    summary["Average Quantity per Customer"] = (
        summary["Total Quantity"] / summary["Purchasing Customers"]
    )

    # Map the cluster size and calculate percentage of segment customers who bought this.
    summary["Segment Customers"] = summary["Cluster"].map(segment_sizes)
    summary["Customer Reach (%)"] = (
        summary["Purchasing Customers"] / summary["Segment Customers"] * 100
    )

    # Flag concentrated bulk purchasing: high quantity but low reach (<= 10% of customers).
    high_quantity_threshold = summary["Total Quantity"].quantile(0.90)
    summary["Purchase Pattern"] = np.where(
        (summary["Total Quantity"] >= high_quantity_threshold)
        & (summary["Customer Reach (%)"] <= 10.0),
        "Concentrated bulk purchasing",
        "Normal purchasing",
    )

    return summary


# Get the top N products purchased by a specific customer segment.
def get_top_products_for_segment(product_summary, segment_name, top_n=10):
    if product_summary.empty:
        return pd.DataFrame()

    # Filter for the chosen segment.
    segment_df = product_summary[product_summary["Cluster"] == segment_name].copy()

    # Sort descending by Total Quantity and pick the top N products.
    top_df = segment_df.sort_values("Total Quantity", ascending=False).head(top_n)

    # Keep only the relevant display columns (excluding 'Top Purchasing Segment').
    columns_to_show = [
        "Description",
        "Total Quantity",
        "Purchasing Customers",
        "Sales / Revenue",
        "Average Quantity per Customer",
        "Customer Reach (%)",
        "Purchase Pattern",
    ]

    return top_df[columns_to_show].reset_index(drop=True)
