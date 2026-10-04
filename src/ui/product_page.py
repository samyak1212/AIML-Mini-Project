import streamlit as st

from customer.customer_analysis import CLUSTER_ORDER
from product.product_analysis import (
    build_product_summary,
    get_top_products_for_segment,
)


# Show product-level analysis with a single top 10 table for the selected customer segment.
def show_product_page(transactions, customers, analysis):
    st.header("Product and ML Analysis")
    st.write(
        "Analyze product performance across K-Means customer segments. Examine total quantity, "
        "purchasing customers, revenue, and identify concentrated bulk purchasing vs popular products."
    )

    # 1. Top summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Silhouette Score", f"{analysis['silhouette']:.3f}")
    col2.metric("Total Unique Products", f"{transactions['Description'].nunique():,}")
    total_rev = transactions["Revenue"].sum()
    col3.metric("Total Product Revenue", f"{total_rev:,.2f}")

    # 2. Build product summary data
    product_summary = build_product_summary(transactions, customers)

    # 3. Customer segment selection dropdown
    st.subheader("Top 10 Products by Customer Segment")
    st.caption("Select a customer segment from the dropdown to view its top 10 most purchased products.")

    selected_segment = st.selectbox(
        "Select Customer Segment",
        options=CLUSTER_ORDER,
        index=0,
    )

    # 4. Single Top 10 products table (excluding 'Top Purchasing Segment')
    top_10 = get_top_products_for_segment(product_summary, selected_segment, top_n=10)

    st.write(f"**Top 10 Products Bought by {selected_segment} (by Total Quantity)**")
    st.dataframe(
        format_product_table(top_10),
        use_container_width=True,
        hide_index=True,
    )

    # 5. Business interpretation note on concentrated bulk purchasing
    st.info(
        "Concentrated Bulk Purchasing Interpretation: A product with extremely high total quantity "
        "but low customer reach (<= 10% of customers in that segment) is categorized as concentrated "
        "bulk purchasing, rather than being automatically called a broadly popular product."
    )


# Format numbers in the product table for clean display.
def format_product_table(df):
    formatted = df.copy()

    for col in [
        "Total Quantity",
        "Sales / Revenue",
        "Average Quantity per Customer",
        "Customer Reach (%)",
    ]:
        if col in formatted.columns:
            formatted[col] = formatted[col].round(2)

    if "Purchasing Customers" in formatted.columns:
        formatted["Purchasing Customers"] = formatted["Purchasing Customers"].astype(int)

    return formatted
