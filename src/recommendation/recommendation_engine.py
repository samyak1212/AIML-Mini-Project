import pandas as pd


# Get popular products from transaction history to populate the demo dropdown.
def get_catalog_products(transactions_df, limit=150):
    if transactions_df.empty:
        return []

    # Sort products by total transaction count so top popular products appear first.
    popular = (
        transactions_df.groupby("Description")["Invoice"]
        .count()
        .sort_values(ascending=False)
        .head(limit)
        .index.tolist()
    )
    return popular


# Get the typical unit price for a given product from historical transactions.
def get_product_unit_price(transactions_df, product_name):
    matches = transactions_df[transactions_df["Description"] == product_name]
    if matches.empty:
        return 1.0
    return float(matches["Price"].median())


# Generate product recommendations using Association Rule Learning (Frequently Bought Together).
def get_product_recommendations(transactions_df, selected_product, top_n=5):
    if transactions_df.empty or not selected_product:
        return pd.DataFrame()

    # 1. Find all invoice IDs where the selected product was purchased.
    invoices_with_product = transactions_df[
        transactions_df["Description"] == selected_product
    ]["Invoice"].unique()

    total_target_invoices = len(invoices_with_product)
    if total_target_invoices == 0:
        return pd.DataFrame()

    # 2. Find other items bought in those exact same invoices (market basket).
    basket_items = transactions_df[
        (transactions_df["Invoice"].isin(invoices_with_product))
        & (transactions_df["Description"] != selected_product)
    ]

    if basket_items.empty:
        # Fallback to general popular products if no co-purchases exist
        fallback = (
            transactions_df[transactions_df["Description"] != selected_product]
            .groupby("Description", as_index=False)
            .agg(Times_Bought_Together=("Invoice", "nunique"), Price=("Price", "mean"))
            .sort_values("Times_Bought_Together", ascending=False)
            .head(top_n)
        )
        fallback["Confidence (%)"] = 0.0
        return fallback.rename(
            columns={
                "Description": "Recommended Product",
                "Times_Bought_Together": "Times Bought Together",
                "Price": "Estimated Unit Price",
            }
        )

    # 3. Calculate co-occurrence count and confidence percentage for each companion product.
    co_occurrences = basket_items.groupby("Description", as_index=False).agg(
        Times_Bought_Together=("Invoice", "nunique"),
        Estimated_Unit_Price=("Price", "median"),
    )

    # Confidence: P(Recommended Product | Selected Product)
    co_occurrences["Confidence (%)"] = (
        co_occurrences["Times_Bought_Together"] / total_target_invoices
    ) * 100

    # Pick top N highest co-purchased products.
    top_recs = co_occurrences.sort_values("Times_Bought_Together", ascending=False).head(top_n)

    top_recs = top_recs.rename(
        columns={
            "Description": "Recommended Product",
            "Times_Bought_Together": "Times Bought Together (Invoices)",
            "Estimated_Unit_Price": "Unit Price (£)",
        }
    )

    return top_recs.reset_index(drop=True)
