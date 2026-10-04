import pandas as pd
import streamlit as st

from customer.customer_analysis import (
    CLUSTER_ORDER,
    build_business_summary,
    build_daily_time_summary,
)


# Show customer clustering results, model evaluation, business summary, and daily time analysis.
def show_customer_page(transactions, customers, analysis):
    st.header("Customer Segmentation")
    st.write(
        "Customers are segmented using K-Means (K=3) based on two engineered features: "
        "Average Quantity per Line and Spend per Unique Product. Cluster names describe "
        "purchasing behaviour without assuming wholesale or retail labels."
    )

    # 1. Key metrics overview
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Segmented Customers", f"{len(customers):,}")
    total_rev = transactions["Revenue"].sum()
    col2.metric("Total Sales / Revenue", f"{total_rev:,.2f}")
    col3.metric("Silhouette Score", f"{analysis['silhouette']:.3f}")

    # 2. Model evaluation explanation
    st.subheader("Model Evaluation")
    st.metric("Silhouette Score", f"{analysis['silhouette']:.3f}")
    st.info(
        "Accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix are "
        "not applicable here because K-Means is an unsupervised clustering algorithm "
        "and the dataset contains no ground truth customer segment labels. "
        "Cluster quality is evaluated using the Silhouette Score (~0.744)."
    )

    # 3. 2D scatter plot with thin cross centroids
    show_cluster_scatter_plot(customers, analysis)

    # 4. Business analysis summary table by segment
    st.subheader("Business Analysis by Customer Segment")
    st.caption("Key commercial and behavioral metrics aggregated across each customer cluster.")
    business_summary = build_business_summary(transactions)
    st.dataframe(
        format_business_table(business_summary),
        use_container_width=True,
        hide_index=True,
    )

    # 5. Time analysis: Single everyday chart and all daily data
    show_daily_time_analysis(transactions)


# Display the 2D scatter plot of customer clusters with thin cross centers.
def show_cluster_scatter_plot(customers, analysis):
    st.subheader("K-Means Cluster Visualization (with Cluster Centers)")
    st.write(
        "Each point represents a customer. Axes show the 2% tail-clipped feature values "
        "that were standardized using StandardScaler before K-Means clustering. "
        "Cluster centers (centroids) are marked with simple thin cross markers (**+**)."
    )

    # Customer scatter data
    customer_points = customers[
        [
            "Customer ID",
            "Cluster",
            "Average Quantity per Line (Clipped)",
            "Spend per Unique Product (Clipped)",
        ]
    ].copy()
    customer_points["Point Type"] = "Customer"

    # Centroid data
    centroids = analysis.get("centroids")
    if centroids is None or not isinstance(centroids, pd.DataFrame):
        centroids = (
            customers.groupby("Cluster", as_index=False)[
                [
                    "Average Quantity per Line (Clipped)",
                    "Spend per Unique Product (Clipped)",
                ]
            ]
            .mean()
            .copy()
        )
    else:
        centroids = centroids.copy()

    centroids["Customer ID"] = "Centroid"
    centroids["Point Type"] = "Cluster Center"

    # Combine both datasets for layered Vega-Lite chart
    combined_data = pd.concat(
        [
            customer_points,
            centroids[
                [
                    "Customer ID",
                    "Cluster",
                    "Average Quantity per Line (Clipped)",
                    "Spend per Unique Product (Clipped)",
                    "Point Type",
                ]
            ],
        ],
        ignore_index=True,
    )

    chart_spec = {
        "layer": [
            {
                # Layer 1: Customer points
                "transform": [{"filter": "datum['Point Type'] === 'Customer'"}],
                "mark": {"type": "point", "filled": True, "size": 36, "opacity": 0.65},
                "encoding": {
                    "x": {
                        "field": "Average Quantity per Line (Clipped)",
                        "type": "quantitative",
                        "title": "Average Quantity per Line (2% clipped)",
                    },
                    "y": {
                        "field": "Spend per Unique Product (Clipped)",
                        "type": "quantitative",
                        "title": "Spend per Unique Product (2% clipped)",
                    },
                    "color": {"field": "Cluster", "type": "nominal"},
                    "tooltip": [
                        {"field": "Customer ID", "type": "nominal"},
                        {"field": "Cluster", "type": "nominal"},
                        {
                            "field": "Average Quantity per Line (Clipped)",
                            "type": "quantitative",
                            "format": ".2f",
                        },
                        {
                            "field": "Spend per Unique Product (Clipped)",
                            "type": "quantitative",
                            "format": ".2f",
                        },
                    ],
                },
            },
            {
                # Layer 2: Cluster center thin simple cross marker (no text above)
                "transform": [{"filter": "datum['Point Type'] === 'Cluster Center'"}],
                "mark": {
                    "type": "point",
                    "shape": "cross",
                    "size": 180,
                    "strokeWidth": 1.5,
                    "color": "#FF4B4B",
                },
                "encoding": {
                    "x": {
                        "field": "Average Quantity per Line (Clipped)",
                        "type": "quantitative",
                    },
                    "y": {
                        "field": "Spend per Unique Product (Clipped)",
                        "type": "quantitative",
                    },
                    "tooltip": [
                        {"field": "Cluster", "type": "nominal", "title": "Cluster Center"},
                        {
                            "field": "Average Quantity per Line (Clipped)",
                            "type": "quantitative",
                            "format": ".2f",
                            "title": "Center Avg Qty",
                        },
                        {
                            "field": "Spend per Unique Product (Clipped)",
                            "type": "quantitative",
                            "format": ".2f",
                            "title": "Center Spend",
                        },
                    ],
                },
            },
        ],
        "height": 450,
    }

    st.vega_lite_chart(combined_data, chart_spec, use_container_width=True)

    # Centroid coordinates and clipping bounds
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        with st.expander("Cluster Center Coordinates (Centroids)"):
            display_centroids = centroids[
                [
                    "Cluster",
                    "Average Quantity per Line (Clipped)",
                    "Spend per Unique Product (Clipped)",
                ]
            ].round(2)
            st.dataframe(display_centroids, use_container_width=True, hide_index=True)
    with col_c2:
        with st.expander("2% Percentile Tail Clipping Bounds"):
            st.dataframe(analysis["clipping_bounds"], use_container_width=True, hide_index=True)


# Show one single everyday sales line chart and all daily data records.
def show_daily_time_analysis(transactions):
    st.subheader("Time Analysis (Everyday Trend)")
    st.caption(
        "Daily Sales / Revenue across all customer segments for every day present in the dataset. "
        "No date or frequency filters applied."
    )

    daily_summary = build_daily_time_summary(transactions)

    # Pivot for single line chart of daily sales by segment
    chart_df = daily_summary.pivot(
        index="Date", columns="Cluster", values="Sales / Revenue"
    ).reindex(columns=CLUSTER_ORDER).fillna(0)

    st.write("**Daily Sales / Revenue by Customer Segment**")
    st.line_chart(chart_df, use_container_width=True)

    st.write("**All Daily Data Present**")
    st.caption(f"Showing all {len(daily_summary):,} daily segment records in the dataset.")
    st.dataframe(
        format_daily_table(daily_summary),
        use_container_width=True,
        hide_index=True,
    )


# Format numbers in the business analysis table.
def format_business_table(df):
    formatted = df.copy()
    round_cols = [
        "Average Spending per Customer",
        "Average Quantity per Customer",
        "Average Unique Products",
        "Average Orders",
        "Total Sales / Revenue",
        "Total Quantity Purchased",
    ]
    for col in round_cols:
        if col in formatted.columns:
            formatted[col] = formatted[col].round(2)
    return formatted


# Format numbers and types in the daily time table.
def format_daily_table(df):
    formatted = df.copy()
    for col in [
        "Sales / Revenue",
        "Quantity",
        "Average Spending / Customer",
        "Average Quantity / Customer",
    ]:
        if col in formatted.columns:
            formatted[col] = formatted[col].round(2)

    for col in ["Orders", "Active Customers"]:
        if col in formatted.columns:
            formatted[col] = formatted[col].astype(int)

    preferred_order = [
        "Date",
        "Cluster",
        "Sales / Revenue",
        "Quantity",
        "Orders",
        "Active Customers",
        "Average Spending / Customer",
        "Average Quantity / Customer",
    ]
    cols = [c for c in preferred_order if c in formatted.columns]
    return formatted[cols]
