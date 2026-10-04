import streamlit as st

from customer.customer_analysis import classify_simulated_customer
from recommendation.recommendation_engine import (
    get_catalog_products,
    get_product_recommendations,
    get_product_unit_price,
)


# Show the interactive demo for product recommendations and real-time customer cluster classification.
def show_recommendation_page(transactions, customers, analysis):
    st.header("Product Recommendations & Customer Classification Demo")
    st.write(
        "Simulate a customer purchase: select a product, specify the order quantity, "
        "and receive real-time product recommendations (Association Rule Learning) "
        "while automatically classifying the customer into a K-Means cluster."
    )

    # Load list of top catalog products for the dropdown
    catalog = get_catalog_products(transactions, limit=150)

    # 1. Interactive Demo Section
    st.subheader("1. Interactive Order Simulator")

    col_sim1, col_sim2 = st.columns([1.5, 1.2])

    with col_sim1:
        selected_product = st.selectbox(
            "Select a Product",
            options=catalog,
            index=0,
            help="Choose any product from the catalog to simulate an order.",
        )

        unit_price = get_product_unit_price(transactions, selected_product)
        st.caption(f"Estimated Unit Price: **£{unit_price:.2f}**")

        order_quantity = st.number_input(
            "Order Quantity (Units)",
            min_value=1,
            max_value=2500,
            value=6,
            step=1,
            help="Change the quantity to test how purchasing volume affects cluster classification.",
        )

        total_spend = order_quantity * unit_price
        st.write(f"**Total Order Value:** £{total_spend:,.2f}")

    # 2. Real-Time Customer Cluster Classification
    with col_sim2:
        st.subheader("Customer Classification")

        # The simulated order has 1 unique product line
        avg_qty_per_line = float(order_quantity)
        spend_per_unique_product = float(total_spend)

        # Classify the simulated customer using K-Means model
        predicted_cluster, clipped_q, clipped_s = classify_simulated_customer(
            avg_qty_per_line, spend_per_unique_product, analysis
        )

        # Display the predicted cluster prominently
        st.success(f"**Classified Segment:** {predicted_cluster}")

        # Metrics for the simulated customer
        m1, m2 = st.columns(2)
        m1.metric("Avg Quantity / Line", f"{avg_qty_per_line:.1f}")
        m2.metric("Spend / Unique Product", f"£{spend_per_unique_product:,.2f}")

        # Explain the classification reasoning
        if predicted_cluster == "Regular Customers":
            st.info(
                "Classification Rationale: Low to moderate quantity and spend per line. "
                "Matches standard individual retail buying behavior."
            )
        elif predicted_cluster == "Mid-Value High-Quantity Buyers":
            st.info(
                "Classification Rationale: Higher quantity per line than regular shoppers. "
                "Represents volume purchasers with intermediate total spend."
            )
        else:
            st.info(
                "Classification Rationale: Very large purchasing quantity and high order value. "
                "Exhibits heavy buyer behavior."
            )

    # 3. Product Recommendations Section
    st.divider()
    st.subheader(f"2. Recommended Products for: {selected_product}")
    st.caption(
        "Products frequently purchased together with this item in the same invoice "
        "(Association Rule Learning from Module 4 of the syllabus)."
    )

    recommendations = get_product_recommendations(
        transactions, selected_product, top_n=5
    )

    if not recommendations.empty:
        # Format columns for display
        display_recs = recommendations.copy()
        if "Confidence (%)" in display_recs.columns:
            display_recs["Confidence (%)"] = display_recs["Confidence (%)"].round(1).astype(str) + "%"
        if "Unit Price (£)" in display_recs.columns:
            display_recs["Unit Price (£)"] = display_recs["Unit Price (£)"].round(2)

        st.dataframe(display_recs, use_container_width=True, hide_index=True)
    else:
        st.info("No co-purchase records found for this specific item.")

    st.caption(
        "Association rule confidence represents the percentage of baskets containing "
        f"'{selected_product}' that also included the companion product."
    )
