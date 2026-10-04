import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from data_preprocessing.clean_data import clean_transactions, load_raw_data

# The two engineered features used for K-Means clustering.
CLUSTER_FEATURES = ["Average Quantity per Line", "Spend per Unique Product"]

# Behavior-based names for the 3 clusters (ordered from low to high activity).
CLUSTER_ORDER = [
    "Regular Customers",
    "Mid-Value High-Quantity Buyers",
    "High-Value / Heavy Buyers",
]


# Compute customer-level aggregate features from transaction data.
def build_customer_features(transactions_df):
    grouped = transactions_df.groupby("Customer ID", as_index=False).agg(
        Total_Quantity=("Quantity", "sum"),
        Transaction_Lines=("Quantity", "count"),
        Total_Revenue=("Revenue", "sum"),
        Unique_Products=("Description", "nunique"),
        Orders=("Invoice", "nunique"),
    )

    # Feature 1: Average Quantity per Line
    grouped["Average Quantity per Line"] = (
        grouped["Total_Quantity"] / grouped["Transaction_Lines"]
    )

    # Feature 2: Spend per Unique Product
    grouped["Spend per Unique Product"] = (
        grouped["Total_Revenue"] / grouped["Unique_Products"]
    )

    return grouped


# Cap extreme feature values at the 2nd and 98th percentiles.
def apply_tail_clipping(customers_df):
    features_to_clip = customers_df[CLUSTER_FEATURES].copy()

    lower_limits = features_to_clip.quantile(0.02)
    upper_limits = features_to_clip.quantile(0.98)

    # Clip values within bounds instead of removing customers.
    clipped = features_to_clip.clip(lower=lower_limits, upper=upper_limits, axis=1)

    bounds_df = pd.DataFrame(
        {"Feature": CLUSTER_FEATURES, "2% Cap": lower_limits.values, "98% Cap": upper_limits.values}
    )

    return clipped, bounds_df


# Map numeric cluster IDs to behavior-based names based on feature averages.
def assign_behavior_labels(customers_df, cluster_labels):
    temp_df = customers_df.copy()
    temp_df["Cluster ID"] = cluster_labels

    # Calculate the average values of the two features for each cluster.
    cluster_means = temp_df.groupby("Cluster ID")[CLUSTER_FEATURES].mean()

    # Sum of ranks orders clusters from lowest to highest purchasing volume.
    rank_sum = (
        cluster_means[CLUSTER_FEATURES[0]].rank()
        + cluster_means[CLUSTER_FEATURES[1]].rank()
    )
    sorted_cluster_ids = rank_sum.sort_values().index.tolist()

    # Map the lowest cluster to Regular, middle to Mid-Value, highest to High-Value.
    id_to_name = dict(zip(sorted_cluster_ids, CLUSTER_ORDER))
    return id_to_name


# Run complete customer segmentation pipeline using K-Means (K=3).
def run_customer_clustering(csv_file_path):
    raw_df = load_raw_data(csv_file_path)
    clean_df = clean_transactions(raw_df)

    customers = build_customer_features(clean_df)
    clipped_features, clipping_bounds = apply_tail_clipping(customers)

    # Standardize features using StandardScaler.
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(clipped_features)

    # Run K-Means with K=3.
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_values)

    # Assign behavior-based cluster names.
    name_mapping = assign_behavior_labels(customers, cluster_labels)
    customers["Cluster ID"] = cluster_labels
    customers["Cluster"] = customers["Cluster ID"].map(name_mapping)

    # Add clipped feature values for display in the scatter plot.
    customers["Average Quantity per Line (Clipped)"] = clipped_features["Average Quantity per Line"]
    customers["Spend per Unique Product (Clipped)"] = clipped_features["Spend per Unique Product"]

    # Merge cluster names into the transactions table.
    clustered_transactions = clean_df.merge(
        customers[["Customer ID", "Cluster ID", "Cluster"]],
        on="Customer ID",
        how="inner",
    )

    # Calculate cluster center coordinates back in the clipped feature space.
    unscaled_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    centroids = pd.DataFrame(
        unscaled_centers,
        columns=[
            "Average Quantity per Line (Clipped)",
            "Spend per Unique Product (Clipped)",
        ],
    )
    centroids["Cluster ID"] = range(len(centroids))
    centroids["Cluster"] = centroids["Cluster ID"].map(name_mapping)

    # Calculate silhouette score.
    score = float(silhouette_score(scaled_values, cluster_labels))

    return {
        "raw_transactions": raw_df,
        "transactions": clustered_transactions,
        "customers": customers,
        "clipping_bounds": clipping_bounds,
        "centroids": centroids,
        "silhouette": score,
        "scaler": scaler,
        "kmeans": kmeans,
        "name_mapping": name_mapping,
    }


# Classify a new or simulated customer order into one of the 3 customer clusters.
def classify_simulated_customer(avg_qty, spend_per_product, analysis):
    clipping_bounds = analysis["clipping_bounds"]

    # Extract 2% percentile tail caps used during training
    qty_row = clipping_bounds[clipping_bounds["Feature"] == "Average Quantity per Line"].iloc[0]
    spend_row = clipping_bounds[clipping_bounds["Feature"] == "Spend per Unique Product"].iloc[0]

    # Clip values within bounds
    clipped_qty = float(min(max(avg_qty, qty_row["2% Cap"]), qty_row["98% Cap"]))
    clipped_spend = float(min(max(spend_per_product, spend_row["2% Cap"]), spend_row["98% Cap"]))

    scaler = analysis.get("scaler")
    kmeans = analysis.get("kmeans")
    name_mapping = analysis.get("name_mapping")

    # If scaler and kmeans models are available, predict directly
    if scaler is not None and kmeans is not None and name_mapping is not None:
        point_df = pd.DataFrame([[clipped_qty, clipped_spend]], columns=CLUSTER_FEATURES)
        scaled_point = scaler.transform(point_df)
        cluster_id = int(kmeans.predict(scaled_point)[0])
        return name_mapping[cluster_id], clipped_qty, clipped_spend

    # Fallback: Nearest centroid in clipped feature space
    centroids = analysis["centroids"]
    best_cluster = None
    min_dist = float("inf")
    for _, row in centroids.iterrows():
        c_q = row["Average Quantity per Line (Clipped)"]
        c_s = row["Spend per Unique Product (Clipped)"]
        dist = ((clipped_qty - c_q) ** 2 + (clipped_spend - c_s) ** 2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            best_cluster = row["Cluster"]

    return best_cluster, clipped_qty, clipped_spend


# Calculate business performance metrics aggregated by customer cluster.
def build_business_summary(transactions_df):
    if transactions_df.empty:
        return pd.DataFrame()

    # Aggregate by customer first.
    customer_agg = transactions_df.groupby(["Cluster", "Customer ID"], as_index=False).agg(
        Spending=("Revenue", "sum"),
        Quantity=("Quantity", "sum"),
        Unique_Products=("Description", "nunique"),
        Orders=("Invoice", "nunique"),
    )

    # Aggregate across each cluster.
    summary = customer_agg.groupby("Cluster", as_index=False).agg(
        **{
            "Number of Customers": ("Customer ID", "nunique"),
            "Average Spending per Customer": ("Spending", "mean"),
            "Average Quantity per Customer": ("Quantity", "mean"),
            "Average Unique Products": ("Unique_Products", "mean"),
            "Average Orders": ("Orders", "mean"),
            "Total Sales / Revenue": ("Spending", "sum"),
            "Total Quantity Purchased": ("Quantity", "sum"),
        }
    )

    # Sort rows according to the standard cluster order.
    summary["Cluster"] = pd.Categorical(summary["Cluster"], categories=CLUSTER_ORDER, ordered=True)
    return summary.sort_values("Cluster").reset_index(drop=True)


# Calculate daily sales and customer activity measures for every day.
def build_daily_time_summary(transactions_df):
    if transactions_df.empty:
        return pd.DataFrame()

    df = transactions_df.copy()
    df["Date"] = df["InvoiceDate"].dt.date

    # Group by Date and Cluster.
    daily = df.groupby(["Date", "Cluster"], as_index=False).agg(
        **{
            "Sales / Revenue": ("Revenue", "sum"),
            "Quantity": ("Quantity", "sum"),
            "Orders": ("Invoice", "nunique"),
            "Active Customers": ("Customer ID", "nunique"),
        }
    )

    # Compute per-customer averages.
    daily["Average Spending / Customer"] = (
        daily["Sales / Revenue"] / daily["Active Customers"]
    )
    daily["Average Quantity / Customer"] = (
        daily["Quantity"] / daily["Active Customers"]
    )

    return daily.sort_values(["Date", "Cluster"]).reset_index(drop=True)
