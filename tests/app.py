import streamlit as st

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
import heapq
from collections import deque
from mlxtend.frequent_patterns import apriori, association_rules



from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans, AgglomerativeClustering

from sklearn.mixture import GaussianMixture

from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_score

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    confusion_matrix,

    roc_auc_score,

    mean_absolute_error,

    mean_squared_error,

    r2_score

)



st.set_page_config(

    page_title="Smart Retail Intelligence",

    page_icon="🛒",

    layout="wide"

)



st.title("Smart Retail Intelligence System")

st.write(

    "Customer segmentation, prediction, spending analysis "

    "and product intelligence using Machine Learning."

)





# =========================================================

# LOAD DATA

# =========================================================



@st.cache_data

def load_data():



    df = pd.read_csv(

        "data/online_retail_II-Year 2009-2010.csv",

        encoding="latin1"

    )



    df.columns = [c.strip() for c in df.columns]



    rename = {}



    for c in df.columns:



        name = (

            c.lower()

            .replace(" ", "")

            .replace("_", "")

        )



        if name == "invoiceno":

            rename[c] = "Invoice"



        elif name == "description":

            rename[c] = "Description"



        elif name == "quantity":

            rename[c] = "Quantity"



        elif name in ["invoicedate", "date"]:

            rename[c] = "InvoiceDate"



        elif name in ["unitprice", "price"]:

            rename[c] = "UnitPrice"



        elif name in ["customerid", "customer"]:

            rename[c] = "CustomerID"



        elif name == "country":

            rename[c] = "Country"



    df = df.rename(columns=rename)



    df["InvoiceDate"] = pd.to_datetime(

        df["InvoiceDate"],

        errors="coerce"

    )



    df["Quantity"] = pd.to_numeric(

        df["Quantity"],

        errors="coerce"

    )



    df["UnitPrice"] = pd.to_numeric(

        df["UnitPrice"],

        errors="coerce"

    )



    df["CustomerID"] = pd.to_numeric(

        df["CustomerID"],

        errors="coerce"

    )



    return df





# =========================================================

# CLEAN DATA

# =========================================================



@st.cache_data

def clean_data(df):



    df = df.dropna(

        subset=[

            "CustomerID",

            "InvoiceDate",

            "Quantity",

            "UnitPrice"

        ]

    )



    df = df[

        (df["Quantity"] > 0) &

        (df["UnitPrice"] > 0)

    ].copy()



    df = df[

        ~df["Invoice"]

        .astype(str)

        .str.startswith("C")

    ].copy()



    df["Amount"] = (

        df["Quantity"] *

        df["UnitPrice"]

    )



    return df





# =========================================================

# CUSTOMER FEATURES

# =========================================================



@st.cache_data

def create_customer_features(df):



    reference_date = (

        df["InvoiceDate"].max()

        + pd.Timedelta(days=1)

    )



    customer = df.groupby("CustomerID").agg(



        Recency=(

            "InvoiceDate",

            lambda x:

            (reference_date - x.max()).days

        ),



        Frequency=(

            "Invoice",

            "nunique"

        ),



        Monetary=(

            "Amount",

            "sum"

        ),



        Products=(

            "Description",

            "nunique"

        ),



        Quantity=(

            "Quantity",

            "sum"

        )

    )



    customer["AverageOrderValue"] = (

        customer["Monetary"] /

        customer["Frequency"]

    )



    return customer





# =========================================================

# LOAD AND CLEAN

# =========================================================



try:



    raw_df = load_data()

    df = clean_data(raw_df)

    customer = create_customer_features(df)



except Exception as e:



    st.error(

        f"Could not load the dataset: {e}"

    )



    st.stop()





# =========================================================

# SIDEBAR

# =========================================================



st.sidebar.title("Navigation")




# =========================================================
# PRODUCT INTELLIGENCE FUNCTIONS
# =========================================================

@st.cache_data
def create_product_basket(df):
    required = {"Invoice", "Description", "Quantity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing))
        )

    work = df.dropna(
        subset=["Invoice", "Description", "Quantity"]
    ).copy()

    work["Quantity"] = pd.to_numeric(
        work["Quantity"], errors="coerce"
    )
    work = work.dropna(subset=["Quantity"])
    work = work[work["Quantity"] > 0]
    work["Invoice"] = work["Invoice"].astype(str)
    work["Description"] = (
        work["Description"].astype(str).str.strip()
    )
    work = work[work["Description"] != ""]

    basket = (
        work.groupby(["Invoice", "Description"])["Quantity"]
        .sum()
        .unstack(fill_value=0)
    )
    return (basket > 0).astype(int)


@st.cache_data
def run_product_apriori(basket, min_support, min_confidence):
    frequent = apriori(
        basket,
        min_support=float(min_support),
        use_colnames=True,
        low_memory=True
    )

    if frequent.empty:
        return frequent, pd.DataFrame()

    rules = association_rules(
        frequent,
        metric="confidence",
        min_threshold=float(min_confidence)
    )

    if not rules.empty:
        rules = rules.sort_values(
            ["lift", "confidence", "support"],
            ascending=False
        ).reset_index(drop=True)

    return frequent, rules


def build_product_graph(rules):
    graph = {}

    for _, row in rules.iterrows():
        cost = 1.0 / max(float(row["lift"]), 0.0001)

        for a in row["antecedents"]:
            graph.setdefault(a, {})

            for b in row["consequents"]:
                graph.setdefault(b, {})

                if b not in graph[a] or cost < graph[a][b]:
                    graph[a][b] = cost
                    graph[b][a] = cost

    return graph


def product_bfs(graph, start, goal):
    queue = deque([[start]])
    seen = {start}

    while queue:
        path = queue.popleft()
        node = path[-1]

        if node == goal:
            return path

        for neighbor in graph.get(node, {}):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(path + [neighbor])

    return None


def product_dfs(graph, start, goal):
    stack = [[start]]
    seen = set()

    while stack:
        path = stack.pop()
        node = path[-1]

        if node == goal:
            return path

        if node in seen:
            continue

        seen.add(node)

        for neighbor in graph.get(node, {}):
            if neighbor not in seen:
                stack.append(path + [neighbor])

    return None


def product_ucs(graph, start, goal):
    heap = [(0.0, [start])]
    best = {start: 0.0}

    while heap:
        cost, path = heapq.heappop(heap)
        node = path[-1]

        if node == goal:
            return path, cost

        if cost > best.get(node, float("inf")):
            continue

        for neighbor, edge in graph.get(node, {}).items():
            new_cost = cost + edge

            if new_cost < best.get(neighbor, float("inf")):
                best[neighbor] = new_cost
                heapq.heappush(
                    heap, (new_cost, path + [neighbor])
                )

    return None, float("inf")


def product_heuristic(graph, node, goal):
    return graph.get(node, {}).get(goal, 1.0)


def product_greedy(graph, start, goal):
    heap = [
        (product_heuristic(graph, start, goal), [start])
    ]
    seen = set()

    while heap:
        _, path = heapq.heappop(heap)
        node = path[-1]

        if node == goal:
            return path

        if node in seen:
            continue

        seen.add(node)

        for neighbor in graph.get(node, {}):
            if neighbor not in seen:
                heapq.heappush(
                    heap,
                    (
                        product_heuristic(
                            graph, neighbor, goal
                        ),
                        path + [neighbor]
                    )
                )

    return None


def product_astar(graph, start, goal):
    heap = [
        (
            product_heuristic(graph, start, goal),
            0.0,
            [start]
        )
    ]
    best = {start: 0.0}

    while heap:
        _, cost, path = heapq.heappop(heap)
        node = path[-1]

        if node == goal:
            return path, cost

        if cost > best.get(node, float("inf")):
            continue

        for neighbor, edge in graph.get(node, {}).items():
            new_cost = cost + edge

            if new_cost < best.get(neighbor, float("inf")):
                best[neighbor] = new_cost
                priority = (
                    new_cost
                    + product_heuristic(
                        graph, neighbor, goal
                    )
                )

                heapq.heappush(
                    heap,
                    (
                        priority,
                        new_cost,
                        path + [neighbor]
                    )
                )

    return None, float("inf")


def product_path_cost(graph, path):
    if not path or len(path) < 2:
        return 0.0

    return sum(
        graph[path[i]][path[i + 1]]
        for i in range(len(path) - 1)
    )


def single_product_recommendations(graph, product, n=5):
    return sorted(
        graph.get(product, {}).items(),
        key=lambda x: x[1]
    )[:n]


def shopping_recommendations(rules, cart, n=5):
    selected = set(cart)

    if rules.empty or not selected:
        return pd.DataFrame()

    candidates = {}

    for _, row in rules.iterrows():
        antecedents = set(row["antecedents"])
        consequents = set(row["consequents"])

        if not antecedents.issubset(selected):
            continue

        confidence = float(row["confidence"])
        lift = float(row["lift"])
        support = float(row["support"])
        score = confidence * lift

        for product in consequents:
            if product in selected:
                continue

            if (
                product not in candidates
                or score > candidates[product]["Score"]
            ):
                candidates[product] = {
                    "Product": product,
                    "Confidence": confidence,
                    "Lift": lift,
                    "Support": support,
                    "Score": score,
                    "Based On": ", ".join(
                        sorted(map(str, antecedents))
                    )
                }

    if not candidates:
        return pd.DataFrame()

    result = pd.DataFrame(candidates.values())
    result = (
        result
        .sort_values(
            ["Score", "Lift", "Confidence"],
            ascending=False
        )
        .head(int(n))
        .reset_index(drop=True)
    )
    result.insert(
        0, "Rank", range(1, len(result) + 1)
    )
    return result



page = st.sidebar.radio(

    "Select Section",

    [

        "Overview",

        "Customer Segmentation",

        "PCA",

        "Classification",

        "Regression",

        "Product Intelligence"

    ]

)





# =========================================================

# OVERVIEW

# =========================================================



if page == "Overview":



    st.header("Dataset Overview")



    col1, col2, col3, col4 = st.columns(4)



    col1.metric(

        "Raw Records",

        len(raw_df)

    )



    col2.metric(

        "Clean Transactions",

        len(df)

    )



    col3.metric(

        "Customers",

        df["CustomerID"].nunique()

    )



    col4.metric(

        "Products",

        df["Description"].nunique()

    )



    st.subheader("Data After Cleaning")



    st.write(

        "The dataset is cleaned by removing missing customer/date "

        "values, invalid quantities/prices and cancelled invoices."

    )



    st.dataframe(

        df.head(20),

        use_container_width=True

    )



    st.subheader("Customer Features")



    st.write(

        "Customer-level RFM-style features are created from "

        "the transaction data."

    )



    st.dataframe(

        customer.head(20),

        use_container_width=True

    )





# =========================================================

# CUSTOMER SEGMENTATION

# =========================================================



elif page == "Customer Segmentation":



    st.header("Customer Segmentation")



    features = [

        "Recency",

        "Frequency",

        "Monetary",

        "Products",

        "Quantity",

        "AverageOrderValue"

    ]



    X = np.log1p(

        customer[features]

    )



    X = StandardScaler().fit_transform(X)



    # =========================================================
    # FEATURE PAIR EXPLORER
    # =========================================================

    st.subheader("Feature Pair Explorer")
    st.write(
        "Select any two meaningful customer features to view their 2D relationship. "
        "No clustering is performed here. Use this graph to decide which feature pair "
        "looks most useful for customer segmentation."
    )

    pair_col1, pair_col2 = st.columns(2)

    with pair_col1:
        x_feature = st.selectbox(
            "X-axis Feature",
            features,
            index=features.index("Monetary"),
            key="seg_x_feature"
        )

    with pair_col2:
        y_options = [f for f in features if f != x_feature]
        default_y = "Frequency" if "Frequency" in y_options else y_options[0]
        y_feature = st.selectbox(
            "Y-axis Feature",
            y_options,
            index=y_options.index(default_y),
            key="seg_y_feature"
        )

    pair_df = customer[[x_feature, y_feature]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna().copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        pair_df[x_feature],
        pair_df[y_feature],
        s=24,
        alpha=0.55
    )
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.set_title(f"{x_feature} vs {y_feature}")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        f"Showing {len(pair_df):,} customers. Select different features to compare "
        "their natural 2D distribution. No K-Means or silhouette score is used in this section."
    )

    st.divider()

    st.subheader("Finding the Best Number of Clusters")



    # Elbow includes K=1. Silhouette starts at K=2 because K=1 is undefined.

    elbow_k_values = range(1, 11)
    silhouette_k_values = range(2, 11)

    inertia = []
    silhouettes = []

    for k in elbow_k_values:
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        model.fit(X)
        inertia.append(model.inertia_)

    for k in silhouette_k_values:
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        labels = model.fit_predict(X)
        silhouettes.append(silhouette_score(X, labels))

    best_k = list(silhouette_k_values)[int(np.argmax(silhouettes))]

    col1, col2 = st.columns(2)



    with col1:



        fig, ax = plt.subplots()



        ax.plot(

            list(elbow_k_values),

            inertia,

            marker="o"

        )



        ax.set_xlabel("Number of Clusters")

        ax.set_ylabel("Inertia")

        ax.set_title("Elbow Method (K=1 to 10)")



        st.pyplot(fig)



    with col2:



        fig, ax = plt.subplots()



        ax.plot(

            list(silhouette_k_values),

            silhouettes,

            marker="o"

        )



        ax.set_xlabel("Number of Clusters")

        ax.set_ylabel("Silhouette Score")

        ax.set_title("Silhouette Scores (K=2 to 10)")



        st.pyplot(fig)



    st.success(

        f"Best K-Means K = {best_k}"

    )



    st.metric(

        "Best Silhouette Score",

        round(max(silhouettes), 3)

    )



    st.caption(
        "K=1 is evaluated in the Elbow Method because inertia is defined for one cluster. "
        "Silhouette Score starts at K=2 because it is undefined for a single cluster."
    )

    # K-Means

    kmeans = KMeans(

        n_clusters=best_k,

        random_state=42,

        n_init=10

    )



    kmeans_labels = kmeans.fit_predict(X)



    # Hierarchical

    hierarchical = AgglomerativeClustering(

        n_clusters=best_k

    )



    hierarchical_labels = (

        hierarchical.fit_predict(X)

    )



    # GMM

    gmm = GaussianMixture(

        n_components=best_k,

        random_state=42

    )



    gmm_labels = gmm.fit_predict(X)



    st.subheader("Clustering Comparison")



    comparison = pd.DataFrame({



        "Algorithm": [

            "K-Means",

            "Hierarchical",

            "GMM"

        ],



        "Silhouette Score": [



            silhouette_score(

                X,

                kmeans_labels

            ),



            silhouette_score(

                X,

                hierarchical_labels

            ),



            silhouette_score(

                X,

                gmm_labels

            )

        ]

    })



    comparison["Silhouette Score"] = (

        comparison["Silhouette Score"]

        .round(3)

    )



    st.dataframe(

        comparison,

        use_container_width=True

    )





# =========================================================

# PCA

# =========================================================



elif page == "PCA":



    st.header("PCA Customer Visualization")



    features = [

        "Recency",

        "Frequency",

        "Monetary",

        "Products",

        "Quantity",

        "AverageOrderValue"

    ]



    X = np.log1p(

        customer[features]

    )



    X = StandardScaler().fit_transform(X)



    kmeans = KMeans(

        n_clusters=2,

        random_state=42,

        n_init=10

    )



    clusters = kmeans.fit_predict(X)



    pca = PCA(

        n_components=2

    )



    pca_data = pca.fit_transform(X)



    variance1 = (

        pca.explained_variance_ratio_[0]

        * 100

    )



    variance2 = (

        pca.explained_variance_ratio_[1]

        * 100

    )



    col1, col2 = st.columns(2)



    col1.metric(

        "Principal Components",

        2

    )



    col2.metric(

        "Variance Explained",

        f"{variance1 + variance2:.2f}%"

    )



    st.write(

        f"PC1 explains {variance1:.2f}% of the variance."

    )



    st.write(

        f"PC2 explains {variance2:.2f}% of the variance."

    )



    fig, ax = plt.subplots()



    scatter = ax.scatter(

        pca_data[:, 0],

        pca_data[:, 1],

        c=clusters,

        s=15

    )



    ax.set_xlabel("PC1")

    ax.set_ylabel("PC2")

    ax.set_title(

        "Customer Segments using PCA"

    )



    st.pyplot(fig)





# =========================================================

# CLASSIFICATION

# =========================================================



elif page == "Classification":



    st.header(

        "Customer High-Value Classification"

    )



    st.write(

        "The model predicts whether a customer belongs "

        "to the high-value group."

    )



    split_date = df["InvoiceDate"].quantile(

        0.80

    )



    past = df[

        df["InvoiceDate"] <= split_date

    ].copy()



    future = df[

        df["InvoiceDate"] > split_date

    ].copy()



    past_customer = past.groupby(

        "CustomerID"

    ).agg(



        LastPurchase=(

            "InvoiceDate",

            "max"

        ),



        Frequency=(

            "Invoice",

            "nunique"

        ),



        Monetary=(

            "Amount",

            "sum"

        ),



        Products=(

            "Description",

            "nunique"

        ),



        Quantity=(

            "Quantity",

            "sum"

        )

    )



    past_customer["Recency"] = (

        split_date -

        past_customer["LastPurchase"]

    ).dt.days



    past_customer.drop(

        columns="LastPurchase",

        inplace=True

    )



    past_customer["AverageOrderValue"] = (

        past_customer["Monetary"] /

        past_customer["Frequency"]

    )



    future_spending = (

        future

        .groupby("CustomerID")["Amount"]

        .sum()

    )



    data = past_customer.join(

        future_spending.rename(

            "FutureSpending"

        ),

        how="inner"

    ).dropna()



    threshold = data[

        "FutureSpending"

    ].quantile(0.75)



    data["HighValue"] = (

        data["FutureSpending"] >= threshold

    ).astype(int)



    features = [

        "Recency",

        "Frequency",

        "Monetary",

        "Products",

        "Quantity",

        "AverageOrderValue"

    ]



    X = data[features]

    y = data["HighValue"]



    X_train, X_test, y_train, y_test = (

        train_test_split(

            X,

            y,

            test_size=0.20,

            random_state=42,

            stratify=y

        )

    )



    models = {



        "Logistic Regression":

            LogisticRegression(

                max_iter=1000

            ),



        "KNN":

            KNeighborsClassifier(

                n_neighbors=5

            ),



        "SVM":

            SVC(

                probability=True,

                random_state=42

            ),



        "Random Forest":

            RandomForestClassifier(

                n_estimators=100,

                random_state=42

            )

    }



    results = []



    for name, model in models.items():



        pipeline = Pipeline([



            (

                "scale",

                StandardScaler()

            ),



            (

                "model",

                model

            )

        ])



        pipeline.fit(

            X_train,

            y_train

        )



        predictions = pipeline.predict(

            X_test

        )



        probabilities = (

            pipeline.predict_proba(

                X_test

            )[:, 1]

        )



        results.append({



            "Model": name,



            "Accuracy":

                accuracy_score(

                    y_test,

                    predictions

                ),



            "Precision":

                precision_score(

                    y_test,

                    predictions,

                    zero_division=0

                ),



            "Recall":

                recall_score(

                    y_test,

                    predictions,

                    zero_division=0

                ),



            "F1":

                f1_score(

                    y_test,

                    predictions,

                    zero_division=0

                ),



            "AUC":

                roc_auc_score(

                    y_test,

                    probabilities

                )

        })



    results_df = pd.DataFrame(

        results

    )



    for col in [

        "Accuracy",

        "Precision",

        "Recall",

        "F1",

        "AUC"

    ]:

        results_df[col] = (

            results_df[col]

            .round(3)

        )



    st.subheader("Model Performance")



    st.dataframe(

        results_df,

        use_container_width=True

    )



    best_model = results_df.loc[

        results_df["F1"].idxmax()

    ]



    st.success(

        f"Best model based on F1 Score: "

        f"{best_model['Model']}"

    )



    selected_model = st.selectbox(

        "View Confusion Matrix",

        list(models.keys())

    )



    model = models[selected_model]



    pipeline = Pipeline([



        (

            "scale",

            StandardScaler()

        ),



        (

            "model",

            model

        )

    ])



    pipeline.fit(

        X_train,

        y_train

    )



    predictions = pipeline.predict(

        X_test

    )



    matrix = confusion_matrix(

        y_test,

        predictions

    )



    st.subheader(

        f"{selected_model} Confusion Matrix"

    )



    st.write(matrix)





# =========================================================

# REGRESSION

# =========================================================



elif page == "Regression":



    st.header(

        "Customer Spending Prediction"

    )



    st.write(

        "The regression models predict future customer spending."

    )



    split_date = df["InvoiceDate"].quantile(

        0.80

    )



    past = df[

        df["InvoiceDate"] <= split_date

    ].copy()



    future = df[

        df["InvoiceDate"] > split_date

    ].copy()



    past_customer = past.groupby(

        "CustomerID"

    ).agg(



        LastPurchase=(

            "InvoiceDate",

            "max"

        ),



        Frequency=(

            "Invoice",

            "nunique"

        ),



        Monetary=(

            "Amount",

            "sum"

        ),



        Products=(

            "Description",

            "nunique"

        ),



        Quantity=(

            "Quantity",

            "sum"

        )

    )



    past_customer["Recency"] = (

        split_date -

        past_customer["LastPurchase"]

    ).dt.days



    past_customer.drop(

        columns="LastPurchase",

        inplace=True

    )



    past_customer["AverageOrderValue"] = (

        past_customer["Monetary"] /

        past_customer["Frequency"]

    )



    future_spending = (

        future

        .groupby("CustomerID")["Amount"]

        .sum()

    )



    data = past_customer.join(

        future_spending.rename(

            "FutureSpending"

        ),

        how="inner"

    ).dropna()



    features = [

        "Recency",

        "Frequency",

        "Monetary",

        "Products",

        "Quantity",

        "AverageOrderValue"

    ]



    X = data[features]

    y = data["FutureSpending"]



    X_train, X_test, y_train, y_test = (

        train_test_split(

            X,

            y,

            test_size=0.20,

            random_state=42

        )

    )



    models = {



        "Linear Regression":

            LinearRegression(),



        "Random Forest Regressor":

            RandomForestRegressor(

                n_estimators=100,

                random_state=42

            )

    }



    results = []



    for name, model in models.items():



        pipeline = Pipeline([



            (

                "scale",

                StandardScaler()

            ),



            (

                "model",

                model

            )

        ])



        pipeline.fit(

            X_train,

            y_train

        )



        predictions = pipeline.predict(

            X_test

        )



        mse = mean_squared_error(

            y_test,

            predictions

        )



        results.append({



            "Model": name,



            "MAE":

                mean_absolute_error(

                    y_test,

                    predictions

                ),



            "MSE":

                mse,



            "RMSE":

                np.sqrt(mse),



            "R2":

                r2_score(

                    y_test,

                    predictions

                )

        })



    results_df = pd.DataFrame(

        results

    )



    for col in [

        "MAE",

        "MSE",

        "RMSE",

        "R2"

    ]:

        results_df[col] = (

            results_df[col]

            .round(3)

        )



    st.subheader(

        "Regression Model Performance"

    )



    st.dataframe(

        results_df,

        use_container_width=True

    )



    best = results_df.loc[

        results_df["R2"].idxmax()

    ]



    st.success(

        f"Best model based on R²: "

        f"{best['Model']}"

    )






elif page == "Product Intelligence":

    st.header("Product Intelligence & Online Shopping")
    st.write(
        "Apriori association rules, product graph search and "
        "an interactive online-shopping recommendation demo."
    )

    min_support = st.sidebar.slider(
        "Minimum Support",
        0.005, 0.20, 0.02, 0.005,
        key="product_min_support"
    )

    min_confidence = st.sidebar.slider(
        "Minimum Confidence",
        0.10, 1.00, 0.30, 0.05,
        key="product_min_confidence"
    )

    top_n = st.sidebar.slider(
        "Top Rules",
        5, 100, 20, 5,
        key="product_top_rules"
    )

    try:
        basket = create_product_basket(df)

        with st.spinner(
            "Running Apriori and generating product associations..."
        ):
            frequent, rules = run_product_apriori(
                basket,
                min_support,
                min_confidence
            )

    except Exception as e:
        st.error(
            f"Product Intelligence could not be initialized: {e}"
        )
        st.stop()

    if basket.empty:
        st.error("No valid product transactions were found.")
        st.stop()

    graph = build_product_graph(rules)
    graph_nodes = list(graph.keys())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", len(basket))
    c2.metric("Products", len(basket.columns))
    c3.metric("Frequent Itemsets", len(frequent))
    c4.metric("Association Rules", len(rules))

    st.divider()

    # =====================================================
    # ONLINE SHOPPING DEMO
    # =====================================================

    st.subheader("🛒 Online Shopping Demo")
    st.write(
        "Select products currently in your cart. "
        "The system recommends what you should buy next."
    )

    if rules.empty:
        st.warning(
            "No association rules were generated. "
            "Lower Minimum Support or Minimum Confidence."
        )
    else:
        shopping_products = sorted(
            set(basket.columns).intersection(graph_nodes),
            key=str
        )

        if not shopping_products:
            st.warning(
                "No products are available in the association graph."
            )
        else:
            cart = st.multiselect(
                "Select products currently in your cart",
                shopping_products,
                placeholder="Choose one or more products...",
                key="shopping_cart"
            )

            recommendation_limit = st.slider(
                "Number of products to recommend",
                min_value=1,
                max_value=100,
                value=10,
                step=1,
                key="shopping_recommendation_limit"
            )

            if not cart:
                st.info(
                    "Select one or more products to simulate "
                    "an online shopping cart."
                )
            else:
                st.markdown("### Your Cart")

                for product in cart:
                    st.write(f"✓ {product}")

                recommendations = shopping_recommendations(
                    rules,
                    cart,
                    recommendation_limit
                )

                st.markdown("### Recommended Next Items")

                if recommendations.empty:
                    st.info(
                        "No recommendation was found for this cart. "
                        "Try another item or lower the Apriori thresholds."
                    )
                else:
                    for _, row in recommendations.head(10).iterrows():
                        a, b, c, d = st.columns(
                            [0.08, 0.46, 0.20, 0.26]
                        )

                        a.write(f"**#{int(row['Rank'])}**")
                        b.write(f"**{row['Product']}**")
                        c.write(f"Lift: **{row['Lift']:.2f}**")
                        d.write(
                            f"Confidence: "
                            f"**{row['Confidence']:.1%}**"
                        )

                    display = recommendations.copy()
                    display["Confidence"] = display[
                        "Confidence"
                    ].map(lambda x: f"{x:.1%}")
                    display["Support"] = display[
                        "Support"
                    ].map(lambda x: f"{x:.2%}")
                    display["Lift"] = display[
                        "Lift"
                    ].map(lambda x: f"{x:.2f}")
                    display["Score"] = display[
                        "Score"
                    ].map(lambda x: f"{x:.3f}")

                    st.dataframe(
                        display[
                            [
                                "Rank",
                                "Product",
                                "Based On",
                                "Confidence",
                                "Lift",
                                "Support",
                                "Score"
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True
                    )

    st.divider()

    # =====================================================
    # APRIORI RULES
    # =====================================================

    st.subheader("Apriori Association Rules")

    if rules.empty:
        st.warning("No association rules are available.")
    else:
        display_rules = rules.head(top_n).copy()

        display_rules["antecedents"] = display_rules[
            "antecedents"
        ].apply(
            lambda x: ", ".join(sorted(map(str, x)))
        )

        display_rules["consequents"] = display_rules[
            "consequents"
        ].apply(
            lambda x: ", ".join(sorted(map(str, x)))
        )

        display_rules = display_rules[
            [
                "antecedents",
                "consequents",
                "support",
                "confidence",
                "lift"
            ]
        ]

        display_rules.columns = [
            "Antecedents",
            "Consequents",
            "Support",
            "Confidence",
            "Lift"
        ]

        st.dataframe(
            display_rules,
            use_container_width=True,
            hide_index=True
        )

        chart = display_rules.head(10).copy()
        chart["Rule"] = (
            chart["Antecedents"]
            + " → "
            + chart["Consequents"]
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(
            chart["Rule"].iloc[::-1],
            chart["Lift"].iloc[::-1]
        )
        ax.set_xlabel("Lift")
        ax.set_title("Top Association Rules by Lift")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # =====================================================
    # PRODUCT GRAPH
    # =====================================================

    st.subheader("Product Association Graph")

    g1, g2 = st.columns(2)
    g1.metric("Graph Nodes", len(graph_nodes))
    g2.metric(
        "Graph Edges",
        sum(len(v) for v in graph.values()) // 2
    )

    if graph_nodes:
        rows = []

        for product, neighbors in graph.items():
            for neighbor, cost in neighbors.items():
                if str(product) <= str(neighbor):
                    rows.append(
                        {
                            "Product A": product,
                            "Product B": neighbor,
                            "Cost": round(cost, 4),
                            "Lift": round(1 / cost, 4)
                        }
                    )

        if rows:
            st.dataframe(
                pd.DataFrame(rows)
                .sort_values("Lift", ascending=False)
                .head(50),
                use_container_width=True,
                hide_index=True
            )

    # =====================================================
    # SEARCH ALGORITHMS
    # =====================================================

    st.subheader("Product Graph Search Algorithms")

    if len(graph_nodes) < 2:
        st.warning(
            "At least two connected products are required."
        )
    else:
        s1, s2 = st.columns(2)

        start = s1.selectbox(
            "Start Product",
            graph_nodes,
            key="product_start"
        )

        goals = [
            x for x in graph_nodes
            if x != start
        ]

        goal = s2.selectbox(
            "Goal Product",
            goals,
            key="product_goal"
        )

        if st.button(
            "Run All Search Algorithms",
            type="primary",
            use_container_width=True,
            key="run_product_search"
        ):
            paths = {
                "BFS": (
                    product_bfs(graph, start, goal),
                    None
                ),
                "DFS": (
                    product_dfs(graph, start, goal),
                    None
                ),
            }

            ucs_path, ucs_cost = product_ucs(
                graph, start, goal
            )
            greedy_path = product_greedy(
                graph, start, goal
            )
            astar_path, astar_cost = product_astar(
                graph, start, goal
            )

            paths["UCS"] = (ucs_path, ucs_cost)
            paths["Greedy"] = (greedy_path, None)
            paths["A*"] = (astar_path, astar_cost)

            rows = []

            for algorithm, (path, known_cost) in paths.items():
                cost = (
                    known_cost
                    if known_cost is not None
                    else (
                        product_path_cost(graph, path)
                        if path
                        else None
                    )
                )

                rows.append(
                    {
                        "Algorithm": algorithm,
                        "Path": (
                            " → ".join(map(str, path))
                            if path else "No path"
                        ),
                        "Steps": (
                            len(path) - 1
                            if path else None
                        ),
                        "Cost": (
                            round(cost, 4)
                            if cost is not None
                            else None
                        )
                    }
                )

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True
            )

            tabs = st.tabs(
                ["BFS", "DFS", "UCS", "Greedy", "A*"]
            )

            for tab, algorithm in zip(
                tabs,
                ["BFS", "DFS", "UCS", "Greedy", "A*"]
            ):
                path, known_cost = paths[algorithm]

                with tab:
                    if path:
                        cost = (
                            known_cost
                            if known_cost is not None
                            else product_path_cost(
                                graph, path
                            )
                        )

                        st.success(
                            f"{algorithm} found a path with "
                            f"{len(path) - 1} steps."
                        )
                        st.write(
                            " → ".join(map(str, path))
                        )
                        st.metric(
                            "Path Cost",
                            round(cost, 4)
                        )
                    else:
                        st.error(
                            f"{algorithm} could not find a path."
                        )

    # =====================================================
    # SINGLE PRODUCT RECOMMENDATIONS
    # =====================================================

    st.subheader("Single Product Recommendations")

    if graph_nodes:
        selected_product = st.selectbox(
            "Select a product",
            graph_nodes,
            key="single_product"
        )

        count = st.slider(
            "Number of recommendations",
            1,
            100,
            5,
            1,
            key="single_product_count"
        )

        recommendations = single_product_recommendations(
            graph,
            selected_product,
            count
        )

        if recommendations:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Rank": i,
                            "Recommended Product": product,
                            "Edge Cost": round(cost, 4),
                            "Association Strength (Lift)": round(
                                1 / cost, 4
                            )
                        }
                        for i, (product, cost)
                        in enumerate(
                            recommendations,
                            start=1
                        )
                    ]
                ),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info(
                "No recommendations are available for this product."
            )



# FOOTER

# =========================================================



st.sidebar.markdown("---")



st.sidebar.write(

    "Smart Retail Intelligence System"

)



st.sidebar.write(

    "Machine Learning + AI Search"

)