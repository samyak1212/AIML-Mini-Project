import streamlit as st

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



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



page = st.sidebar.radio(

    "Select Section",

    [

        "Overview",

        "Customer Segmentation",

        "PCA",

        "Classification",

        "Regression"

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



    st.subheader("Finding the Best Number of Clusters")



    k_values = range(2, 7)



    inertia = []

    silhouettes = []



    for k in k_values:



        model = KMeans(

            n_clusters=k,

            random_state=42,

            n_init=10

        )



        labels = model.fit_predict(X)



        inertia.append(

            model.inertia_

        )



        silhouettes.append(

            silhouette_score(

                X,

                labels

            )

        )



    best_k = list(k_values)[

        int(np.argmax(silhouettes))

    ]



    col1, col2 = st.columns(2)



    with col1:



        fig, ax = plt.subplots()



        ax.plot(

            list(k_values),

            inertia,

            marker="o"

        )



        ax.set_xlabel("Number of Clusters")

        ax.set_ylabel("Inertia")

        ax.set_title("Elbow Method")



        st.pyplot(fig)



    with col2:



        fig, ax = plt.subplots()



        ax.plot(

            list(k_values),

            silhouettes,

            marker="o"

        )



        ax.set_xlabel("Number of Clusters")

        ax.set_ylabel("Silhouette Score")

        ax.set_title("Silhouette Scores")



        st.pyplot(fig)



    st.success(

        f"Best K-Means K = {best_k}"

    )



    st.metric(

        "Best Silhouette Score",

        round(max(silhouettes), 3)

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





# =========================================================

# FOOTER

# =========================================================



st.sidebar.markdown("---")



st.sidebar.write(

    "Smart Retail Intelligence System"

)



st.sidebar.write(

    "Machine Learning + AI Search"

)