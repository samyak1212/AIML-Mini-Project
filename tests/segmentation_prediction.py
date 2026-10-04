"""Smart Retail Intelligence - Part 1
Customer segmentation, classification, regression and PCA.
Dataset: data/online_retail.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix, roc_auc_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor



# 1. Load and clean the dataset.
df = pd.read_csv("data/online_retail_II-Year 2009-2010.csv")
df.columns = [c.strip() for c in df.columns]
rename = {}
for c in df.columns:
    k = c.lower().replace(" ", "").replace("_", "")
    if k == "invoiceno": rename[c] = "Invoice"
    elif k == "description": rename[c] = "Description"
    elif k == "quantity": rename[c] = "Quantity"
    elif k in ("invoicedate", "date"): rename[c] = "InvoiceDate"
    elif k in ("unitprice", "price"): rename[c] = "UnitPrice"
    elif k in ("customerid", "customer"): rename[c] = "CustomerID"
    elif k == "country": rename[c] = "Country"
df = df.rename(columns=rename)

needed = ["Invoice", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
df = df.dropna(subset=["CustomerID", "InvoiceDate", "Quantity", "UnitPrice"])
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
df = df[~df["Invoice"].astype(str).str.startswith("C")].copy()
df["Amount"] = df["Quantity"] * df["UnitPrice"]

print("Transactions:", len(df))
print("Customers:", df.CustomerID.nunique())

# 2. Create RFM/customer features.
ref = df.InvoiceDate.max() + pd.Timedelta(days=1)
customer = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (ref - x.max()).days),
    Frequency=("Invoice", "nunique"),
    Monetary=("Amount", "sum"),
    Products=("Description", "nunique"),
    Quantity=("Quantity", "sum")
)
customer["AverageOrderValue"] = customer.Monetary / customer.Frequency
features = ["Recency", "Frequency", "Monetary", "Products", "Quantity", "AverageOrderValue"]
X = np.log1p(customer[features])
X = StandardScaler().fit_transform(X)

# 3. K-Means: choose k using silhouette score and show the elbow curve.
inertia, sil = [], []
for k in range(2, 7):
    m = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = m.fit_predict(X)
    inertia.append(m.inertia_)
    sil.append(silhouette_score(X, labels))
best_k = range(2, 7)[int(np.argmax(sil))]
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
customer["KMeans"] = kmeans.fit_predict(X)

plt.plot(range(2, 7), inertia, marker="o")
plt.xlabel("Clusters"); plt.ylabel("Inertia"); plt.title("K-Means Elbow Method")
plt.show()
print("K-Means k:", best_k, "Silhouette:", round(max(sil), 3))

# 4. Compare hierarchical clustering and GMM.
h = AgglomerativeClustering(n_clusters=best_k)
g = GaussianMixture(n_components=best_k, random_state=42)
customer["Hierarchical"] = h.fit_predict(X)
customer["GMM"] = g.fit_predict(X)
print("Hierarchical silhouette:", round(silhouette_score(X, customer.Hierarchical), 3))
print("GMM silhouette:", round(silhouette_score(X, customer.GMM), 3))

# 5. PCA gives a simple 2-D visualization of customer segments.
pca = PCA(n_components=2)
p = pca.fit_transform(X)
plt.scatter(p[:, 0], p[:, 1], c=customer.KMeans, s=12)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Customer Segments - PCA")
plt.show()

# 6. Classification: predict whether a customer is high value.
customer["HighValue"] = (customer.Monetary >= customer.Monetary.quantile(.75)).astype(int)
Xc, yc = customer[features], customer.HighValue
Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=.2, random_state=42, stratify=yc)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(5),
    "SVM": SVC(probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}
print("\nCLASSIFICATION")
for name, model in models.items():
    pipe = Pipeline([("scale", StandardScaler()), ("model", model)])
    pipe.fit(Xtr, ytr); pred = pipe.predict(Xte); prob = pipe.predict_proba(Xte)[:, 1]
    print(name)
    print(" Accuracy:", round(accuracy_score(yte, pred), 3),
          "Precision:", round(precision_score(yte, pred), 3),
          "Recall:", round(recall_score(yte, pred), 3),
          "F1:", round(f1_score(yte, pred), 3),
          "AUC:", round(roc_auc_score(yte, prob), 3))
    print(" Confusion matrix:\n", confusion_matrix(yte, pred))

# 7. Hyperparameter tuning + cross-validation for Random Forest.
pipe = Pipeline([("scale", StandardScaler()), ("model", RandomForestClassifier(random_state=42))])
params = {"model__n_estimators": [50, 100], "model__max_depth": [None, 10]}
grid = GridSearchCV(pipe, params, cv=5, scoring="f1", n_jobs=-1)
grid.fit(Xtr, ytr)
print("Best RF parameters:", grid.best_params_)
print("Best CV F1:", round(grid.best_score_, 3))

# 8. Regression: predict spending in the later 20% of the time period.
split = df.InvoiceDate.quantile(.8)
past, future = df[df.InvoiceDate <= split], df[df.InvoiceDate > split]
past_c = past.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (split - x.max()).days),
    Frequency=("Invoice", "nunique"), Monetary=("Amount", "sum"),
    Products=("Description", "nunique"), Quantity=("Quantity", "sum")
)
past_c["AverageOrderValue"] = past_c.Monetary / past_c.Frequency
future_spend = future.groupby("CustomerID").Amount.sum().rename("FutureSpending")
reg = past_c.join(future_spend).replace([np.inf, -np.inf], np.nan).dropna()
Xr, yr = reg[features], reg.FutureSpending
Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=.2, random_state=42)
reg_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
}
print("\nREGRESSION")
for name, model in reg_models.items():
    pipe = Pipeline([("scale", StandardScaler()), ("model", model)])
    pipe.fit(Xtr, ytr); pred = pipe.predict(Xte)
    mse = mean_squared_error(yte, pred)
    print(name, "MAE:", round(mean_absolute_error(yte, pred), 2),
          "MSE:", round(mse, 2), "RMSE:", round(np.sqrt(mse), 2),
          "R2:", round(r2_score(yte, pred), 3))

# 9. Save cleaned data for the recommendation/search program.
customer.to_csv("data/customer_features.csv")
df[["Invoice", "Description", "Quantity"]].to_csv("data/retail_transactions_clean.csv", index=False)
print("\nDone. Saved customer_features.csv and retail_transactions_clean.csv")
