# Smart Retail Intelligence

An end-to-end AI/ML retail analytics application built with **Python**, **scikit-learn**, and **Streamlit**. It applies unsupervised machine learning algorithms (K-Means Clustering and Association Rule Learning) strictly from the AI/ML syllabus to analyze transaction data, segment customer purchasing behavior, analyze product demand, and provide live product recommendations.

---

## Dataset

This project uses the **Online Retail II** transaction dataset:
```text
data/online_retail_II-Year 2009-2010.csv
```

### Data Preprocessing & Feature Engineering
1. **Cleaning:** Removes cancellations and invalid entries by filtering for `Quantity > 0`, `Price > 0`, non-null `Customer ID`, valid `Invoice`, and valid `InvoiceDate`.
2. **Customer Feature Engineering:**
   * **Average Quantity per Line:** $\frac{\text{Total Quantity}}{\text{Transaction Lines}}$
   * **Spend per Unique Product:** $\frac{\text{Total Revenue}}{\text{Unique Products Purchased}}$
3. **2% Tail Clipping:** Extreme values in both features are capped between the 2nd and 98th percentiles to handle outliers without deleting customer records.
4. **StandardScaler:** Standardizes features before calculating Euclidean distances.
5. **K-Means Clustering ($K=3$):** Partitions customers into 3 behavioral clusters with a Silhouette Score of **$\approx 0.744$**.
6. **Behavior-Based Cluster Labels:**
   * **Regular Customers:** Lower spending and purchasing quantity.
   * **Mid-Value High-Quantity Buyers:** Higher quantity and spending than regular customers.
   * **High-Value / Heavy Buyers:** Very high purchasing volume and expenditure.

---

## Application Features & Pages

The Streamlit dashboard contains 4 dedicated pages accessible from the sidebar:

1. **Dataset Information:**
   * Raw dataset preview directly in the starting section.
   * Dataset-level summary metrics (records, unique invoices, unique products, raw customer count, missing values, date range).
   * Column schema metadata and descriptive statistics (mean, std, min, percentiles, max) for numerical fields.
   * Purely dataset-only information with zero segmentation leaks.

2. **Customer Segmentation:**
   * Key customer metrics and Silhouette score evaluation explanation.
   * 2D interactive scatter plot of the clusters with simple thin cross markers (`+`) at the centroids.
   * Commercial business analysis table aggregated by customer segment.
   * Single everyday sales line chart across the entire timeframe and full daily records table.

3. **Product Analysis:**
   * Dropdown selector for customer segments.
   * Single Top 10 products table ranked by total quantity purchased for the selected segment.
   * Automatic concentrated bulk purchasing detection (high volume with $\le 10\%$ customer reach).

4. **Product Recommendations & Customer Classification Demo:**
   * **Interactive Order Simulator:** Choose any catalog item and specify order quantity.
   * **Live Customer Classification:** Standardizes the simulated order and classifies the customer into one of the 3 K-Means clusters in real time with an explanation.
   * **Frequently Bought Together Recommendations:** Employs Association Rule Learning (basket co-occurrence) to recommend top complementary products.

---

## Project Structure

```text
Mini Project/
├── data/
│   └── online_retail_II-Year 2009-2010.csv   # Source dataset
├── src/
│   ├── main.py                               # Application entry point & navigation
│   ├── data_preprocessing/
│   │   └── clean_data.py                     # CSV loading & transaction cleaning
│   ├── customer/
│   │   └── customer_analysis.py              # Feature engineering, K-Means (K=3), summaries
│   ├── product/
│   │   └── product_analysis.py               # Product metrics & Top 10 segment table
│   ├── recommendation/
│   │   └── recommendation_engine.py          # Association rule recommendations
│   └── ui/
│       ├── dataset_page.py                   # Dataset Information UI
│       ├── customer_page.py                  # Customer Segmentation UI
│       ├── product_page.py                   # Product Analysis UI
│       └── recommendation_page.py            # Demo simulator, recommendations & classifier UI
├── tests/                                    # Scratch/notebook exploration files
├── requirements.txt                          # Project Python dependencies
└── readme.md                                 # Project documentation
```

---

## Requirements

The project dependencies in `requirements.txt`:

```text
numpy
pandas
matplotlib
scikit-learn
mlxtend
jupyter
ipykernel
streamlit
```

### Dependency Breakdown:
* **`streamlit`**: Interactive web dashboard and UI components.
* **`pandas`** & **`numpy`**: Data manipulation, aggregation, and mathematical operations.
* **`scikit-learn`**: `KMeans`, `StandardScaler`, and `silhouette_score`.
* **`mlxtend`**: Association rule and frequent pattern utilities.
* **`matplotlib`**: Plotting support for notebook analyses.
* **`jupyter`** & **`ipykernel`**: Running Jupyter notebooks in `tests/`.

---

## Step-by-Step: How to Run

### Step 1: Open the Project Directory
Open PowerShell or your preferred terminal in the project root:
```powershell
cd "c:\Users\91935\OneDrive\Documents\AIML\Mini Project"
```

### Step 2: Create and Activate Virtual Environment
If you do not already have a virtual environment set up:
```powershell
# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# (Or on Windows Command Prompt: .venv\Scripts\activate.bat)
```

> **Note for PowerShell execution policy:** If script execution is restricted, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
> ```

### Step 3: Install Required Dependencies
Install all packages from `requirements.txt`:
```powershell
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application
Launch the app from the project root:
```powershell
streamlit run src/main.py
```

The application will start and open automatically in your browser at `http://localhost:8501`.
