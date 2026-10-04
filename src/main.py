import importlib
from pathlib import Path
import streamlit as st

# Import project modules
import customer.customer_analysis
import data_preprocessing.clean_data
import product.product_analysis
import recommendation.recommendation_engine
import ui.customer_page
import ui.dataset_page
import ui.product_page
import ui.recommendation_page

# Reload modules so live Streamlit runs always use the latest code
importlib.reload(data_preprocessing.clean_data)
importlib.reload(customer.customer_analysis)
importlib.reload(product.product_analysis)
importlib.reload(recommendation.recommendation_engine)
importlib.reload(ui.dataset_page)
importlib.reload(ui.customer_page)
importlib.reload(ui.product_page)
importlib.reload(ui.recommendation_page)

from customer.customer_analysis import run_customer_clustering
from data_preprocessing.clean_data import load_raw_data
from ui.customer_page import show_customer_page
from ui.dataset_page import show_dataset_page
from ui.product_page import show_product_page
from ui.recommendation_page import show_recommendation_page


# Resolve CSV data path so it works from either the project root or src folder.
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_FILE = BASE_DIR / "data" / "online_retail_II-Year 2009-2010.csv"
if not CSV_FILE.exists():
    CSV_FILE = Path("data/online_retail_II-Year 2009-2010.csv").resolve()


# Load the raw dataset from CSV and cache it.
@st.cache_data(show_spinner="Loading raw dataset...")
def get_raw_dataset(file_path):
    return load_raw_data(file_path)


# Run the customer clustering pipeline and cache the results.
@st.cache_data(show_spinner="Running customer segmentation (K-Means K=3)...")
def get_customer_analysis(file_path):
    return run_customer_clustering(file_path)


# Main function to configure Streamlit app and navigate between pages.
def main():
    st.set_page_config(page_title="Smart Retail Intelligence", layout="wide")

    file_path_str = str(CSV_FILE)

    # Load raw data and clustering results
    raw_transactions = get_raw_dataset(file_path_str)
    analysis = get_customer_analysis(file_path_str)

    # In case of an outdated cache object, re-run cleanly
    if not isinstance(analysis, dict) or "transactions" not in analysis:
        st.cache_data.clear()
        analysis = run_customer_clustering(file_path_str)

    transactions = analysis["transactions"]
    customers = analysis["customers"]

    st.title("Smart Retail Intelligence")
    st.caption("Customer clustering, business analysis, and product intelligence.")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio(
        "Select Page",
        [
            "Dataset Information",
            "Customer Segmentation",
            "Product Analysis",
            "Product Recommendations",
        ],
    )

    # Page routing
    if selected_page == "Dataset Information":
        show_dataset_page(raw_transactions)
    elif selected_page == "Customer Segmentation":
        show_customer_page(transactions, customers, analysis)
    elif selected_page == "Product Analysis":
        show_product_page(transactions, customers, analysis)
    else:
        show_recommendation_page(transactions, customers, analysis)


if __name__ == "__main__":
    main()
