import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.db import DatabaseManager
from src.processing import DataProcessor
from src.visuals import Visualizer

# --- Configuration ---
st.set_page_config(
    page_title="Kograph Local",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Kograph Local: Reading Statistics Dashboard"
    }
)

# --- Theme colors to match Visualizer ---
# We can't easily override Streamlit's full theme from here without .streamlit/config.toml,
# but we can ensure our Plotly charts use the correct template.

# --- Data Loading ---
@st.cache_data
def load_data():
    """
    Load and process data from the database. 
    Cached to prevent reloading on every interaction.
    """
    db = DatabaseManager()
    raw_data, books = db.get_raw_data()
    
    processor = DataProcessor(raw_data, books)
    
    # Base Data (Kindle)
    kindle_data = processor.process()
    
    # Combined Data (Paper + Kindle)
    paper_books_path = 'data/paper_books.csv'
    if os.path.exists(paper_books_path):
        combined_data = processor.get_data_with_paper_books(paper_books_path)
    else:
        combined_data = kindle_data
        
    return kindle_data, combined_data

try:
    with st.spinner('Loading reading data...'):
        kindle_df, combined_df = load_data()
        
    # Initialize Visualizers
    viz = Visualizer(kindle_df)
    viz_timeline = Visualizer(combined_df)
    
    st.success(f"Data Loaded Successfully! Found {len(kindle_df)} sessions.")
    
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- Sidebar ---
st.sidebar.title("Kograph Local")
st.sidebar.markdown("---")

# Standard Sidebar Filters
years = sorted(kindle_df['year'].unique().tolist(), reverse=True)
years.insert(0, "All Time")

selected_year = st.sidebar.selectbox("Select Year", years)

# --- Main Content ---
st.title("📚 Reading Dashboard")

if selected_year == "All Time":
    st.write("Displaying data for **All Time**")
else:
    st.write(f"Displaying data for **{selected_year}**")

# Debug / Verification
with st.expander("Raw Data Preview"):
    st.dataframe(kindle_df.head())
