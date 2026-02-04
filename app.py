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
    
    # Combined Data (Paper + Kindle + Audio)
    paper_books_path = 'data/paper_books.csv'
    audio_books_path = 'data/audio_books.csv'
    
    combined_data = kindle_data
    
    if os.path.exists(paper_books_path):
        combined_data = processor.get_data_with_paper_books(paper_books_path)
    
    if os.path.exists(audio_books_path):
        # Pass the current combined_df (which might already have paper books)
        combined_data = processor.get_data_with_audio_books(audio_books_path, current_combined_df=combined_data)
        
    return kindle_data, combined_data

try:
    with st.spinner('Loading reading data...'):
        kindle_df, combined_df = load_data()
        
    # Initialize Visualizers
    viz = Visualizer(kindle_df)
    viz_timeline = Visualizer(combined_df)
    
# --- Sidebar Filters ---
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()
    
# --- Sidebar Filters ---
st.sidebar.title("Kograph Local")
st.sidebar.markdown("---")

# 1. Year Filter
years = sorted(kindle_df['year'].unique().tolist(), reverse=True)
years.insert(0, "All Time")
selected_year = st.sidebar.selectbox("Select Year", years)

# 2. Format Filter
formats = ['Ebook', 'Paperback', 'Audiobook']
selected_formats = st.sidebar.multiselect("Format", formats, default=formats)

# --- Filter Data ---
filtered_df = kindle_df.copy()

# Year Filtering
if selected_year != "All Time":
    filtered_df = filtered_df[filtered_df['year'] == selected_year]
    # combined_df filtering (for timeline)
    filtered_combined = combined_df[combined_df['year'] == selected_year]
else:
    filtered_combined = combined_df.copy()

# Format Filtering (Note: `kindle_df` is primarily Kindle data, but logic might expand)
# Ideally `kindle_df` should have a 'format' column if we want to filter mixed sources there, 
# but currently `kindle_df` implies digital. `combined_df` has explicit formats.
# For now, we apply format filtering mainly to the Timeline data which supports both.
if selected_formats:
    # Normalize selection to lowercase for matching
    fmt_filter = [f.lower() for f in selected_formats]
    if 'format' in filtered_combined.columns:
        filtered_combined = filtered_combined[filtered_combined['format'].isin(fmt_filter)]

# Re-Initialize Visualizers with Filtered Data
viz = Visualizer(filtered_df)
viz_timeline = Visualizer(filtered_combined)

# --- Main Dashboard ---
st.title("📚 Reading Dashboard")

# Construct Filter Status Message
filter_status = f"**Year:** {selected_year}"

if len(selected_formats) == len(formats):
    fmt_status = "All Formats"
elif len(selected_formats) == 0:
    fmt_status = "None"
else:
    fmt_status = ", ".join(selected_formats)

filter_status += f" • **Formats:** {fmt_status}"

st.markdown(f"Displaying data for: {filter_status}")

# --- Metrics Row ---
col1, col2, col3, col4 = st.columns(4)

total_hours = filtered_df['duration'].sum() / 3600
books_read = filtered_df['id_book'].nunique()
# Estimate pages (simple heuristic or from data if available, falling back to books)
# For now, let's show sessions count
total_sessions = len(filtered_df)

# Longest streak calculation
streaks = viz._calculate_streaks(filtered_df)
longest_streak = max(streaks) if streaks else 0
current_streak = streaks[-1] if streaks else 0

col1.metric("Total Hours", f"{total_hours:.1f}h")
col2.metric("Books Read", f"{books_read}")
col3.metric("Sessions", f"{total_sessions}")
col4.metric("Longest Streak", f"{longest_streak} days")

st.markdown("---")

# --- 1. Weekly Activity ---
st.subheader("Weekly Activity")
# Determine year for plot function (int or None)
plot_year = int(selected_year) if selected_year != "All Time" else None

fig_weekly = viz.plot_weekly_activity() # Visualizer already has filtered data? 
# Wait, Visualizer was re-inited with `filtered_df`. 
# If `filtered_df` is ALREADY filtered by year, we should pass year=None to functions 
# OR rely on the fact that `filtered_df` contains only that year's data.
# The `plot_weekly_activity` method has a `year` argument that does its own filtering.
# If we pass `year=None`, it uses all data in `self.data`.
# Since `self.data` is now `filtered_df`, `year=None` is correct and safer.
# HOWEVER, some charts like the Calendar Grid rely on knowing the specific year to draw the grid.
# If `self.data` is filtered, we still might need to tell the plot function "this is 2024" for titles/structure.
# Let's pass `year=plot_year` if it's set, otherwise None.
# But if `filtered_df` is already filtered, `plot_weekly_activity(year=2024)` might try to filter again.
# Let's check `visuals.py` implementation quickly. It does: `df = df[df['year'] == year]`
# If `df` is already filtered, this is redundant but harmless.
try:
    fig_weekly = viz.plot_weekly_activity(year=plot_year)
    if fig_weekly:
        st.plotly_chart(fig_weekly, use_container_width=True)
    else:
        st.info("No data available for this period.")
except Exception as e:
    st.error(f"Could not render Weekly Activity: {e}")

# --- 2. Reading Calendar ---
st.subheader("Reading Calendar")
if selected_year == "All Time":
    st.info("Please select a specific Year in the sidebar to view the Reading Calendar grid.")
else:
    try:
        # The calendar plot NEEDS a year to render the grid properly.
        # If "All Time" is selected, it defaults to the latest year in the code.
        # We should probably show the latest year if All Time is selected, or handle it gracefully.
        fig_calendar = viz.plot_reading_calendar(year=plot_year)
        if fig_calendar:
            st.plotly_chart(fig_calendar, use_container_width=True)
        else:
            st.info("No data available.")
    except Exception as e:
        st.error(f"Could not render Reading Calendar: {e}")

# --- 3. Book Timeline (The Complex One) ---
st.subheader("Book Timeline")
try:
    fig_timeline_plot = viz_timeline.plot_book_timeline(year=plot_year)
    if fig_timeline_plot:
        st.plotly_chart(fig_timeline_plot, use_container_width=True)
    else:
         st.info("No timeline data available.")
except Exception as e:
    st.error(f"Could not render Timeline: {e}")

