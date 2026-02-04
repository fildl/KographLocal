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
# Use combined data for main visualizer too, so all charts get all formats
viz = Visualizer(filtered_combined)
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

total_hours = filtered_combined['duration'].sum() / 3600
books_read = filtered_combined['id_book'].nunique()
# Estimate pages (simple heuristic or from data if available, falling back to books)
# For now, let's show sessions count
total_sessions = len(filtered_combined)

# Longest streak calculation
streaks = viz._calculate_streaks(filtered_combined)
longest_streak = max(streaks) if streaks else 0
current_streak = streaks[-1] if streaks else 0

col1.metric("Total Hours", f"{total_hours:.1f}h")
col2.metric("Books Read", f"{books_read}")
col3.metric("Sessions", f"{total_sessions}")
col4.metric("Longest Streak", f"{longest_streak} days")

st.markdown("---")

# Determine year for plot function (int or None)
plot_year = int(selected_year) if selected_year != "All Time" else None

# --- 1. Book Timeline ---
st.subheader("Book Timeline")
try:
    fig_timeline_plot = viz_timeline.plot_book_timeline(year=plot_year)
    if fig_timeline_plot:
        st.plotly_chart(fig_timeline_plot, use_container_width=True)
    else:
         st.info("No timeline data available.")
except Exception as e:
    st.error(f"Could not render Timeline: {e}")

# --- 2. Activity Patterns ---
st.subheader("Activity Patterns")

try:
    fig_weekly = viz.plot_weekly_activity(year=plot_year)
    if fig_weekly:
        st.plotly_chart(fig_weekly, use_container_width=True)
    else:
        st.info("No data available for this period.")
except Exception as e:
    st.error(f"Could not render Weekly Activity: {e}")

try:
    fig_hourly = viz.plot_time_of_day(year=plot_year)
    if fig_hourly:
        st.plotly_chart(fig_hourly, use_container_width=True)
    else:
         st.info("No data available.")
except Exception as e:
    st.error(f"Could not render Time of Day: {e}")

# --- 3. Reading Calendar ---
st.subheader("Reading Calendar")
if selected_year == "All Time":
    st.info("Please select a specific Year in the sidebar to view the Reading Calendar grid.")
else:
    try:
        fig_calendar = viz.plot_reading_calendar(year=plot_year)
        if fig_calendar:
            st.plotly_chart(fig_calendar, use_container_width=True)
        else:
            st.info("No data available.")
    except Exception as e:
        st.error(f"Could not render Reading Calendar: {e}")


# --- 5. Reading Habits (Streaks) ---
st.subheader("Reading Streaks")
# --- 5b. Streak Histogram (Distribution) ---
# Shown first or second? Usually Calendar is larger. 
# User asked "one below the other". 
# Let's keep the order: Calendar then Histogram.

try:
    # 5b. Streak Histogram (Distribution)
    fig_streak_hist = viz.plot_streaks(year=plot_year)
    if fig_streak_hist:
        st.plotly_chart(fig_streak_hist, use_container_width=True)
    else:
        st.info("No streak data available.")
except Exception as e:
    st.error(f"Could not render Streak Histogram: {e}")

try:
    # 5a. Streak Calendar (3x4 Grid)
    if selected_year == "All Time":
         st.info("Select a specific year to view daily streak calendar.")
    else:
        fig_streak_cal = viz.plot_streak_calendar(year=plot_year)
        if fig_streak_cal:
            st.plotly_chart(fig_streak_cal, use_container_width=True)
        else:
            st.info("No streak data for this year.")
except Exception as e:
    st.error(f"Could not render Streak Calendar: {e}")

# --- 6. Books Completed ---
st.subheader("Books Completed")
try:
    fig_completed = viz.plot_books_completed(year=plot_year)
    if fig_completed:
        st.plotly_chart(fig_completed, use_container_width=True)
    else:
        st.info("No books completed in this period.")
except Exception as e:
    st.error(f"Could not render Books Completed: {e}")