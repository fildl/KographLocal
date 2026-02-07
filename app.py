import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys
try:
    from numbers_parser import Document
    NUMBERS_AVAILABLE = True
except ImportError:
    NUMBERS_AVAILABLE = False

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
    combined_data = kindle_data
    audio_books_path = 'data/audio_books.csv'
    
    if os.path.exists(audio_books_path):
        combined_data = processor.get_data_with_audio_books(audio_books_path, current_combined_df=combined_data)
        
    # --- Metadata & Paper Books from Numbers ---
    metadata_path = '/Users/filippodiludovico/Library/Mobile Documents/com~apple~Numbers/Documents/reading.numbers'
    metadata_df = None
    
    if NUMBERS_AVAILABLE and os.path.exists(metadata_path):
        # 1. Load Paper Books from Numbers
        combined_data = processor.get_paper_books_from_numbers(metadata_path, current_combined_df=combined_data)
        
        # 2. Enrich with Metadata (Country, Purchase Date)
        combined_data, metadata_df = processor.get_data_with_metadata(metadata_path, current_combined_df=combined_data)
        
    return kindle_data, combined_data, metadata_df

try:
    with st.spinner('Loading reading data...'):
        kindle_df, combined_df, metadata_raw = load_data()
        
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

current_year = pd.Timestamp.now().year
default_index = 0
if current_year in years:
    default_index = years.index(current_year)

selected_year = st.sidebar.selectbox("Select Year", years, index=default_index)

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

# Streak calculations
streaks = viz._calculate_streaks(filtered_combined)
longest_streak = max(streaks) if streaks else 0
current_streak = streaks[-1] if streaks else 0

# Daily Average Calculation (Minutes)
total_minutes = filtered_combined['duration'].sum() / 60
if not filtered_combined.empty:
    min_date = filtered_combined['date'].min()
    max_date = filtered_combined['date'].max()
    
    if selected_year != "All Time":
        # For a specific year, use 365/366 days if the year is over, or days so far if current
        # Actually simplest is just max - min + 1 of the specific filtered data?
        # If I filter 2024, and have data from Jan 1 to Dec 31, it's 366.
        # If I filter 2024, and have data only from Feb 1 to Feb 5, it's 5 days.
        # "Daily Average" usually implies "Average over the whole period".
        # Let's use the full year days if "All Time" is NOT selected to represent "Yearly Pace"?
        # Or just (max-min) which represents "Active Period Average".
        # User request "daily average" usually implies "How much I read on average".
        # Using (max-min) is safer.
        days_span = (max_date - min_date).days + 1
    else:
        days_span = (max_date - min_date).days + 1
else:
    days_span = 1

daily_average = total_minutes / days_span if days_span > 0 else 0

# Create two rows of metrics
c1, c2, c3 = st.columns(3)
c1.metric("Total Hours", f"{total_hours:.1f}h")
c2.metric("Books Read", f"{books_read}")
c3.metric("Sessions", f"{total_sessions}")

c4, c5, c6 = st.columns(3)
c4.metric("Daily Average", f"{daily_average:.0f}m")
c5.metric("Longest Streak", f"{longest_streak} days")
c6.metric("Current Streak", f"{current_streak} days")

# --- Books Purchased Metric ---
if metadata_raw is not None and 'purchase_date' in metadata_raw.columns:
    # Use the RAW metadata for purchase counts
    purchase_df = metadata_raw
    
    if selected_year != "All Time":
        y = int(selected_year)
        purchased_count = purchase_df[purchase_df['purchase_date'].dt.year == y].shape[0]
        label = f"Purchased in {selected_year}"
    else:
        # Only count items that HAVE a purchase date
        purchased_count = purchase_df['purchase_date'].notna().sum()
        label = "Total Library (Purchased)"
        
    st.metric(label, f"{purchased_count}")
elif 'purchase_date' in filtered_combined.columns:
    # Fallback to matched data if raw metadata unavailable (shouldn't happen if NUMBERS_AVAILABLE)
    if selected_year != "All Time":
        purchased_count = filtered_combined[filtered_combined['purchase_date'].dt.year == int(selected_year)]['title'].nunique()
        label = f"Purchased in {selected_year} (Read)"
    else:
        purchased_count = filtered_combined.loc[filtered_combined['purchase_date'].notna(), 'title'].nunique()
        label = "Total Purchased (Read)"
        
    st.metric(label, f"{purchased_count}")

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

try:
    fig_dist = viz.plot_reading_distribution(year=plot_year)
    if fig_dist:
        st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.info("No distribution data available.")
except Exception as e:
    st.error(f"Could not render Reading Distribution: {e}")

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

# --- 4. Reading Habits (Patterns) ---
st.subheader("Reading Habits")
try:
    fig_patterns = viz.plot_reading_patterns(year=plot_year)
    if fig_patterns:
        st.plotly_chart(fig_patterns, use_container_width=True)
    else:
        st.info("No pattern data available.")
except Exception as e:
    st.error(f"Could not render Reading Patterns: {e}")

# --- 5. Reading Streaks ---
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

# --- 7. Cumulative Pages ---
st.subheader("Cumulative Pages")
try:
    fig_pages = viz.plot_cumulative_pages(year=plot_year)
    if fig_pages:
        st.plotly_chart(fig_pages, use_container_width=True)
    else:
        st.info("No page data available (Audiobooks may not have page counts).")
except Exception as e:
    st.error(f"Could not render Cumulative Pages: {e}")
# --- 8. Library Insights (Optional) ---
# Only show if relevant metadata exists
try:
    if 'author_country' in filtered_combined.columns or 'purchase_date' in filtered_combined.columns:
        st.subheader("Library Insights")
        
        col_lib1, col_lib2 = st.columns(2)
        
        with col_lib1:
            try:
                fig_country = viz.plot_country_distribution(year=plot_year)
                if fig_country:
                    st.plotly_chart(fig_country, use_container_width=True)
                else:
                     pass 
            except Exception as e:
                st.e(f"Error rendering Country Dist: {e}")
                
        with col_lib2:
            try:
                fig_purchase = viz.plot_purchase_timeline(year=plot_year)
                if fig_purchase:
                    st.plotly_chart(fig_purchase, use_container_width=True)
                else:
                    pass
            except Exception as e:
                st.error(f"Error rendering Purchase Timeline: {e}")
except Exception as e:
    pass
