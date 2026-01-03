from src.db import DatabaseManager
from src.processing import DataProcessor
from src.visuals import Visualizer
import os
import shutil

def save_plot(fig, filename, year_folder):
    """Saves a Plotly figure to the specified folder with consistent naming."""
    folder_path = f'output/{year_folder}'
    os.makedirs(folder_path, exist_ok=True)
    path = f'{folder_path}/{filename}'
    fig.write_html(path)
    print(f"  ✓ Saved: {path}")

def main():
    print("--- Kograph Local Generator ---")
    
    # 1. Load & Process Data
    print("Loading data...")
    db = DatabaseManager()
    page_stats, books = db.get_raw_data()
    
    processor = DataProcessor(page_stats, books)
    df = processor.process()
    viz = Visualizer(df)
    
    # 2. Key Years Identification
    years = sorted(df['year'].unique().tolist(), reverse=True)
    print(f"Found years: {years}")
    
    # 3. Weekly Activity (Supports All Time + Per Year)
    print("\n[Weekly Activity]")
    # All Time
    fig_all = viz.plot_weekly_activity() # default is all time
    if fig_all:
        save_plot(fig_all, 'weekly_activity.html', 'all_time')
    
    # Per Year
    for year in years:
        fig = viz.plot_weekly_activity(year=year)
        if fig:
            save_plot(fig, 'weekly_activity.html', str(year))
            
    # 4. Reading Calendar (Years Only)
    print("\n[Reading Calendar]")
    for year in years:
        fig = viz.plot_reading_calendar(year=year)
        if fig:
            save_plot(fig, 'reading_calendar.html', str(year))

    # 5. Time of Day Distribution
    print("\n[Time of Day]")
    # All Time
    fig_all = viz.plot_time_of_day()
    if fig_all:
        save_plot(fig_all, 'time_of_day.html', 'all_time')
    
    # Per Year
    for year in years:
        fig = viz.plot_time_of_day(year=year)
        if fig:
            save_plot(fig, 'time_of_day.html', str(year))

    print("\n--- Generation Complete ---")

if __name__ == "__main__":
    main()
