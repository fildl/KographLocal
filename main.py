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
    raw_data, books = db.get_raw_data()
    
    # Process
    print("Processing data...")
    processor = DataProcessor(raw_data, books)
    
    # 1. Base Data (Kindle Only)
    kindle_data = processor.process()
    
    # 2. Combined Data (Kindle + Paper) for Timeline
    paper_books_path = 'data/paper_books.csv'
    if os.path.exists(paper_books_path):
        print(f"Found paper books data at {paper_books_path}")
        combined_data = processor.get_data_with_paper_books(paper_books_path)
    else:
        combined_data = kindle_data
    
    # Initialize Visualizers
    viz = Visualizer(kindle_data)            # For statistics and standard charts
    viz_timeline = Visualizer(combined_data) # For Book Timeline only
    
    # 2. Key Years Identification (from Kindle data usually, or Combined? 
    # Let's use Combined years so we don't miss paper-only years if any, 
    # but strictly user asked paper books ONLY for timeline. 
    # So general charts should probably strictly follow Kindle years? 
    # But if I read a paper book in 2024 and no kindle book, do I want charts for 2024? 
    # User said "paper_books.csv deve essere usato solo per questo grafico timeline". 
    # So other charts should NOT see paper books. Thus they should follow Kindle years.)
    years = sorted(kindle_data['year'].unique().tolist(), reverse=True)
    print(f"Found years (Kindle): {years}")
    
    # 3. Weekly Activity (Supports All Time + Per Year)
    print("\n[Weekly Activity]")
    # All Time
    fig_all = viz.plot_weekly_activity()
    if fig_all:
        save_plot(fig_all, 'weekly_activity.html', 'all_time')
        
    # Per Year
    for year in years:
        fig = viz.plot_weekly_activity(year=year)
        if fig:
            save_plot(fig, 'weekly_activity.html', str(year))

    # 4. Reading Calendar
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

    # 6. Streaks & Metrics
    print("\n[Streaks & Metrics]")
    # All Time
    fig_all = viz.plot_streaks()
    if fig_all:
        save_plot(fig_all, 'streaks.html', 'all_time')
        
    # Per Year
    for year in years:
        fig = viz.plot_streaks(year=year)
        if fig:
            save_plot(fig, 'streaks.html', str(year))

    # 7. Streak Calendar
    print("\n[Streak Calendar]")
    for year in years:
        fig = viz.plot_streak_calendar(year=year)
        if fig:
            save_plot(fig, 'streak_calendar.html', str(year))

    # 8. Book Timeline (Using Combined Data)
    print("\n[Book Timeline]")
    
    # Identify years for timeline (can be different if paper books extend range)
    timeline_years = sorted(combined_data['year'].unique().tolist(), reverse=True)
    
    # All Time
    fig_all = viz_timeline.plot_book_timeline()
    if fig_all:
        save_plot(fig_all, 'timeline.html', 'all_time')
        
    # Per Year
    for year in timeline_years: # Iterate ALL years found in combined data
        fig = viz_timeline.plot_book_timeline(year=year)
        if fig:
            save_plot(fig, 'timeline.html', str(year))

    # 9. Daily Pattern
    print("\n[Daily Pattern]")
    # All Time
    fig_all = viz.plot_daily_pattern()
    if fig_all:
        save_plot(fig_all, 'daily_pattern.html', 'all_time')
    
    # Per Year
    for year in years:
        fig = viz.plot_daily_pattern(year=year)
        if fig:
            save_plot(fig, 'daily_pattern.html', str(year))

    # 10. Monthly Pattern
    print("\n[Monthly Pattern]")
    # All Time
    fig_all = viz.plot_monthly_pattern()
    if fig_all:
        save_plot(fig_all, 'monthly_pattern.html', 'all_time')
    
    # Per Year
    for year in years:
        fig = viz.plot_monthly_pattern(year=year)
        if fig:
            save_plot(fig, 'monthly_pattern.html', str(year))

    # 11. Session Duration Analysis
    print("\n[Session Duration]")
    # All Time
    fig_all = viz.plot_session_duration()
    if fig_all:
        save_plot(fig_all, 'session_duration.html', 'all_time')
    
    # Per Year
    for year in years:
        fig = viz.plot_session_duration(year=year)
        if fig:
            save_plot(fig, 'session_duration.html', str(year))

    print("\n--- Generation Complete ---")

if __name__ == "__main__":
    main()
