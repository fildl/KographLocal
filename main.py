from src.db import DatabaseManager
from src.processing import DataProcessor
from src.visuals import Visualizer
from src.report import DashboardGenerator
import os
import shutil
import plotly.graph_objects as go

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
    
    years = sorted(kindle_data['year'].unique().tolist(), reverse=True)
    print(f"Found years (Kindle): {years}")
    
    # --- Initialize Dashboard Generators ---
    generators = {}
    
    # All Time Generator
    generators['all_time'] = DashboardGenerator(viz.THEME_COLORS)
    
    # Per Year Generators
    for year in years:
        generators[str(year)] = DashboardGenerator(viz.THEME_COLORS)
        
    
    # --- Feature Generation Loop ---
    
    # Helper to add to generator
    def process_plot(fig, name, year_key, responsive=False):
        if fig:
            save_plot(fig, f'{name}.html', year_key)
            
            # If dashboard needs responsive width (like timeline), modify a copy
            if responsive and year_key in generators:
                fig_dash = go.Figure(fig)
                fig_dash.update_layout(width=1200)
                generators[year_key].add_plot(name, fig_dash)
            elif year_key in generators:
                generators[year_key].add_plot(name, fig)

    # 3. Weekly Activity
    print("\n[Weekly Activity]")
    process_plot(viz.plot_weekly_activity(), 'weekly_activity', 'all_time')
    for year in years:
        process_plot(viz.plot_weekly_activity(year=year), 'weekly_activity', str(year))

    # 4. Reading Calendar
    print("\n[Reading Calendar]")
    for year in years:
        process_plot(viz.plot_reading_calendar(year=year), 'reading_calendar', str(year))

    # 5. Time of Day Distribution
    print("\n[Time of Day]")
    process_plot(viz.plot_time_of_day(), 'time_of_day', 'all_time')
    for year in years:
        process_plot(viz.plot_time_of_day(year=year), 'time_of_day', str(year))

    # 6. Streaks & Metrics
    print("\n[Streaks & Metrics]")
    process_plot(viz.plot_streaks(), 'streaks', 'all_time')
    for year in years:
        process_plot(viz.plot_streaks(year=year), 'streaks', str(year))

    # 7. Streak Calendar
    print("\n[Streak Calendar]")
    for year in years:
        process_plot(viz.plot_streak_calendar(year=year), 'streak_calendar', str(year))

    # 8. Book Timeline
    print("\n[Book Timeline]")
    timeline_years = sorted(combined_data['year'].unique().tolist(), reverse=True)
    process_plot(viz_timeline.plot_book_timeline(), 'timeline', 'all_time', responsive=True)
    for year in timeline_years:
        year_str = str(year)
        if year_str not in generators:
            generators[year_str] = DashboardGenerator(viz.THEME_COLORS)
            
        process_plot(viz_timeline.plot_book_timeline(year=year), 'timeline', year_str, responsive=True)


    # 9. Daily Pattern
    print("\n[Daily Pattern]")
    process_plot(viz.plot_daily_pattern(), 'daily_pattern', 'all_time')
    for year in years:
        process_plot(viz.plot_daily_pattern(year=year), 'daily_pattern', str(year))

    # 10. Monthly Pattern
    print("\n[Monthly Pattern]")
    process_plot(viz.plot_monthly_pattern(), 'monthly_pattern', 'all_time')
    for year in years:
        process_plot(viz.plot_monthly_pattern(year=year), 'monthly_pattern', str(year))

    # 11. Session Duration Analysis
    print("\n[Session Duration]")
    process_plot(viz.plot_session_duration(), 'session_duration', 'all_time')
    for year in years:
        process_plot(viz.plot_session_duration(year=year), 'session_duration', str(year))

    # --- Generate Dashboards ---
    print("\n[Generating Dashboards]")
    for key, generator in generators.items():
        filename = 'dashboard.html'
        path = f'output/{key}/{filename}'
        if key == 'all_time':
            title_year = "All Time"
        else:
            title_year = key
            
        generator.generate(title_year, path)

    print("\n--- Generation Complete ---")

if __name__ == "__main__":
    main()
