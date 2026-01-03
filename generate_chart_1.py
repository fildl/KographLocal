from src.db import DatabaseManager
from src.processing import DataProcessor
from src.visuals import Visualizer
import os

def main():
    print("Generating Weekly Activity Chart...")
    
    # Load Data
    db = DatabaseManager()
    page_stats, books = db.get_raw_data()
    
    processor = DataProcessor(page_stats, books)
    df = processor.process()
    
    # Helper to save with directory structure
    def save_plot(fig, filename, year_folder):
        folder_path = f'output/{year_folder}'
        os.makedirs(folder_path, exist_ok=True)
        path = f'{folder_path}/{filename}'
        fig.write_html(path)
        print(f"✓ Chart saved to: {path}")

    # Visualize All Time
    viz = Visualizer(df)
    fig_all = viz.plot_weekly_activity()
    save_plot(fig_all, 'weekly_activity.html', 'all_time')

    # Visualize Current Year (Automatic)
    import datetime
    current_year = datetime.datetime.now().year
    fig_current = viz.plot_weekly_activity(year=current_year)
    if fig_current:
        save_plot(fig_current, 'weekly_activity.html', str(current_year))

    # Visualize Fixed 2025 (For checking)
    fig_2025 = viz.plot_weekly_activity(year=2025)
    if fig_2025:
        save_plot(fig_2025, 'weekly_activity.html', '2025')

if __name__ == "__main__":
    main()
