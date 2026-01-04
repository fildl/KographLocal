# Kograph Local

Kograph Local is a Python-based generator that visualizes your **Koreader reading statistics**  and **Paper Books history**. It generates an interactive HTML dashboard with detailed analytics, timelines, and trends.

## Features

-   **Dashboard Generation**: Automatically builds a `dashboard.html` for "All Time" and each specific year.
-   **Multi-Source Data**:
    -   **Koreader**: Automatically syncs with `data/statistics.sqlite3` from your connected device.
    -   **Paper Books**: Integrates `data/paper_books.csv` to track physical reading history.
-   **Interactive Visualizations** (Plotly):
    -   **Timeline**: Gantt-style view of all books read, with accurate start/end dates.
    -   **Weekly Activity**: Heatmap-style bar chart of reading hours per day.
    -   **Reading Calendar**: Monthly calendar highlighting reading days and intensity.
    -   **Streaks**: Analysis of reading streaks (Current, Longest, Total Days).
    -   **Time of Day**: Radar/Area chart showing when you read most.
    -   **Sessions**: Average session duration analysis by weekday and month.
    -   **Cumulative Growth**: Stacked area chart showing total pages read over time (Kindle vs Paper).
-   **Responsive Design**: Dark-themed dashboard optimized for desktop viewing.

## Setup

1.  **Requirements**:
    -   Python 3.9+
    -   Install dependencies:
        ```bash
        pip install -r requirements.txt
        ```
        *(pandas, plotly, numpy)*

2.  **Data Setup**:
    -   **Kindle**: Connect your Kindle. The script looks for `statistics.sqlite3` in `/Volumes/Kindle/koreader/settings/` and copies it to `data/`.
    -   **Paper Books**: Create `data/paper_books.csv` with the following columns:
        ```csv
        title, authors, pages, start_date, end_date, language
        The Great Gatsby, F. Scott Fitzgerald, 180, 2024-01-01, 2024-01-10, en
        ```

## Usage

Simply run the main script:

```bash
python main.py
```

It will:
1.  Load data from `data/statistics.sqlite3` (updating from Kindle if connected).
2.  Load paper books from `data/paper_books.csv`.
3.  Process reading sessions and calculate metrics.
4.  Generate HTML plots and Dashboards in the `output/` folder:
    -   `output/all_time/dashboard.html`
    -   `output/2026/dashboard.html`
    -   `output/2025/dashboard.html`
    -   ...

## Customization

-   **Theme**: Edit `THEME_COLORS` in `src/visuals.py` to change the color palette (default: Dark Mode with Cyan/Pink/Yellow accents).
-   **Session Threshold**: Adjust `gap_minutes` in `src/processing.py` (default: 5 min) to define how breaks are handled.

## Project Structure

-   `src/db.py`: Database connection and file syncing.
-   `src/processing.py`: Data cleaning, sessionization, and metric calculation.
-   `src/visuals.py`: Plotly chart generation (Visualizer class).
-   `src/report.py`: HTML Dashboard assembly (DashboardGenerator class).
-   `main.py`: Entry point and orchestration.

## Notes

-   **Accuracy**: Proper `start_date` and `end_date` in `paper_books.csv` are crucial for the timeline and calendar accuracy.
-   **Koreader Bugs**: The script automatically filters out "ghost" reading sessions (< 5 mins) and fixes zero-duration bugs common in Koreader stats.
