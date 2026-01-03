import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as mdates
from scipy.interpolate import make_interp_spline
from matplotlib.patches import FancyBboxPatch

class KindlePlots:
    def __init__(self, reading_data):
        self.reading_data = reading_data

        font_path = '/Library/Fonts/SF-Pro-Text-Regular.otf'
        font_manager.fontManager.addfont(font_path)
    
    # Style configuration matching your workout plots
    STYLE_CONFIG = {
        'figure_size': (16, 10),
        'bar_width': 0.8,
        'bar_alpha': 0.9,
        'bar_edge_color': 'white',
        'bar_edge_width': 0.5,
        'title_fontsize': 20,
        'title_fontweight': 'bold',
        'title_pad': 25,
        'xlabel_fontsize': 14,
        'xlabel_pad': 15,
        'ylabel_fontsize': 14,
        'ylabel_pad': 15,
        'tick_labelsize': 12,
        'legend_title_fontsize': 14,
        'legend_fontsize': 12,
        'total_text_fontsize': 11,
        'total_text_fontweight': 'bold',
        'total_text_color': 'white',
        'total_text_offset_ratio': 0.01,
        'dpi': 300,
        'font_family': 'SF Pro Text',
        'title_font_family': None,    # Can override for titles
        'text_font_family': None      # Can override for text elements
    }

    THEME_CONFIG = {
        'style': 'whitegrid',
        'rc': {
            "axes.facecolor": "#1c1c1c",
            "figure.facecolor": "#1c1c1c",
            "grid.color": "#444444",
            "grid.linestyle": "--",
            "text.color": "white",
            "axes.labelcolor": "lightgray",
            "xtick.color": "lightgray",
            "ytick.color": "lightgray",
        }
    }

    # Book colors for consistent visualization
    BOOK_COLORS = [
        '#ff6b9d', '#feca57', '#48dbfb', '#0abde3', '#ee5a6f',
        '#ff9ff3', '#54a0ff', '#5f27cd', '#00d2d3', '#ff9f43',
        '#1dd1a1', '#576574', '#c44569', '#f8b500', '#6c5ce7'
    ]

    PLOTS_COLORS = [
        '#ff6b9d', '#feca57', '#0abde3', '#ee5a6f', '#ff9ff3', 
        '#54a0ff', '#00d2d3', '#ff9f43', '#1dd1a1', '#10ac84', 
        '#2e86de', '#341f97', '#c44569', '#f8b500', '#6c5ce7', '#222f3e', '#8395a7'
    ]

    def create_weekly_reading_time(self, year=None, format_filter='kindle', output_dir='output', figsize=None):
        """Create a histogram showing weekly total reading time.
        If year is specified, only shows data for that year.
        
        edit this function:
        * rename it create_weekly_reading_time
        * if the function is called without year specific, do a line instead of histogram
        * if it is called with year specific, do not change the plot
        """
        if self.reading_data is None or self.reading_data.empty:
            print("Cannot generate plot: Reading data is empty.")
            return
        
        sns.set_theme(**self.THEME_CONFIG)
        Path(output_dir).mkdir(exist_ok=True)
        
        if figsize is None:
            figsize = self.STYLE_CONFIG['figure_size']
        
        # Prepare data
        df = self.reading_data.copy()

        # Filter by format if specified
        if format_filter:
            df = df[df['format'] == format_filter]
            if df.empty:
                print(f"No {format_filter} reading data found.")
                return
        
        # Filter by year if specified
        if year is not None:
            df = df[df['start_datetime'].dt.year == year].copy()
            if df.empty:
                print(f"No reading data found for year {year}.")
                return
            output_subdir = f'{output_dir}/{year}'
            filename = f'reading_time_weekly_{year}.png'
        else:
            output_subdir = f'{output_dir}/overall'
            filename = 'reading_time_weekly.png'

        Path(output_subdir).mkdir(parents=True, exist_ok=True)
        
        df['week'] = df['start_datetime'].dt.to_period('W')
        
        # Calculate weekly reading time in minutes
        weekly_reading = df.groupby('week')['duration'].sum().reset_index()
        weekly_reading['duration_minutes'] = weekly_reading['duration'] / 60
        
        # Create labels based on year filter
        weekly_reading['week_timestamp'] = weekly_reading['week'].dt.to_timestamp()
        
        if year is not None:
            # For specific year, show only month names at first week of each month
            weekly_reading['month'] = weekly_reading['week_timestamp'].dt.month
            weekly_reading['week_labels'] = ''
            
            # Mark first week of each month
            for month in weekly_reading['month'].unique():
                first_week_idx = weekly_reading[weekly_reading['month'] == month].index[0]
                month_name = weekly_reading.loc[first_week_idx, 'week_timestamp'].strftime('%B')
                weekly_reading.loc[first_week_idx, 'week_labels'] = month_name
        else:
            # For all years, show YYYY-MM at first week of each month
            weekly_reading['year_month'] = weekly_reading['week_timestamp'].dt.strftime('%Y-%m')
            weekly_reading['week_labels'] = ''
            
            # Mark first week of each month
            for year_month in weekly_reading['year_month'].unique():
                first_week_idx = weekly_reading[weekly_reading['year_month'] == year_month].index[0]
                weekly_reading.loc[first_week_idx, 'week_labels'] = year_month
        
        weekly_reading['week'] = weekly_reading['week_timestamp']
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        if year is not None:
            # Histogram for specific year
            ax.bar(range(len(weekly_reading)), weekly_reading['duration_minutes'],
                width=0.8,
                color=self.PLOTS_COLORS[0],
                alpha=self.STYLE_CONFIG['bar_alpha'],
                edgecolor=self.STYLE_CONFIG['bar_edge_color'],
                linewidth=self.STYLE_CONFIG['bar_edge_width'])
        else:
            # Line plot for all years
            ax.plot(weekly_reading['week'], weekly_reading['duration_minutes'],
                color=self.PLOTS_COLORS[0],
                linewidth=2,
                marker='o',
                markersize=4,
                alpha=0.8)
            ax.fill_between(weekly_reading['week'], weekly_reading['duration_minutes'],
                alpha=0.3,
                color=self.PLOTS_COLORS[0])
            ax.set_ylim(bottom=0)
            ax.set_xlim(left=weekly_reading['week'].min(), right=weekly_reading['week'].max())
        
        title = 'Weekly Reading Time'
        if year is not None:
            title += f' ({year})'
        
        ax.set_title(
            title,
            fontfamily=self.STYLE_CONFIG['font_family'],
            fontsize=self.STYLE_CONFIG['title_fontsize'],
            fontweight=self.STYLE_CONFIG['title_fontweight'],
            pad=self.STYLE_CONFIG['title_pad']
        )
        ax.set_xlabel(
            'Week',
            fontfamily=self.STYLE_CONFIG['font_family'],
            fontsize=self.STYLE_CONFIG['xlabel_fontsize'],
            labelpad=self.STYLE_CONFIG['xlabel_pad']
        )
        ax.set_ylabel(
            'Reading Time (minutes)',
            fontfamily=self.STYLE_CONFIG['font_family'],
            fontsize=self.STYLE_CONFIG['ylabel_fontsize'],
            labelpad=self.STYLE_CONFIG['ylabel_pad']
        )
        
        # Set tick parameters
        ax.tick_params(axis='both', which='major', labelsize=self.STYLE_CONFIG['tick_labelsize'])
        ax.tick_params(axis='x', rotation=0)

        if year is not None:
            # Histogram x-axis
            ax.set_xticks(range(len(weekly_reading)))
            ax.set_xticklabels(weekly_reading['week_labels'])
        else:
            # Line plot x-axis
            # Show labels for first week of each month
            label_positions = []
            label_texts = []
            for i, row in weekly_reading.iterrows():
                if row['week_labels']:
                    label_positions.append(row['week'])
                    label_texts.append(row['week_labels'])
            ax.set_xticks(label_positions)
            ax.set_xticklabels(label_texts)
        
        # Grid styling
        ax.grid(axis='y', alpha=1.0)
        ax.grid(axis='x', alpha=0)
        ax.set_axisbelow(True)
        
        # Add total text annotations
        if year is not None:
            # Text on top of bars for histogram
            totals = weekly_reading['duration_minutes']
            y_offset = totals.max() * self.STYLE_CONFIG['total_text_offset_ratio']
            for i, total in enumerate(totals):
                if total > 0:
                    ax.text(
                        i,
                        total + y_offset,
                        f'{total:.0f}m',
                        ha='center',
                        va='bottom',
                        fontfamily=self.STYLE_CONFIG['font_family'],
                        fontsize=self.STYLE_CONFIG['total_text_fontsize'],
                        fontweight=self.STYLE_CONFIG['total_text_fontweight'],
                        color=self.STYLE_CONFIG['total_text_color']
                    )
        
        plt.tight_layout()
        
        save_path = Path(output_subdir) / filename
        plt.savefig(save_path, dpi=self.STYLE_CONFIG['dpi'], bbox_inches='tight', facecolor=fig.get_facecolor())
        
        print(f"Weekly reading histogram saved to {save_path}")
        return save_path

    def create_reading_streaks(self, year=None, format_filter='kindle', output_dir='output', figsize=None):
        """Create a visualization showing reading streaks (consecutive days with reading activity).
        If year is specified, only shows data for that year."""
        if self.reading_data is None or self.reading_data.empty:
            print("Cannot generate plot: Reading data is empty.")
            return
        
        sns.set_theme(**self.THEME_CONFIG)
        Path(output_dir).mkdir(exist_ok=True)
        
        if figsize is None:
            figsize = self.STYLE_CONFIG['figure_size']
        
        # Prepare data
        df = self.reading_data.copy()

        # Filter by format if specified
        if format_filter:
            df = df[df['format'] == format_filter]
            if df.empty:
                print(f"No {format_filter} reading data found.")
                return
        
        # Filter by year if specified
        if year is not None:
            df = df[df['start_datetime'].dt.year == year].copy()
            if df.empty:
                print(f"No reading data found for year {year}.")
                return
            output_subdir = f'{output_dir}/{year}'
            filename = f'reading_streaks_distribution_{year}.png'
        else:
            output_subdir = f'{output_dir}/overall'
            filename = 'reading_streaks_distribution.png'

        Path(output_subdir).mkdir(parents=True, exist_ok=True)
        
        # Get dates with at least 10 minutes of reading
        daily_reading = df.groupby(df['start_datetime'].dt.date)['duration'].sum()
        min_reading_minutes = 10
        reading_dates = daily_reading[daily_reading >= min_reading_minutes * 60].index
        reading_dates = pd.Series(reading_dates).sort_values().reset_index(drop=True)
        
        # Calculate streaks
        streaks = []
        current_streak = 1
        
        for i in range(1, len(reading_dates)):
            if (reading_dates.iloc[i] - reading_dates.iloc[i-1]).days == 1:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 1
        
        # Don't forget the last streak
        if current_streak > 0:
            streaks.append(current_streak)
        
        if not streaks:
            print("No reading streaks found.")
            return
        
        # Create histogram of streak lengths
        fig, ax = plt.subplots(figsize=figsize)
        
        bins = np.arange(0.5, max(streaks) + 1.5, 1)  # Center bins on integers
        n, bins, patches = ax.hist(streaks, bins=bins, 
                                 color=self.PLOTS_COLORS[1],
                                 alpha=self.STYLE_CONFIG['bar_alpha'],
                                 edgecolor=self.STYLE_CONFIG['bar_edge_color'],
                                 linewidth=self.STYLE_CONFIG['bar_edge_width'])
        
        title = 'Reading Streaks Distribution'
        if year is not None:
            title += f' ({year})'
        
        ax.set_title(
            title,
            fontsize=self.STYLE_CONFIG['title_fontsize'],
            fontweight=self.STYLE_CONFIG['title_fontweight'],
            pad=self.STYLE_CONFIG['title_pad']
        )
        ax.set_xlabel(
            'Streak Length (consecutive days)',
            fontsize=self.STYLE_CONFIG['xlabel_fontsize'],
            labelpad=self.STYLE_CONFIG['xlabel_pad']
        )
        ax.set_ylabel(
            'Number of Streaks',
            fontsize=self.STYLE_CONFIG['ylabel_fontsize'],
            labelpad=self.STYLE_CONFIG['ylabel_pad']
        )
        
        # Set tick parameters
        ax.tick_params(axis='both', which='major', labelsize=self.STYLE_CONFIG['tick_labelsize'])
        ax.set_xticks(range(1, max(streaks) + 1))  # Show all streak values
        
        # Grid styling
        ax.grid(axis='y', alpha=1.0)
        ax.grid(axis='x', alpha=0)
        ax.set_axisbelow(True)
        
        # Add statistics text
        max_streak = max(streaks)
        avg_streak = np.mean(streaks)
        total_streaks = len(streaks)
        
        stats_text = f'Longest: {max_streak} days\nAverage: {avg_streak:.1f} days\nTotal streaks: {total_streaks}'
        ax.text(0.98, 0.98, stats_text,
                transform=ax.transAxes,
                ha='right', va='top',
                fontsize=self.STYLE_CONFIG['legend_fontsize'],
                color=self.STYLE_CONFIG['total_text_color'],
                bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
        
        plt.tight_layout()
        
        save_path = Path(output_subdir) / filename
        plt.savefig(save_path, dpi=self.STYLE_CONFIG['dpi'], bbox_inches='tight', facecolor=fig.get_facecolor())
        
        print(f"Reading streaks visualization saved to {save_path}")
        return save_path
    
    def create_yearly_reading_calendar(self, year, format_filter='kindle', output_dir='output', figsize=None):
        """Create a minimal calendar visualization showing daily reading activity for a specific year.
        Shows grey dots arranged in a 3x4 month grid with Monday as the first day of each week.
        Dots are colored based on daily reading time."""
        if self.reading_data is None or self.reading_data.empty:
            print("Cannot generate plot: Reading data is empty.")
            return
        
        if year is None:
            print("Year must be specified for calendar visualization.")
            return
        
        sns.set_theme(**self.THEME_CONFIG)
        Path(output_dir).mkdir(exist_ok=True)
        
        if figsize is None:
            figsize = (14, 12)  # Adjusted for calendar layout
        
        # Prepare data
        df = self.reading_data.copy()

        # Filter by format if specified
        if format_filter:
            df = df[df['format'] == format_filter]
            if df.empty:
                print(f"No {format_filter} reading data found.")
                return
        
        # Filter by year
        df = df[df['start_datetime'].dt.year == year].copy()
        if df.empty:
            print(f"No reading data found for year {year}.")
            return
        
        output_subdir = f'{output_dir}/{year}'
        filename = f'reading_calendar_{year}.png'
        Path(output_subdir).mkdir(parents=True, exist_ok=True)
        
        # Calculate daily reading time in minutes
        df['date'] = df['start_datetime'].dt.date
        daily_reading = df.groupby('date')['duration'].sum().reset_index()
        daily_reading['duration_minutes'] = daily_reading['duration'] / 60
        
        # Create complete year date range
        start_date = pd.Timestamp(f'{year}-01-01')
        end_date = pd.Timestamp(f'{year}-12-31')
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create complete dataset with reading data
        complete_data = pd.DataFrame({'date': all_dates.date})
        complete_data = complete_data.merge(daily_reading, on='date', how='left').fillna(0)
        complete_data['datetime'] = pd.to_datetime(complete_data['date'])
        complete_data['month'] = complete_data['datetime'].dt.month
        complete_data['day_of_week'] = complete_data['datetime'].dt.dayofweek  # Monday = 0
        complete_data['day'] = complete_data['datetime'].dt.day
        
        # Create the plot with 3x4 subplot grid
        fig, axes = plt.subplots(3, 4, figsize=figsize)
        axes = axes.flatten()
        
        # Color mapping for reading time
        max_reading = complete_data['duration_minutes'].max()
        if max_reading == 0:
            max_reading = 1  # Avoid division by zero
        
        # Custom colormap - grey to bright color
        colors = ['#444444', '#666666', '#ffd166', '#ff9f43', '#ee5a6f']
        reading_cmap = LinearSegmentedColormap.from_list('reading', colors, N=100)
        
        for month_idx in range(12):
            ax = axes[month_idx]
            month = month_idx + 1
            month_data = complete_data[complete_data['month'] == month].copy()
            
            # Clear the subplot
            ax.clear()
            ax.set_xlim(-0.5, 6.5)
            ax.set_ylim(-0.5, 5.5)
            ax.set_aspect('equal')
            
            # Calculate week positions for each date
            for _, row in month_data.iterrows():
                date_obj = row['datetime']
                day_of_week = row['day_of_week']
                
                # Calculate which week of the month this date falls in
                first_day_of_month = date_obj.replace(day=1)
                first_day_weekday = first_day_of_month.dayofweek
                
                # Calculate week number (0-based)
                days_from_first = (date_obj - first_day_of_month).days
                week_num = (days_from_first + first_day_weekday) // 7
                
                # Position in grid (x=day_of_week, y=week_num flipped)
                x = day_of_week
                y = 5 - week_num  # Flip y to have first week at top
                
                # Color based on reading time
                if row['duration_minutes'] == 0:
                    color = '#333333'  # Dark grey for no reading
                    alpha = 0.3
                else:
                    normalized_time = row['duration_minutes'] / max_reading
                    color = reading_cmap(normalized_time)
                    alpha = 0.8
                
                # Draw dot
                ax.scatter(x, y, s=120, c=[color], alpha=alpha, 
                        edgecolor='white', linewidth=0.5)
            
            # Remove ticks and labels
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        
        # Main title with year on second line - centered on main plot area
        fig.suptitle(f'Daily Reading Activity Calendar\n{year}',
                    fontsize=self.STYLE_CONFIG['title_fontsize'],
                    fontweight=self.STYLE_CONFIG['title_fontweight'],
                    y=0.95, x=0.5)
        
        # Create colorbar axes
        cbar_ax = fig.add_axes([0.95, 0.2, 0.02, 0.6])
        norm = Normalize(vmin=0, vmax=max_reading)
        cbar = ColorbarBase(cbar_ax, cmap=reading_cmap, norm=norm)
        cbar.set_label('Reading Time (minutes)', fontsize=self.STYLE_CONFIG['ylabel_fontsize'],
                    labelpad=15)
        cbar.ax.tick_params(labelsize=self.STYLE_CONFIG['tick_labelsize'])
        
        plt.subplots_adjust(top=0.88, right=0.92, hspace=0.3, wspace=0.2)
        
        save_path = Path(output_subdir) / filename
        plt.savefig(save_path, dpi=self.STYLE_CONFIG['dpi'], bbox_inches='tight', 
                    facecolor=fig.get_facecolor())
        
        print(f"Reading calendar saved to {save_path}")
        return save_path
    
    def create_yearly_streak_calendar(self, year, format_filter='kindle', output_dir='output', figsize=None, min_reading_minutes=10):
        """Create a minimal calendar visualization showing reading streaks for a specific year.
        Shows grey dots arranged in a 3x4 month grid with Monday as the first day of each week.
        Dots that are part of a streak are colored based on the streak length."""
        if self.reading_data is None or self.reading_data.empty:
            print("Cannot generate plot: Reading data is empty.")
            return
        
        if year is None:
            print("Year must be specified for calendar visualization.")
            return
        
        sns.set_theme(**self.THEME_CONFIG)
        Path(output_dir).mkdir(exist_ok=True)
        
        if figsize is None:
            figsize = (16, 12)  # Increased width from 14 to 16 for more room
        
        # Prepare data
        df = self.reading_data.copy()

        # Filter by format if specified
        if format_filter:
            df = df[df['format'] == format_filter]
            if df.empty:
                print(f"No {format_filter} reading data found.")
                return
        
        # Filter by year
        df = df[df['start_datetime'].dt.year == year].copy()
        if df.empty:
            print(f"No reading data found for year {year}.")
            return
        
        output_subdir = f'{output_dir}/{year}'
        filename = f'reading_streaks_calendar_{year}.png'
        Path(output_subdir).mkdir(parents=True, exist_ok=True)
        
        # Calculate daily reading time
        df['date'] = df['start_datetime'].dt.date
        daily_reading = df.groupby('date')['duration'].sum()
        
        # Identify dates with sufficient reading (part of potential streaks)
        reading_dates = set(daily_reading[daily_reading >= min_reading_minutes * 60].index)
        
        # Create complete year date range
        start_date = pd.Timestamp(f'{year}-01-01')
        end_date = pd.Timestamp(f'{year}-12-31')
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Identify streaks and assign each date to its streak length
        date_to_streak_length = {}
        if reading_dates:
            sorted_dates = sorted(reading_dates)
            current_streak = [sorted_dates[0]]
            all_streaks = []
            
            for i in range(1, len(sorted_dates)):
                if (sorted_dates[i] - sorted_dates[i-1]).days == 1:
                    current_streak.append(sorted_dates[i])
                else:
                    # Save current streak if it has more than 1 day
                    if len(current_streak) > 1:
                        all_streaks.append(current_streak.copy())
                    current_streak = [sorted_dates[i]]
            
            # Don't forget the last streak
            if len(current_streak) > 1:
                all_streaks.append(current_streak.copy())
            
            # Assign streak length to each date in a streak
            for streak in all_streaks:
                streak_length = len(streak)
                for date in streak:
                    date_to_streak_length[date] = streak_length
        
        # Create complete dataset
        complete_data = pd.DataFrame({'date': all_dates.date})
        complete_data['datetime'] = pd.to_datetime(complete_data['date'])
        complete_data['month'] = complete_data['datetime'].dt.month
        complete_data['day_of_week'] = complete_data['datetime'].dt.dayofweek  # Monday = 0
        complete_data['day'] = complete_data['datetime'].dt.day
        complete_data['streak_length'] = complete_data['date'].map(date_to_streak_length).fillna(0).astype(int)
        complete_data['has_reading'] = complete_data['date'].isin(reading_dates)
        
        # Create the plot with 3x4 subplot grid
        fig, axes = plt.subplots(3, 4, figsize=figsize)
        axes = axes.flatten()
        
        # Create color map for streak lengths
        if date_to_streak_length:
            max_streak_length = max(date_to_streak_length.values())
            min_streak_length = 2  # Minimum streak is 2 days
            
            # Create color gradient from light to intense colors based on streak length
            streak_colors = ['#ffd166', '#ff9f43', '#ee5a6f', '#ff6b9d', '#c44569']
            streak_cmap = LinearSegmentedColormap.from_list('streak', streak_colors, N=max(max_streak_length - 1, 5))
            
            def get_streak_color(length):
                if length == 0:
                    return None
                # Normalize to 0-1 range
                normalized = (length - min_streak_length) / max(max_streak_length - min_streak_length, 1)
                return streak_cmap(normalized)
        else:
            max_streak_length = 0
            get_streak_color = lambda x: None
        
        for month_idx in range(12):
            ax = axes[month_idx]
            month = month_idx + 1
            month_data = complete_data[complete_data['month'] == month].copy()
            
            # Clear the subplot
            ax.clear()
            ax.set_xlim(-0.5, 6.5)
            ax.set_ylim(-0.5, 5.5)
            ax.set_aspect('equal')
            
            # Calculate week positions for each date
            for _, row in month_data.iterrows():
                date_obj = row['datetime']
                day_of_week = row['day_of_week']
                
                # Calculate which week of the month this date falls in
                first_day_of_month = date_obj.replace(day=1)
                first_day_weekday = first_day_of_month.dayofweek
                
                # Calculate week number (0-based)
                days_from_first = (date_obj - first_day_of_month).days
                week_num = (days_from_first + first_day_weekday) // 7
                
                # Position in grid (x=day_of_week, y=week_num flipped)
                x = day_of_week
                y = 5 - week_num  # Flip y to have first week at top
                
                # Color based on streak length
                if row['streak_length'] > 0:
                    color = get_streak_color(row['streak_length'])
                    alpha = 0.9
                    edge_width = 1.0
                elif row['has_reading']:
                    color = '#666666'  # Medium grey for reading days not in streaks
                    alpha = 0.5
                    edge_width = 0.5
                else:
                    color = '#333333'  # Dark grey for no reading
                    alpha = 0.3
                    edge_width = 0.5
                
                # Draw dot
                ax.scatter(x, y, s=120, c=[color], alpha=alpha, 
                        edgecolor='white', linewidth=edge_width)
            
            # Remove ticks and labels
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        
        # Main title with year on second line - centered on main plot area
        fig.suptitle(f'Reading Streak Calendar\n{year}',
                    fontsize=self.STYLE_CONFIG['title_fontsize'],
                    fontweight=self.STYLE_CONFIG['title_fontweight'],
                    y=0.95, x=0.5)
        
        # Add colorbar for streak lengths if there are any streaks
        if date_to_streak_length:
            from matplotlib.colorbar import ColorbarBase
            from matplotlib.colors import Normalize
            
            # Create colorbar axes - moved further right to avoid overlap
            cbar_ax = fig.add_axes([0.88, 0.3, 0.02, 0.4])
            norm = Normalize(vmin=2, vmax=max_streak_length)
            cbar = ColorbarBase(cbar_ax, cmap=streak_cmap, norm=norm)
            cbar.set_label('Streak Length (days)', fontsize=self.STYLE_CONFIG['ylabel_fontsize'],
                        labelpad=15)
            cbar.ax.tick_params(labelsize=self.STYLE_CONFIG['tick_labelsize'])
            
            # Set integer ticks
            import numpy as np
            if max_streak_length <= 10:
                ticks = list(range(2, max_streak_length + 1))
            else:
                # For longer streaks, show fewer ticks
                ticks = np.linspace(2, max_streak_length, min(6, max_streak_length - 1)).astype(int)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([str(t) for t in ticks])
        
        # Add legend for non-streak dots
        legend_elements = [
            plt.scatter([], [], s=120, c='#666666', alpha=0.5, 
                    edgecolor='white', linewidth=0.5, label='Reading (no streak)'),
            plt.scatter([], [], s=120, c='#333333', alpha=0.3, 
                    edgecolor='white', linewidth=0.5, label='No reading')
        ]
        
        # Create legend axes - positioned further right and made smaller
        legend_ax = fig.add_axes([0.83, 0.15, 0.15, 0.08])
        legend_ax.axis('off')
        legend_ax.legend(handles=legend_elements, loc='center left', 
                        fontsize=self.STYLE_CONFIG['legend_fontsize'] - 2,
                        frameon=False)
        
        # Adjusted subplot positioning to leave more room on the right
        plt.subplots_adjust(top=0.88, right=0.80, hspace=0.3, wspace=0.2)
        
        save_path = Path(output_subdir) / filename
        plt.savefig(save_path, dpi=self.STYLE_CONFIG['dpi'], bbox_inches='tight', 
                    facecolor=fig.get_facecolor())
        
        print(f"Streak calendar saved to {save_path}")
        return save_path

    def create_hourly_session_distribution(self, year=None, format_filter='kindle', output_dir='output', figsize=None):
        """Create a bar chart showing average distribution of reading sessions during hours of the day.
        If year is specified, only shows data for that year."""
        if self.reading_data is None or self.reading_data.empty:
            print("Cannot generate plot: Reading data is empty.")
            return
        
        sns.set_theme(**self.THEME_CONFIG)
        Path(output_dir).mkdir(exist_ok=True)
        
        if figsize is None:
            figsize = self.STYLE_CONFIG['figure_size']
        
        # Prepare data
        df = self.reading_data.copy()

        # Filter by format if specified
        if format_filter:
            df = df[df['format'] == format_filter]
            if df.empty:
                print(f"No {format_filter} reading data found.")
                return
        
        # Filter by year if specified
        if year is not None:
            df = df[df['start_datetime'].dt.year == year].copy()
            if df.empty:
                print(f"No reading data found for year {year}.")
                return
            output_subdir = f'{output_dir}/{year}'
            filename = f'session_distribution_hourly_{year}.png'
        else:
            output_subdir = f'{output_dir}/overall'
            filename = 'session_distribution_hourly.png'

        Path(output_subdir).mkdir(parents=True, exist_ok=True)

        # Calculate average reading time by 30-minute slots
        df['hour_slot'] = df['start_datetime'].dt.hour + (df['start_datetime'].dt.minute // 30) * 0.5
        hourly_reading = df.groupby('hour_slot')['duration'].sum().reset_index()
        hourly_reading['duration_minutes'] = hourly_reading['duration'] / 60

        # Calculate total days in the dataset period
        if year is not None:
            total_days = 366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365
        else:
            # For overall data, calculate actual date range
            min_date = df['start_datetime'].dt.date.min()
            max_date = df['start_datetime'].dt.date.max()
            total_days = (max_date - min_date).days + 1

        # Calculate average duration per day for each slot
        hourly_reading['avg_duration_minutes'] = hourly_reading['duration_minutes'] / total_days

        # Ensure all 30-minute slots 0-23.5 are represented
        all_slots = pd.DataFrame({'hour_slot': np.arange(0, 24, 0.5)})
        hourly_sessions = all_slots.merge(hourly_reading, on='hour_slot', how='left').fillna(0)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Smooth the line using spline interpolation
        x_smooth = np.linspace(hourly_sessions['hour_slot'].min(), hourly_sessions['hour_slot'].max(), 300)
        spl = make_interp_spline(hourly_sessions['hour_slot'], hourly_sessions['avg_duration_minutes'], k=3)
        y_smooth = spl(x_smooth)
        
        # Prevent line from going below 0
        y_smooth = np.maximum(y_smooth, 0)
        
        ax.plot(x_smooth, y_smooth,
            color=self.PLOTS_COLORS[2],
            linewidth=3,
            alpha=0.9)
        
        # Fill area under the curve
        ax.fill_between(x_smooth, y_smooth,
            alpha=0.3,
            color=self.PLOTS_COLORS[2])
        
        title = 'Reading Sessions Distribution by Hour of Day'
        if year is not None:
            title += f' ({year})'
        
        ax.set_title(
            title,
            fontsize=self.STYLE_CONFIG['title_fontsize'],
            fontweight=self.STYLE_CONFIG['title_fontweight'],
            pad=self.STYLE_CONFIG['title_pad']
        )
        ax.set_xlabel(
            'Hour of Day',
            fontsize=self.STYLE_CONFIG['ylabel_fontsize'],
            labelpad=self.STYLE_CONFIG['ylabel_pad']
        )
        ax.set_ylabel(
            'Average Reading Time (minutes)',
            fontsize=self.STYLE_CONFIG['ylabel_fontsize'],
            labelpad=self.STYLE_CONFIG['ylabel_pad']
        )
        
        # Set tick parameters
        ax.tick_params(axis='both', which='major', labelsize=self.STYLE_CONFIG['tick_labelsize'])
        ax.set_xticks(range(0, 24, 2))  # Show every 2 hours

        ax.set_xlim(0, hourly_sessions['hour_slot'].max())
        
        # Grid styling
        ax.grid(axis='y', alpha=1.0)
        ax.grid(axis='x', alpha=0)
        
        # Add vertical dashed lines every 2 hours
        for hour in range(0, 24, 2):
            ax.axvline(x=hour, color='#444444', linestyle='--', alpha=0.7)
        
        ax.set_axisbelow(True)
        
        # Set y-axis minimum to 0
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        save_path = Path(output_subdir) / filename
        plt.savefig(save_path, dpi=self.STYLE_CONFIG['dpi'], bbox_inches='tight', facecolor=fig.get_facecolor())
        
        print(f"Hourly session distribution saved to {save_path}")
        return save_path
    
    def plot_average_daily_monthly(self, year=None, format_filter='kindle', output_dir='output', figsize=None):
        """Create a combined plot with two subplots:
        1. Daily session distribution (average reading time by day of week)
        2. Monthly reading histogram (total reading time by month)
        If year is specified, only shows data for that year."""
        if self.reading_data is None or self.reading_data.empty:
            print("Cannot generate plot: Reading data is empty.")
            return
        
        sns.set_theme(**self.THEME_CONFIG)
        Path(output_dir).mkdir(exist_ok=True)
        
        if figsize is None:
            figsize = (18, 12)  # Taller for two subplots
        
        # Prepare data
        df = self.reading_data.copy()

        # Filter by format if specified
        if format_filter:
            df = df[df['format'] == format_filter]
            if df.empty:
                print(f"No {format_filter} reading data found.")
                return
        
        # Filter by year if specified
        if year is not None:
            df = df[df['start_datetime'].dt.year == year].copy()
            if df.empty:
                print(f"No reading data found for year {year}.")
                return
            output_subdir = f'{output_dir}/{year}'
            filename = f'reading_time_daily_monthly_{year}.png'
        else:
            output_subdir = f'{output_dir}/overall'
            filename = 'reading_time_daily_monthly.png'

        Path(output_subdir).mkdir(parents=True, exist_ok=True)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # ============ SUBPLOT 1: Daily Session Distribution ============
        # Calculate average reading time by day of week
        df['day_of_week'] = df['start_datetime'].dt.day_name()
        daily_reading = df.groupby('day_of_week')['duration'].sum().reset_index()
        daily_reading['duration_minutes'] = daily_reading['duration'] / 60
        
        # Count unique dates for each day to calculate average
        daily_dates = df.groupby('day_of_week')['start_datetime'].apply(lambda x: x.dt.date.nunique()).reset_index()
        daily_dates.columns = ['day_of_week', 'unique_dates']
        
        # Merge and calculate average
        daily_sessions = daily_reading.merge(daily_dates, on='day_of_week')
        daily_sessions['avg_duration_minutes'] = daily_sessions['duration_minutes'] / daily_sessions['unique_dates']
        
        # Ensure proper day order
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_sessions['day_of_week'] = pd.Categorical(daily_sessions['day_of_week'], categories=day_order, ordered=True)
        daily_sessions = daily_sessions.sort_values('day_of_week').reset_index(drop=True)
        
        ax1.bar(daily_sessions['day_of_week'], daily_sessions['avg_duration_minutes'],
            width=self.STYLE_CONFIG['bar_width'],
            color=self.PLOTS_COLORS[3],
            alpha=self.STYLE_CONFIG['bar_alpha'],
            edgecolor=self.STYLE_CONFIG['bar_edge_color'],
            linewidth=self.STYLE_CONFIG['bar_edge_width'])
        
        title1 = 'Daily Reading Time'
        if year is not None:
            title1 += f' ({year})'
        
        ax1.set_title(
            title1,
            fontsize=self.STYLE_CONFIG['title_fontsize'],
            fontweight=self.STYLE_CONFIG['title_fontweight'],
            pad=self.STYLE_CONFIG['title_pad']
        )
        ax1.set_ylabel(
            'Average Reading Time (minutes)',
            fontsize=self.STYLE_CONFIG['ylabel_fontsize'],
            labelpad=self.STYLE_CONFIG['ylabel_pad']
        )
        
        # Style first subplot
        ax1.tick_params(axis='both', which='major', labelsize=self.STYLE_CONFIG['tick_labelsize'])
        ax1.tick_params(axis='x', rotation=0)
        ax1.grid(axis='y', alpha=1.0)
        ax1.grid(axis='x', alpha=0)
        ax1.set_axisbelow(True)
        
        # Add total text on top of bars
        totals1 = daily_sessions['avg_duration_minutes']
        y_offset1 = totals1.max() * self.STYLE_CONFIG['total_text_offset_ratio']
        for i, total in enumerate(totals1):
            if total > 0:
                ax1.text(
                    i,
                    total + y_offset1,
                    f'{total:.0f}m',
                    ha='center',
                    va='bottom',
                    fontsize=self.STYLE_CONFIG['total_text_fontsize'],
                    fontweight=self.STYLE_CONFIG['total_text_fontweight'],
                    color=self.STYLE_CONFIG['total_text_color']
                )
        
        # ============ SUBPLOT 2: Monthly Reading Histogram ============
        if year is not None:
            # For specific year: show monthly totals
            df['month'] = df['start_datetime'].dt.to_period('M')
            monthly_reading = df.groupby('month')['duration'].sum().reset_index()
            monthly_reading['duration_minutes'] = monthly_reading['duration'] / 60
            monthly_reading['month_label'] = df.groupby('month')['start_datetime'].first().dt.strftime('%B').values
            monthly_reading['month'] = monthly_reading['month'].dt.to_timestamp()
        else:
            # For all years: show average by month across years
            df['month_name'] = df['start_datetime'].dt.month_name()
            df['year'] = df['start_datetime'].dt.year
            
            # Calculate total reading time per month per year
            monthly_yearly = df.groupby(['year', 'month_name'])['duration'].sum().reset_index()
            monthly_yearly['duration_minutes'] = monthly_yearly['duration'] / 60
            
            # Calculate average across years for each month
            monthly_reading = monthly_yearly.groupby('month_name')['duration_minutes'].mean().reset_index()
            monthly_reading.columns = ['month_label', 'duration_minutes']
            
            # Ensure proper month order
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                        'July', 'August', 'September', 'October', 'November', 'December']
            monthly_reading['month_label'] = pd.Categorical(monthly_reading['month_label'], 
                                                        categories=month_order, ordered=True)
            monthly_reading = monthly_reading.sort_values('month_label').reset_index(drop=True)
        
        ax2.bar(range(len(monthly_reading)), monthly_reading['duration_minutes'],
            width=0.8,
            color=self.PLOTS_COLORS[4],
            alpha=self.STYLE_CONFIG['bar_alpha'],
            edgecolor=self.STYLE_CONFIG['bar_edge_color'],
            linewidth=self.STYLE_CONFIG['bar_edge_width'])
        
        title2 = 'Monthly Reading Time'
        if year is not None:
            title2 += f' ({year})'
        
        # Determine y-axis label for monthly subplot
        if year is not None:
            monthly_ylabel = 'Reading Time (minutes)'
        else:
            monthly_ylabel = 'Average Reading Time (minutes)'
        
        ax2.set_title(
            title2,
            fontsize=self.STYLE_CONFIG['title_fontsize'],
            fontweight=self.STYLE_CONFIG['title_fontweight'],
            pad=self.STYLE_CONFIG['title_pad']
        )
        ax2.set_ylabel(
            monthly_ylabel,
            fontsize=self.STYLE_CONFIG['ylabel_fontsize'],
            labelpad=self.STYLE_CONFIG['ylabel_pad']
        )
        
        # Style second subplot
        ax2.tick_params(axis='both', which='major', labelsize=self.STYLE_CONFIG['tick_labelsize'])
        ax2.tick_params(axis='x', rotation=0)
        ax2.set_xticks(range(len(monthly_reading)))
        ax2.set_xticklabels(monthly_reading['month_label'])
        ax2.grid(axis='y', alpha=1.0)
        ax2.grid(axis='x', alpha=0)
        ax2.set_axisbelow(True)
        
        # Add total text on top of bars
        totals2 = monthly_reading['duration_minutes']
        y_offset2 = totals2.max() * self.STYLE_CONFIG['total_text_offset_ratio']
        for i, total in enumerate(totals2):
            if total > 0:
                ax2.text(
                    i,
                    total + y_offset2,
                    f'{total:.0f}m',
                    ha='center',
                    va='bottom',
                    fontsize=self.STYLE_CONFIG['total_text_fontsize'],
                    fontweight=self.STYLE_CONFIG['total_text_fontweight'],
                    color=self.STYLE_CONFIG['total_text_color']
                )
        
        plt.tight_layout()
        
        save_path = Path(output_subdir) / filename
        plt.savefig(save_path, dpi=self.STYLE_CONFIG['dpi'], bbox_inches='tight', facecolor=fig.get_facecolor())
        
        print(f"Combined daily and monthly reading distribution saved to {save_path}")
        return save_path
    
    def create_session_length_distribution(self, year=None, format_filter='kindle', output_dir='output', figsize=None, session_gap_minutes=10, min_session_length_minutes=5):
        """Create session length distribution with two subplots:
        1. Average session length by day of week
        2. Average session length over time
        Sessions are defined as consecutive page readings with gaps less than session_gap_minutes.
        Only sessions lasting at least min_session_length_minutes are included in statistics.
        If year is specified, only shows data for that year."""
        if self.reading_data is None or self.reading_data.empty:
            print("Cannot generate plot: Session length distribution requires reading data.")
            return
        
        sns.set_theme(**self.THEME_CONFIG)
        Path(output_dir).mkdir(exist_ok=True)
        
        if figsize is None:
            figsize = (18, 12)  # Taller for two subplots
        
        # Prepare data
        df = self.reading_data.copy()

        # Filter by format if specified
        if format_filter:
            df = df[df['format'] == format_filter]
            if df.empty:
                print(f"No {format_filter} reading data found.")
                return
        
        # Filter by year if specified
        if year is not None:
            df = df[df['start_datetime'].dt.year == year].copy()
            if df.empty:
                print(f"No reading data found for year {year}.")
                return
            output_subdir = f'{output_dir}/{year}'
            filename = f'session_length_distribution_{year}.png'
        else:
            output_subdir = f'{output_dir}/overall'
            filename = 'session_length_distribution.png'

        Path(output_subdir).mkdir(parents=True, exist_ok=True)
        
        # ============ GROUP PAGES INTO READING SESSIONS ============
        # Sort by datetime only (not by book, so we capture cross-book sessions)
        df = df.sort_values(['start_datetime']).reset_index(drop=True)
        
        sessions = []
        current_session = None
        
        for idx, row in df.iterrows():
            if current_session is None:
                # Start new session
                current_session = {
                    'books': {row['id_book']: row['title']},  # Track multiple books
                    'start_time': row['start_datetime'],
                    'end_time': row['start_datetime'],
                    'total_duration': row['duration'],
                    'pages_read': 1,
                    'day_of_week': row['start_datetime'].day_name()
                }
            else:
                # Check if this page belongs to current session (only time gap matters)
                time_gap = (row['start_datetime'] - current_session['end_time']).total_seconds() / 60
                
                if time_gap <= session_gap_minutes:
                    # Continue current session (regardless of book)
                    current_session['end_time'] = row['start_datetime']
                    current_session['total_duration'] += row['duration']
                    current_session['pages_read'] += 1
                    current_session['books'][row['id_book']] = row['title']  # Add book to session
                else:
                    # End current session and start new one
                    sessions.append(current_session)
                    current_session = {
                        'books': {row['id_book']: row['title']},
                        'start_time': row['start_datetime'],
                        'end_time': row['start_datetime'],
                        'total_duration': row['duration'],
                        'pages_read': 1,
                        'day_of_week': row['start_datetime'].day_name()
                    }
        
        # Don't forget the last session
        if current_session is not None:
            sessions.append(current_session)
        
        # Convert to DataFrame
        sessions_df = pd.DataFrame(sessions)
        sessions_df['duration_minutes'] = sessions_df['total_duration'] / 60

        # Filter out sessions below minimum length
        sessions_df = sessions_df[sessions_df['duration_minutes'] >= min_session_length_minutes].reset_index(drop=True)
        
        if sessions_df.empty:
            print(f"No sessions found meeting minimum length requirement of {min_session_length_minutes} minutes.")
            return
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # ============ SUBPLOT 1: Session length by day of week ============
        daily_sessions = sessions_df.groupby('day_of_week')['duration_minutes'].mean().reset_index()
        
        # Ensure proper day order
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_sessions['day_of_week'] = pd.Categorical(daily_sessions['day_of_week'], categories=day_order, ordered=True)
        daily_sessions = daily_sessions.sort_values('day_of_week').reset_index(drop=True)
        
        ax1.bar(daily_sessions['day_of_week'], daily_sessions['duration_minutes'],
               width=self.STYLE_CONFIG['bar_width'],
               color=self.PLOTS_COLORS[5],
               alpha=self.STYLE_CONFIG['bar_alpha'],
               edgecolor=self.STYLE_CONFIG['bar_edge_color'],
               linewidth=self.STYLE_CONFIG['bar_edge_width'])
        
        ax1.set_title(
            'Average Session Length by Day of Week',
            fontsize=self.STYLE_CONFIG['title_fontsize'],
            fontweight=self.STYLE_CONFIG['title_fontweight'],
            pad=self.STYLE_CONFIG['title_pad']
        )
        ax1.set_xlabel(
            'Day of Week',
            fontsize=self.STYLE_CONFIG['ylabel_fontsize'],
            labelpad=self.STYLE_CONFIG['ylabel_pad']
        )
        ax1.set_ylabel(
            'Average Session Length (minutes)',
            fontsize=self.STYLE_CONFIG['ylabel_fontsize'],
            labelpad=self.STYLE_CONFIG['ylabel_pad']
        )
        
        # Style first subplot
        ax1.tick_params(axis='both', which='major', labelsize=self.STYLE_CONFIG['tick_labelsize'])
        ax1.tick_params(axis='x', rotation=0)
        ax1.grid(axis='y', alpha=1.0)
        ax1.grid(axis='x', alpha=0)
        ax1.set_axisbelow(True)
        
        # Add value labels on bars
        totals1 = daily_sessions['duration_minutes']
        y_offset1 = totals1.max() * self.STYLE_CONFIG['total_text_offset_ratio']
        for i, total in enumerate(totals1):
            if total > 0:
                ax1.text(
                    i,
                    total + y_offset1,
                    f'{total:.1f}m',
                    ha='center',
                    va='bottom',
                    fontsize=self.STYLE_CONFIG['total_text_fontsize'],
                    fontweight=self.STYLE_CONFIG['total_text_fontweight'],
                    color=self.STYLE_CONFIG['total_text_color']
                )
        
        # ============ SUBPLOT 2: Session length over time ============
        if year is not None:
            # Monthly aggregation for specific year
            sessions_df['month'] = sessions_df['start_time'].dt.to_period('M')
            time_sessions = sessions_df.groupby('month')['duration_minutes'].mean().reset_index()
            time_sessions['time_label'] = sessions_df.groupby('month')['start_time'].first().dt.strftime('%B').values
            time_sessions['month'] = time_sessions['month'].dt.to_timestamp()
            
            x_label = 'Month'
            title_suffix = f' ({year})'
        else:
            # Monthly aggregation for entire dataset
            sessions_df['year_month'] = sessions_df['start_time'].dt.to_period('M')
            time_sessions = sessions_df.groupby('year_month')['duration_minutes'].mean().reset_index()
            time_sessions['time_label'] = time_sessions['year_month'].dt.strftime('%Y-%m')
            time_sessions['year_month'] = time_sessions['year_month'].dt.to_timestamp()
            
            x_label = 'Month'
            title_suffix = ''
        
        ax2.bar(range(len(time_sessions)), time_sessions['duration_minutes'],
               width=0.8,
               color=self.PLOTS_COLORS[6],
               alpha=self.STYLE_CONFIG['bar_alpha'],
               edgecolor=self.STYLE_CONFIG['bar_edge_color'],
               linewidth=self.STYLE_CONFIG['bar_edge_width'])
        
        ax2.set_title(
            f'Average Session Length Over Time{title_suffix}',
            fontsize=self.STYLE_CONFIG['title_fontsize'],
            fontweight=self.STYLE_CONFIG['title_fontweight'],
            pad=self.STYLE_CONFIG['title_pad']
        )
        ax2.set_xlabel(
            x_label,
            fontsize=self.STYLE_CONFIG['ylabel_fontsize'],
            labelpad=self.STYLE_CONFIG['ylabel_pad']
        )
        ax2.set_ylabel(
            'Average Session Length (minutes)',
            fontsize=self.STYLE_CONFIG['ylabel_fontsize'],
            labelpad=self.STYLE_CONFIG['ylabel_pad']
        )
        
        # Style second subplot
        ax2.tick_params(axis='both', which='major', labelsize=self.STYLE_CONFIG['tick_labelsize'])
        ax2.tick_params(axis='x', rotation=45 if not year else 0)
        ax2.set_xticks(range(len(time_sessions)))
        ax2.set_xticklabels(time_sessions['time_label'])
        ax2.grid(axis='y', alpha=1.0)
        ax2.grid(axis='x', alpha=0)
        ax2.set_axisbelow(True)
        
        # Add value labels on bars
        totals2 = time_sessions['duration_minutes']
        y_offset2 = totals2.max() * self.STYLE_CONFIG['total_text_offset_ratio']
        for i, total in enumerate(totals2):
            if total > 0:
                ax2.text(
                    i,
                    total + y_offset2,
                    f'{total:.1f}m',
                    ha='center',
                    va='bottom',
                    fontsize=self.STYLE_CONFIG['total_text_fontsize'],
                    fontweight=self.STYLE_CONFIG['total_text_fontweight'],
                    color=self.STYLE_CONFIG['total_text_color']
                )
        
        plt.tight_layout()
        
        save_path = Path(output_subdir) / filename
        plt.savefig(save_path, dpi=self.STYLE_CONFIG['dpi'], bbox_inches='tight', facecolor=fig.get_facecolor())
        
        print(f"Session length distribution saved to {save_path}")
        return save_path
    
    def create_book_completion_timeline(self, year=None, output_dir='output', figsize=None):
        """Create a timeline showing when each book was started and finished.
        If year is specified, only shows books read during that year."""
        if self.reading_data is None or self.reading_data.empty:
            print("Cannot generate plot: Book completion timeline requires reading data.")
            return
        
        sns.set_theme(**self.THEME_CONFIG)
        Path(output_dir).mkdir(exist_ok=True)
        
        if figsize is None:
            figsize = (24, 12)
        
        # Prepare data
        df = self.reading_data.copy()
        
        # Filter by year if specified
        if year is not None:
            df = df[df['start_datetime'].dt.year == year].copy()
            if df.empty:
                print(f"No reading data found for year {year}.")
                return
            output_subdir = f'{output_dir}/{year}'
            filename = f'books_completion_timeline_{year}.png'
        else:
            output_subdir = f'{output_dir}/overall'
            filename = 'books_completion_timeline.png'

        Path(output_subdir).mkdir(parents=True, exist_ok=True)
        
        # ============ CALCULATE BOOK START/END DATES ============
        book_timeline = df.groupby(['id_book', 'title', 'authors']).agg({
            'start_datetime': ['min', 'max'],
            'duration': 'sum',
            'page': ['min', 'max'],
            'pages': 'first'
        }).reset_index()
        
        # Flatten column names
        book_timeline.columns = ['id_book', 'title', 'authors', 'start_date', 'end_date', 
                                'total_duration', 'first_page', 'last_page', 'total_pages']
        
        # Calculate completion percentage
        book_timeline['pages_read'] = book_timeline['last_page'] - book_timeline['first_page'] + 1
        book_timeline['completion_pct'] = (book_timeline['pages_read'] / book_timeline['total_pages']) * 100
        book_timeline['completion_pct'] = book_timeline['completion_pct'].clip(0, 100)
        
        # Add format information for visual distinction
        book_timeline = book_timeline.merge(
            df[['id_book', 'format']].drop_duplicates(), 
            on='id_book', 
            how='left'
        )
        
        # Determine if book was completed (assuming >90% means completed)
        book_timeline['completed'] = book_timeline['completion_pct'] >= 90
        
        # Sort by start date
        book_timeline = book_timeline.sort_values('start_date').reset_index(drop=True)
        
        # Calculate reading duration in days
        book_timeline['reading_days'] = (book_timeline['end_date'] - book_timeline['start_date']).dt.days + 1
        book_timeline['total_hours'] = book_timeline['total_duration'] / 3600
        
        # ============ CREATE TIMELINE VISUALIZATION ============
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create y positions for books (reverse order so newest is on top)
        y_positions = range(len(book_timeline))
        book_timeline['y_pos'] = y_positions
        
        # Plot timeline bars with reading gaps for Kindle books
        GAP_THRESHOLD = 7  # Days of no reading to show as gap
        
        for idx, row in book_timeline.iterrows():
            start_date = row['start_date']
            end_date = row['end_date']
            y_pos = row['y_pos']
            book_format = row['format']
            book_id = row['id_book']
            
            # Use consistent color for all books
            color = self.BOOK_COLORS[idx % len(self.BOOK_COLORS)]
            
            if book_format == 'paperback':
                # Paperbacks: diagonal hatching pattern
                ax.barh(y_pos, (end_date - start_date).days + 1, 
                       left=start_date, height=0.8,
                       color=color, alpha=self.STYLE_CONFIG['bar_alpha'],
                       edgecolor=self.STYLE_CONFIG['bar_edge_color'],
                       linewidth=self.STYLE_CONFIG['bar_edge_width'],
                       hatch='///')
                
            else:
                # Kindle books: analyze reading pattern and create bars accordingly
                book_data = df[df['id_book'] == book_id].copy()
                book_data['date'] = book_data['start_datetime'].dt.date
                reading_dates = set(book_data['date'])
                
                # Check if book has any significant gaps
                has_significant_gaps = False
                current_date = start_date.date()
                end_date_only = end_date.date()
                temp_date = current_date
                
                # First pass: check for gaps >= GAP_THRESHOLD
                while temp_date <= end_date_only:
                    if temp_date not in reading_dates:
                        gap_start = temp_date
                        gap_length = 0
                        while (temp_date <= end_date_only and 
                               temp_date not in reading_dates):
                            gap_length += 1
                            temp_date += pd.Timedelta(days=1).to_pytimedelta()
                        
                        if gap_length >= GAP_THRESHOLD:
                            has_significant_gaps = True
                            break
                    else:
                        temp_date += pd.Timedelta(days=1).to_pytimedelta()
                
                if not has_significant_gaps:
                    # No significant gaps: draw one continuous solid bar
                    ax.barh(y_pos, (end_date - start_date).days + 1, 
                           left=start_date, height=0.8,
                           color=color, alpha=self.STYLE_CONFIG['bar_alpha'],
                           edgecolor=self.STYLE_CONFIG['bar_edge_color'],
                           linewidth=self.STYLE_CONFIG['bar_edge_width'])
                else:
                    # Has significant gaps: create merged segments
                    current_date = start_date.date()
                    
                    while current_date <= end_date_only:
                        if current_date in reading_dates:
                            # Start a continuous segment (reading + small gaps + reading)
                            segment_start = current_date
                            
                            # Keep extending while we have reading days OR small gaps
                            while current_date <= end_date_only:
                                if current_date in reading_dates:
                                    # Reading day - continue
                                    current_date += pd.Timedelta(days=1).to_pytimedelta()
                                else:
                                    # Gap - check if it's small or large
                                    gap_start_temp = current_date
                                    gap_length = 0
                                    temp_date = current_date
                                    
                                    while (temp_date <= end_date_only and 
                                           temp_date not in reading_dates):
                                        gap_length += 1
                                        temp_date += pd.Timedelta(days=1).to_pytimedelta()
                                    
                                    if gap_length >= GAP_THRESHOLD:
                                        # Large gap - end current segment and handle gap separately
                                        break
                                    else:
                                        # Small gap - include in current segment
                                        current_date = temp_date
                            
                            segment_end = current_date - pd.Timedelta(days=1).to_pytimedelta()
                            segment_days = (segment_end - segment_start).days + 1
                            
                            # Draw continuous solid segment
                            ax.barh(y_pos, segment_days, left=pd.Timestamp(segment_start), height=0.8,
                                   color=color, alpha=self.STYLE_CONFIG['bar_alpha'],
                                   edgecolor=self.STYLE_CONFIG['bar_edge_color'],
                                   linewidth=self.STYLE_CONFIG['bar_edge_width'])
                        
                        else:
                            # We're at a gap - check if it's significant
                            gap_start = current_date
                            gap_length = 0
                            while (current_date <= end_date_only and 
                                   current_date not in reading_dates):
                                gap_length += 1
                                current_date += pd.Timedelta(days=1).to_pytimedelta()
                            
                            # Only draw transparent bar for significant gaps
                            if gap_length >= GAP_THRESHOLD:
                                ax.barh(y_pos, gap_length, left=pd.Timestamp(gap_start), height=0.8,
                                       color=color, alpha=0.15,
                                       edgecolor=self.STYLE_CONFIG['bar_edge_color'],
                                       linewidth=self.STYLE_CONFIG['bar_edge_width'],
                                       linestyle='--')
        
        # ============ FORMATTING ============
        title = 'Book Reading Timeline'
        if year is not None:
            title += f' ({year})'
        
        ax.set_title(
            title,
            fontsize=self.STYLE_CONFIG['title_fontsize'],
            fontweight=self.STYLE_CONFIG['title_fontweight'],
            pad=self.STYLE_CONFIG['title_pad']
        )
        
        # Remove y-axis labels since we'll put titles on/near bars
        ax.set_yticks(y_positions)
        ax.set_yticklabels([])
        
        # Add book titles inside or beside bars
        for idx, row in book_timeline.iterrows():
            # Strip whitespace to prevent text alignment issues
            title_text = row['title'].strip()
            y_pos = row['y_pos']
            start_date = row['start_date']
            end_date = row['end_date']
            bar_duration_days = (end_date - start_date).days + 1
            
            # Determine if title fits inside the bar
            char_width_ratio = 0.50
            bar_usage_ratio = 0.70
            min_bar_days = 15

            title_width_estimate = len(title_text) * char_width_ratio
            usable_bar_width = bar_duration_days * bar_usage_ratio
            
            place_inside = bar_duration_days >= min_bar_days and title_width_estimate <= usable_bar_width

            if place_inside:
                text_x = start_date + pd.Timedelta(days=bar_duration_days / 2)
                ha = 'center'
                max_chars_inside = int(usable_bar_width / char_width_ratio)
                display_title = title_text if len(title_text) <= max_chars_inside else title_text[:max_chars_inside-3] + "..."
            else:
                plot_end = book_timeline['end_date'].max()
                days_to_plot_end = (plot_end - end_date).days
                
                if days_to_plot_end > 30:
                    text_x = end_date + pd.Timedelta(days=3)
                    ha = 'left'
                else:
                    text_x = start_date - pd.Timedelta(days=3)
                    ha = 'right'
                
                display_title = title_text if len(title_text) <= 60 else title_text[:57] + "..."

            # Set text color and background
            if row['format'] == 'paperback' and place_inside:
                # === START: MODIFIED CODE FOR PAPERBACK TEXT BACKGROUND ===
                # This new block manually creates a background patch that is 
                # proportional to the text length and properly centered.
                
                # 1. Estimate text width in data coordinates (days)
                estimated_text_width_days = len(display_title) * char_width_ratio
                
                # 2. Define proportional padding to make the box look balanced
                padding_ratio = 1.25  # Box will be 125% wider than the text
                box_width_days = estimated_text_width_days * (1 + padding_ratio)
                
                # 3. Calculate the box's start position to center it on text_x
                box_start_date = text_x - pd.Timedelta(days=box_width_days / 2)
                box_start_num = mdates.date2num(box_start_date)

                # 4. Define box height and vertical position
                box_height = 0.4
                box_y_pos = y_pos - (box_height / 2)

                # 5. Create and add the background patch
                text_bg_box = FancyBboxPatch(
                    (box_start_num, box_y_pos),
                    width=box_width_days,
                    height=box_height,
                    boxstyle="round,pad=0", # Padding is handled by our width calculation
                    facecolor='white',
                    alpha=0.85, # A bit more opaque for readability
                    edgecolor='none',
                    zorder=3  # Ensure it's above the bar
                )
                ax.add_patch(text_bg_box)
                
                # 6. Draw the text, centered on top of the new patch
                ax.text(text_x, y_pos, display_title,
                       ha='center', va='center',
                       fontsize=self.STYLE_CONFIG['tick_labelsize'],
                       color='black',
                       fontweight='bold',
                       zorder=4) # Ensure it's above the patch

                # === END: MODIFIED CODE ===
            else:
                # Original text rendering for Kindle books or titles outside bars
                text_color = 'white' if place_inside else self.STYLE_CONFIG['total_text_color']
                ax.text(text_x, y_pos, display_title,
                       ha=ha, va='center',
                       fontsize=self.STYLE_CONFIG['tick_labelsize'],
                       color=text_color,
                       fontweight='bold')
        
        # Format x-axis
        ax.tick_params(axis='both', which='major', labelsize=self.STYLE_CONFIG['tick_labelsize'])
        ax.tick_params(axis='x', rotation=0)
        # Format x-axis dates based on year filter
        if year is not None:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        
        # Grid styling
        ax.grid(axis='x', alpha=0.7)
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)
        
        # Add summary statistics
        completed_books = book_timeline['completed'].sum()
        total_books = len(book_timeline)
        avg_reading_days = book_timeline[book_timeline['completed']]['reading_days'].mean()
        
        kindle_books = book_timeline[book_timeline['format'] == 'kindle'].shape[0]
        paper_books = book_timeline[book_timeline['format'] == 'paperback'].shape[0]
        
        stats_text = f'Total Books: {total_books}\n'
        stats_text += f'Kindle: {kindle_books} • Paper: {paper_books}\n'
        if not pd.isna(avg_reading_days):
            stats_text += f'Avg. Reading Time: {avg_reading_days:.1f} days'
        
        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes,
                ha='left', va='top',
                fontsize=self.STYLE_CONFIG['legend_fontsize'],
                color=self.STYLE_CONFIG['total_text_color'],
                bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
        
        # Add legend for bar types
        from matplotlib.patches import Rectangle
        
        legend_elements = [
            Rectangle((0, 0), 1, 1, facecolor=self.BOOK_COLORS[0], 
                     alpha=self.STYLE_CONFIG['bar_alpha'],
                     edgecolor=self.STYLE_CONFIG['bar_edge_color'],
                     linewidth=self.STYLE_CONFIG['bar_edge_width'],
                     label='Kindle'),
            Rectangle((0, 0), 1, 1, facecolor=self.BOOK_COLORS[1], 
                     alpha=self.STYLE_CONFIG['bar_alpha'],
                     edgecolor=self.STYLE_CONFIG['bar_edge_color'],
                     linewidth=self.STYLE_CONFIG['bar_edge_width'],
                     hatch='///', label='Paperback'),
            Rectangle((0, 0), 1, 1, facecolor=self.BOOK_COLORS[2], 
                     alpha=0.15,
                     edgecolor=self.STYLE_CONFIG['bar_edge_color'],
                     linewidth=self.STYLE_CONFIG['bar_edge_width'],
                     linestyle='--', label='Reading Gap (7+ days)')
        ]
        
        ax.legend(handles=legend_elements, 
                 loc='best',
                 fontsize=self.STYLE_CONFIG['legend_fontsize'],
                 title_fontsize=self.STYLE_CONFIG['legend_title_fontsize'],
                 frameon=True,
                 facecolor='black',
                 framealpha=0.7,
                 edgecolor='white')
        
        plt.tight_layout()
        
        save_path = Path(output_subdir) / filename
        plt.savefig(save_path, dpi=self.STYLE_CONFIG['dpi'], bbox_inches='tight', facecolor=fig.get_facecolor())
        
        print(f"Book completion timeline saved to {save_path}")
        return save_path
    
    def create_reading_pace_timeline(self, year=None, format_filter='all', output_dir='output', figsize=None):
        """
        Create a line plot showing cumulative pages read over time, stacked by format.
        """
        if self.reading_data is None or self.reading_data.empty:
            print("Cannot generate plot: Reading data is empty.")
            return
        
        sns.set_theme(**self.THEME_CONFIG)
        Path(output_dir).mkdir(exist_ok=True)
        
        if figsize is None:
            figsize = self.STYLE_CONFIG['figure_size']
        
        # Prepare data
        df = self.reading_data.copy()

        # Filter by format if specified
        if format_filter and format_filter != 'all':
            df = df[df['format'] == format_filter]
            if df.empty:
                print(f"No {format_filter} reading data found.")
                return
        
        # Filter by year if specified
        if year is not None:
            df = df[df['start_datetime'].dt.year == year].copy()
            if df.empty:
                print(f"No reading data found for year {year}.")
                return
            output_subdir = f'{output_dir}/{year}'
            filename = f'reading_pace_timeline_{year}.png'
        else:
            output_subdir = f'{output_dir}/overall'
            filename = 'reading_pace_timeline.png'
        
        Path(output_subdir).mkdir(parents=True, exist_ok=True)

        # Sort by datetime
        df = df.sort_values('start_datetime')

        # Extract date and compute daily pages read per book, preserving format info
        df['date'] = df['start_datetime'].dt.date
        
        # Get max page reached each day for each book, preserving format info
        daily_progress = df.groupby(['date', 'title', 'format'])['page'].max().reset_index()
        
        # Calculate daily pages read for each book
        daily_progress['pages_read'] = daily_progress.groupby(['title', 'format'])['page'].diff().fillna(daily_progress['page'])
        daily_progress.loc[daily_progress['pages_read'] < 0, 'pages_read'] = 0  # handle resets
        
        # Sum pages read across all books for each day, separated by format
        daily_totals = daily_progress.groupby(['date', 'format'])['pages_read'].sum().reset_index()
        daily_totals['date'] = pd.to_datetime(daily_totals['date'])
        
        # Ensure all dates are present for both formats (fill missing days with 0)
        date_range = pd.date_range(start=daily_totals['date'].min(), end=daily_totals['date'].max(), freq='D')
        
        # Create a complete date range dataframe for both formats
        complete_data = []
        for fmt in ['kindle', 'paperback']:
            fmt_data = daily_totals[daily_totals['format'] == fmt].copy()
            if not fmt_data.empty:
                fmt_data = fmt_data.set_index('date')['pages_read'].reindex(date_range, fill_value=0).reset_index()
                fmt_data.columns = ['date', 'pages_read']
                fmt_data['format'] = fmt
            else:
                # Create empty data for missing format
                fmt_data = pd.DataFrame({
                    'date': date_range,
                    'pages_read': 0,
                    'format': fmt
                })
            complete_data.append(fmt_data)
        
        # Combine both formats
        all_data = pd.concat(complete_data, ignore_index=True)
        
        # Pivot to have kindle and paperback as separate columns
        pivot_data = all_data.pivot(index='date', columns='format', values='pages_read').fillna(0)
        
        # Create cumulative sums for stacking
        if 'kindle' in pivot_data.columns:
            pivot_data['kindle_cumsum'] = pivot_data['kindle'].cumsum()
        else:
            pivot_data['kindle_cumsum'] = 0
            
        if 'paperback' in pivot_data.columns:
            pivot_data['paperback_cumsum'] = pivot_data['paperback'].cumsum()
        else:
            pivot_data['paperback_cumsum'] = 0
        
        # Reset index to get date as a column
        pivot_data = pivot_data.reset_index()
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot stacked areas
        has_kindle = 'kindle' in pivot_data.columns and pivot_data['kindle_cumsum'].max() > 0
        has_paperback = 'paperback' in pivot_data.columns and pivot_data['paperback_cumsum'].max() > 0
        
        if has_paperback:
            ax.fill_between(
                pivot_data['date'],
                0,
                pivot_data['paperback_cumsum'],
                color=self.PLOTS_COLORS[7],
                alpha=0.7,
                label='Paperback'
            )
        
        if has_kindle:
            # Stack kindle on top of paperback
            bottom = pivot_data['paperback_cumsum'] if has_paperback else 0
            ax.fill_between(
                pivot_data['date'],
                bottom,
                bottom + pivot_data['kindle_cumsum'],
                color=self.PLOTS_COLORS[8],
                alpha=0.7,
                label='Kindle'
            )
        
        title = 'Cumulative Pages Read'
        if year is not None:
            title += f' ({year})'
        
        ax.set_title(
            title,
            fontsize=self.STYLE_CONFIG['title_fontsize'],
            fontweight=self.STYLE_CONFIG['title_fontweight'],
            pad=self.STYLE_CONFIG['title_pad']
        )
        ax.set_ylabel(
            'Cumulative Pages Read',
            fontsize=self.STYLE_CONFIG['ylabel_fontsize'],
            labelpad=self.STYLE_CONFIG['ylabel_pad']
        )

        # Format x-axis dates based on year filter
        if year is not None:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        
        # Center x-axis labels and don't tilt them
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')
        
        # Set tick parameters
        ax.tick_params(axis='both', which='major', labelsize=self.STYLE_CONFIG['tick_labelsize'])

        # Set x-axis limits to span the full dataset
        ax.set_xlim(pivot_data['date'].min(), pivot_data['date'].max())
        
        # Grid styling
        ax.grid(axis='y', alpha=1.0)
        ax.grid(axis='x', alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
        
        # Add legend if any format has data
        if has_kindle or has_paperback:
            ax.legend(loc='upper left')
        
        plt.tight_layout()
        
        save_path = Path(output_subdir) / filename
        plt.savefig(save_path, dpi=self.STYLE_CONFIG['dpi'], bbox_inches='tight', facecolor=fig.get_facecolor())
        
        print(f"Reading pace timeline saved to {save_path}")
        return save_path