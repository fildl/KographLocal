import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import calendar
import matplotlib.patches as patches
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

class NumbersPlots:
    def __init__(self, reading_catalog_df):
        self.reading_catalog = reading_catalog_df

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
    PLOTS_COLORS = [
        '#ff6b9d', '#feca57', '#0abde3', '#ee5a6f', '#ff9ff3', '#54a0ff',
        '#00d2d3', '#ff9f43', '#1dd1a1', '#10ac84', '#2e86de', '#341f97',
        '#c44569', '#f8b500', '#6c5ce7', '#222f3e', '#8395a7'
    ]

    def create_yearly_books_count(self, output_dir='output', figsize=None):
        """Create a bar chart showing the number of books read each year"""
        if self.reading_catalog is None or self.reading_catalog.empty:
            print("Cannot generate plot: Reading catalog data is empty.")
            return
        
        sns.set_theme(**self.THEME_CONFIG)
        Path(output_dir).mkdir(exist_ok=True)
        
        if figsize is None:
            figsize = self.STYLE_CONFIG['figure_size']
        
        # Prepare data
        df = self.reading_catalog.copy()
        
        # Ensure we have a finish column
        if 'finish' not in df.columns:
            print("Error: 'finish' column not found in reading catalog data.")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Remove rows with missing finish dates (unfinished books)
        df = df.dropna(subset=['finish'])
        
        if df.empty:
            print("No books with finish dates found in reading catalog.")
            return
        
        # Convert finish column to datetime and extract year
        df['finish_date'] = pd.to_datetime(df['finish'], errors='coerce')
        df = df.dropna(subset=['finish_date'])  # Remove invalid dates
        df['finish_year'] = df['finish_date'].dt.year
        
        # Count books per year based on finish date
        yearly_counts = df.groupby('finish_year').size().reset_index(name='book_count')
        yearly_counts = yearly_counts.sort_values('finish_year')
        yearly_counts.rename(columns={'finish_year': 'year'}, inplace=True)
        
        # Create output directory
        output_subdir = f'{output_dir}/overall'
        Path(output_subdir).mkdir(parents=True, exist_ok=True)
        filename = 'books_count_yearly.png'
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create bars
        bars = ax.bar(yearly_counts['year'], yearly_counts['book_count'],
                     width=self.STYLE_CONFIG['bar_width'],
                     color=self.PLOTS_COLORS[9],
                     alpha=self.STYLE_CONFIG['bar_alpha'],
                     edgecolor=self.STYLE_CONFIG['bar_edge_color'],
                     linewidth=self.STYLE_CONFIG['bar_edge_width'])
        
        # Set title and labels
        ax.set_title(
            'Books Read per Year',
            fontfamily=self.STYLE_CONFIG['font_family'],
            fontsize=self.STYLE_CONFIG['title_fontsize'],
            fontweight=self.STYLE_CONFIG['title_fontweight'],
            pad=self.STYLE_CONFIG['title_pad']
        )
        ax.set_xlabel(
            'Year',
            fontfamily=self.STYLE_CONFIG['font_family'],
            fontsize=self.STYLE_CONFIG['ylabel_fontsize'],
            labelpad=self.STYLE_CONFIG['ylabel_pad']
        )
        ax.set_ylabel(
            'Number of Books',
            fontfamily=self.STYLE_CONFIG['font_family'],
            fontsize=self.STYLE_CONFIG['ylabel_fontsize'],
            labelpad=self.STYLE_CONFIG['ylabel_pad']
        )
        
        # Set tick parameters
        ax.tick_params(axis='both', which='major', labelfontfamily=self.STYLE_CONFIG['font_family'], labelsize=self.STYLE_CONFIG['tick_labelsize'])
        ax.tick_params(axis='x', rotation=0)
        
        # Set x-axis to show all years
        ax.set_xticks(yearly_counts['year'])
        ax.set_xticklabels([int(year) for year in yearly_counts['year']])
        
        # Grid styling
        ax.grid(axis='y', alpha=1.0)
        ax.grid(axis='x', alpha=0)
        ax.set_axisbelow(True)
        
        # Add total text annotations on top of bars
        y_offset = yearly_counts['book_count'].max() * self.STYLE_CONFIG['total_text_offset_ratio']
        for i, (year, count) in enumerate(zip(yearly_counts['year'], yearly_counts['book_count'])):
            ax.text(
                year,
                count + y_offset,
                str(int(count)),
                ha='center',
                va='bottom',
                fontfamily=self.STYLE_CONFIG['font_family'],
                fontsize=self.STYLE_CONFIG['total_text_fontsize'],
                fontweight=self.STYLE_CONFIG['total_text_fontweight'],
                color=self.STYLE_CONFIG['total_text_color']
            )
        
        # Add summary statistics
        total_books = yearly_counts['book_count'].sum()
        avg_books = yearly_counts['book_count'].mean()
        years_span = len(yearly_counts)
        
        # Add text box with statistics
        stats_text = f"Total Books: {total_books}\nAverage per Year: {avg_books:.1f}\nYears Tracked: {years_span}"
        ax.text(0.02, 0.98, stats_text,
               transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
               fontfamily=self.STYLE_CONFIG['font_family'],
               fontsize=self.STYLE_CONFIG['total_text_fontsize'],
               color='white')
        
        plt.tight_layout()
        
        save_path = Path(output_subdir) / filename
        plt.savefig(save_path, dpi=self.STYLE_CONFIG['dpi'], bbox_inches='tight', facecolor=fig.get_facecolor())
        
        print(f"Yearly books count plot saved to {save_path}")
        return save_path

    def create_monthly_books_count(self, year=None, output_dir='output', figsize=None):
        """Create a bar chart showing the number of books read each month (optionally for a specific year)"""
        if self.reading_catalog is None or self.reading_catalog.empty:
            print("Cannot generate plot: Reading catalog data is empty.")
            return
        
        sns.set_theme(**self.THEME_CONFIG)
        Path(output_dir).mkdir(exist_ok=True)
        
        if figsize is None:
            figsize = self.STYLE_CONFIG['figure_size']
        
        # Prepare data
        df = self.reading_catalog.copy()
        
        # Ensure we have a finish column
        if 'finish' not in df.columns:
            print("Error: 'finish' column not found in reading catalog data.")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Remove rows with missing finish dates (unfinished books)
        df = df.dropna(subset=['finish'])
        
        if df.empty:
            print("No books with finish dates found in reading catalog.")
            return
        
        # Convert finish column to datetime and extract year/month
        df['finish_date'] = pd.to_datetime(df['finish'], errors='coerce')
        df = df.dropna(subset=['finish_date'])  # Remove invalid dates
        df['finish_year'] = df['finish_date'].dt.year
        df['finish_month'] = df['finish_date'].dt.month
        
        # Filter by year if specified
        if year is not None:
            df = df[df['finish_year'] == year]
            if df.empty:
                print(f"No finished books found for year {year}.")
                return
            output_subdir = f'{output_dir}/{year}'
            filename = f'books_count_monthly_{year}.png'
            title = f'Books Read per Month ({year})'
            
            # Count books per month for specific year
            monthly_counts = df.groupby('finish_month').size().reset_index(name='book_count')
            monthly_counts.rename(columns={'finish_month': 'month'}, inplace=True)
            # Ensure all 12 months are present
            all_months = pd.DataFrame({'month': range(1, 13)})
            monthly_counts = all_months.merge(monthly_counts, on='month', how='left').fillna(0)
            monthly_counts['book_count'] = monthly_counts['book_count'].astype(int)
            
            # Create month names for x-axis
            month_names = [calendar.month_abbr[int(m)] for m in monthly_counts['month']]
            x_labels = month_names
            
        else:
            output_subdir = f'{output_dir}/overall'
            filename = 'books_count_monthly.png'
            title = 'Books Read per Month'
            
            # Create timeline across all years
            min_year = df['finish_year'].min()
            max_year = df['finish_year'].max()
            
            # Generate all year-month combinations
            all_periods = []
            for yr in range(min_year, max_year + 1):
                for month in range(1, 13):
                    all_periods.append({
                        'year': yr,
                        'month': month,
                        'year_month': f"{yr}-{month:02d}"
                    })
            
            complete_periods_df = pd.DataFrame(all_periods)
            
            # Count books per year-month
            period_counts = df.groupby(['finish_year', 'finish_month']).size().reset_index(name='book_count')
            period_counts.rename(columns={'finish_year': 'year', 'finish_month': 'month'}, inplace=True)
            
            # Merge with complete periods, filling missing with 0
            monthly_counts = complete_periods_df.merge(
                period_counts, 
                on=['year', 'month'], 
                how='left'
            ).fillna(0)
            monthly_counts['book_count'] = monthly_counts['book_count'].astype(int)
            monthly_counts = monthly_counts.sort_values(['year', 'month'])
            
            # Create labels - show only first month of each year with YYYY format
            x_labels = []
            for i, row in monthly_counts.iterrows():
                if row['month'] == 1:  # Show only January of each year
                    x_labels.append(str(int(row['year'])))
                else:
                    x_labels.append("")
        
        Path(output_subdir).mkdir(parents=True, exist_ok=True)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        if year is not None:
            # Bar chart for specific year
            bars = ax.bar(range(len(monthly_counts)), monthly_counts['book_count'],
                        width=self.STYLE_CONFIG['bar_width'],
                        color=self.PLOTS_COLORS[10],
                        alpha=self.STYLE_CONFIG['bar_alpha'],
                        edgecolor=self.STYLE_CONFIG['bar_edge_color'],
                        linewidth=self.STYLE_CONFIG['bar_edge_width'])
            
            ax.set_xticks(range(len(monthly_counts)))
            ax.set_xticklabels(x_labels)
        else:
            # Line plot for timeline across years
            x_positions = range(len(monthly_counts))
            ax.plot(x_positions, monthly_counts['book_count'],
                color=self.PLOTS_COLORS[10],
                linewidth=2,
                marker='o',
                markersize=3,
                alpha=0.8)
            ax.fill_between(x_positions, monthly_counts['book_count'],
                        alpha=0.3,
                        color=self.PLOTS_COLORS[10])
            
            # Set x-axis ticks and labels
            tick_positions = [i for i, label in enumerate(x_labels) if label != ""]
            tick_labels = [label for label in x_labels if label != ""]
            
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, ha='right')
            ax.set_xlim(0, len(monthly_counts) - 1)
        
        # Set title and labels
        ax.set_title(
            title,
            fontfamily=self.STYLE_CONFIG['font_family'],
            fontsize=self.STYLE_CONFIG['title_fontsize'],
            fontweight=self.STYLE_CONFIG['title_fontweight'],
            pad=self.STYLE_CONFIG['title_pad']
        )
        ax.set_xlabel(
            'Month' if year is not None else 'Year',
            fontfamily=self.STYLE_CONFIG['font_family'],
            fontsize=self.STYLE_CONFIG['ylabel_fontsize'],
            labelpad=self.STYLE_CONFIG['ylabel_pad']
        )
        ax.set_ylabel(
            'Number of Books',
            fontfamily=self.STYLE_CONFIG['font_family'],
            fontsize=self.STYLE_CONFIG['ylabel_fontsize'],
            labelpad=self.STYLE_CONFIG['ylabel_pad']
        )
        
        # Set tick parameters
        ax.tick_params(axis='both', which='major', labelfontfamily=self.STYLE_CONFIG['font_family'], labelsize=self.STYLE_CONFIG['tick_labelsize'])
        if year is not None:
            ax.tick_params(axis='x', rotation=0)
        
        # Set y-axis formatting
        ax.set_ylim(bottom=0)
        max_count = monthly_counts['book_count'].max()
        if max_count > 0:
            ax.set_ylim(top=max_count * 1.15)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        # Grid styling
        ax.grid(axis='y', alpha=1.0)
        ax.grid(axis='x', alpha=0)
        ax.set_axisbelow(True)
        
        # Add total text annotations on top of bars (only for single year)
        if year is not None:
            y_offset = max_count * self.STYLE_CONFIG['total_text_offset_ratio'] if max_count > 0 else 0.1
            for i, count in enumerate(monthly_counts['book_count']):
                if count > 0:
                    ax.text(
                        i,
                        count + y_offset,
                        str(int(count)),
                        ha='center',
                        va='bottom',
                        fontfamily=self.STYLE_CONFIG['font_family'],
                        fontsize=self.STYLE_CONFIG['total_text_fontsize'],
                        fontweight=self.STYLE_CONFIG['total_text_fontweight'],
                        color=self.STYLE_CONFIG['total_text_color']
                    )
        
        # Add summary statistics
        total_books = monthly_counts['book_count'].sum()
        avg_books = monthly_counts['book_count'].mean()
        
        if year is not None:
            stats_text = f"Total Books ({year}): {total_books}\nAverage per Month: {avg_books:.1f}"
        else:
            stats_text = f"Total Books: {total_books}\nAverage per Month: {avg_books:.1f}"
        
        ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
            fontfamily=self.STYLE_CONFIG['font_family'],
            fontsize=self.STYLE_CONFIG['total_text_fontsize'],
            color='white')
        
        plt.tight_layout()
        
        save_path = Path(output_subdir) / filename
        plt.savefig(save_path, dpi=self.STYLE_CONFIG['dpi'], bbox_inches='tight', facecolor=fig.get_facecolor())
        
        print(f"Monthly books count plot saved to {save_path}")
        plt.close()
        return save_path
    
    def create_yearly_books_bought_vs_read(self, year_start=None, output_dir='output', figsize=None, debug=False):
        """Create a stacked area plot showing books bought vs read each year"""
        if self.reading_catalog is None or self.reading_catalog.empty:
            print("Cannot generate plot: Reading catalog data is empty.")
            return
        
        sns.set_theme(**self.THEME_CONFIG)
        Path(output_dir).mkdir(exist_ok=True)
        
        if figsize is None:
            figsize = self.STYLE_CONFIG['figure_size']
        
        # Prepare data
        df = self.reading_catalog.copy()
        
        # Check required columns
        required_cols = ['purchase', 'finish']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Convert date columns to datetime
        df['purchase_date'] = pd.to_datetime(df['purchase'], errors='coerce')
        df['finish_date'] = pd.to_datetime(df['finish'], errors='coerce')
        
        # Extract years
        df['purchase_year'] = df['purchase_date'].dt.year
        df['finish_year'] = df['finish_date'].dt.year
        
        # Filter by year_start if specified
        if year_start is not None:
            df = df[df['purchase_year'] >= year_start]
            if df.empty:
                print(f"No books found with purchase date from {year_start} onwards.")
                return
        
        # Remove rows without purchase dates (we need this for bought books)
        df_with_purchase = df.dropna(subset=['purchase_date'])
        if df_with_purchase.empty:
            print("No books with purchase dates found.")
            return
        
        # Count books bought per year
        bought_counts = df_with_purchase.groupby('purchase_year').size().reset_index(name='books_bought')
        bought_counts.rename(columns={'purchase_year': 'year'}, inplace=True)
        
        # Count books read per year (only for books with finish dates) 
        df_finished = df.dropna(subset=['finish_date'])
        if not df_finished.empty:
            read_counts = df_finished.groupby('finish_year').size().reset_index(name='books_read')
            read_counts.rename(columns={'finish_year': 'year'}, inplace=True)
        else:
            read_counts = pd.DataFrame(columns=['year', 'books_read'])
        
        # Get all years from both bought and read
        all_years = sorted(set(bought_counts['year'].dropna()) | set(read_counts['year'].dropna()))
        if not all_years:
            print("No valid years found in the data.")
            return
        
        # Create complete year range and merge data
        complete_years = pd.DataFrame({'year': all_years})
        yearly_data = complete_years.merge(bought_counts, on='year', how='left')
        yearly_data = yearly_data.merge(read_counts, on='year', how='left')
        yearly_data = yearly_data.fillna(0)
        yearly_data['books_bought'] = yearly_data['books_bought'].astype(int)
        yearly_data['books_read'] = yearly_data['books_read'].astype(int)
        yearly_data = yearly_data.sort_values('year')
        
        # Calculate percentages for 100% stacked area (bought vs read for each year)
        yearly_data['total_activity'] = yearly_data['books_bought'] + yearly_data['books_read']
        yearly_data['bought_pct'] = np.where(yearly_data['total_activity'] > 0, 
                                           yearly_data['books_bought'] / yearly_data['total_activity'] * 100, 0)
        yearly_data['read_pct'] = np.where(yearly_data['total_activity'] > 0, 
                                          yearly_data['books_read'] / yearly_data['total_activity'] * 100, 0)
        
        # Debug: Print yearly breakdown
        if debug:
            print("\n=== YEARLY BOOKS BREAKDOWN ===")
            for _, row in yearly_data.iterrows():
                year = int(row['year'])
                bought = int(row['books_bought'])
                read = int(row['books_read'])
                total_activity = int(row['total_activity'])
                bought_pct = row['bought_pct']
                read_pct = row['read_pct']
                
                print(f"{year}: Bought={bought}, Read={read}, Total Activity={total_activity} | Bought={bought_pct:.1f}%, Read={read_pct:.1f}%")
            
            total_bought_debug = yearly_data['books_bought'].sum()
            total_read_debug = yearly_data['books_read'].sum()
            total_activity_debug = yearly_data['total_activity'].sum()
            print(f"\nOVERALL: Bought={total_bought_debug}, Read={total_read_debug}, Total Activity={total_activity_debug}")
            print("=" * 50)
        
        # Create output directory and filename
        output_subdir = f'{output_dir}/overall'
        filename = 'books_bought_vs_read.png'
        title_suffix = ''
        
        Path(output_subdir).mkdir(parents=True, exist_ok=True)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create 100% stacked area plot
        ax.fill_between(yearly_data['year'], 0, yearly_data['read_pct'],
                       color=self.PLOTS_COLORS[12], alpha=0.9, label='Books Read')
        ax.fill_between(yearly_data['year'], yearly_data['read_pct'], 100,
                       color=self.PLOTS_COLORS[11], alpha=0.8, label='Books Bought (Unread)')
        
        # Add line plots for better visibility
        ax.plot(yearly_data['year'], yearly_data['read_pct'],
               color=self.PLOTS_COLORS[12], linewidth=2, alpha=1.0)
        
        # Set title and labels
        ax.set_title(
            f'Bought vs Read Books per Year{title_suffix}',
            fontfamily=self.STYLE_CONFIG['font_family'],
            fontsize=self.STYLE_CONFIG['title_fontsize'],
            fontweight=self.STYLE_CONFIG['title_fontweight'],
            pad=self.STYLE_CONFIG['title_pad']
        )
        ax.set_xlabel(
            'Year',
            fontfamily=self.STYLE_CONFIG['font_family'],
            fontsize=self.STYLE_CONFIG['ylabel_fontsize'],
            labelpad=self.STYLE_CONFIG['ylabel_pad']
        )
        ax.set_ylabel(
            'Percentage (%)',
            fontfamily=self.STYLE_CONFIG['font_family'],
            fontsize=self.STYLE_CONFIG['ylabel_fontsize'],
            labelpad=self.STYLE_CONFIG['ylabel_pad']
        )
        
        # Set tick parameters
        ax.tick_params(axis='both', which='major', labelfontfamily=self.STYLE_CONFIG['font_family'], labelsize=self.STYLE_CONFIG['tick_labelsize'])
        ax.tick_params(axis='x', rotation=0)
        
        # Set x-axis to show range from first to last year
        min_year = yearly_data['year'].min()
        max_year = yearly_data['year'].max()
        ax.set_xlim(min_year, max_year)
        ax.set_xticks(yearly_data['year'])
        ax.set_xticklabels([int(year) for year in yearly_data['year']])
        
        # Set y-axis to 0-100% range
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}%'))
        
        # Grid styling
        ax.grid(axis='y', alpha=1.0)
        ax.grid(axis='x', alpha=0)
        ax.set_axisbelow(True)
        
        # Add legend
        ax.legend(loc='upper left', 
                title_fontsize=self.STYLE_CONFIG['legend_title_fontsize'],
                fontsize=self.STYLE_CONFIG['legend_fontsize'],
                frameon=True, facecolor='black', edgecolor='white', framealpha=0.7,
                prop={'family': self.STYLE_CONFIG['font_family']})
        
        # Add summary statistics
        total_bought = yearly_data['books_bought'].sum()
        total_read = yearly_data['books_read'].sum()
        avg_bought = yearly_data['books_bought'].mean()
        avg_read = yearly_data['books_read'].mean()
        
        # Calculate reading rate
        reading_rate = (total_read / total_bought * 100) if total_bought > 0 else 0
        
        stats_text = f"Total Bought: {total_bought}\nTotal Read: {total_read}\nReading Rate: {reading_rate:.1f}%\nAvg Bought/Year: {avg_bought:.1f}\nAvg Read/Year: {avg_read:.1f}"
        
        ax.text(0.98, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
            fontfamily=self.STYLE_CONFIG['font_family'],
            fontsize=self.STYLE_CONFIG['total_text_fontsize'],
            color='white')
        
        plt.tight_layout()
        
        save_path = Path(output_subdir) / filename
        plt.savefig(save_path, dpi=self.STYLE_CONFIG['dpi'], bbox_inches='tight', facecolor=fig.get_facecolor())
        
        print(f"Yearly books bought vs read plot saved to {save_path}")
        plt.close()
        return save_path
    
    def create_statistics_summary(self, year=None, output_dir='output', figsize=None):
        """Create a visual statistics summary with squircle/superellipse shapes
        
        Args:
            year: Optional year to filter data
            output_dir: Output directory for the plot
            figsize: Figure size tuple
        """
        if self.reading_catalog is None or self.reading_catalog.empty:
            print("Cannot generate plot: Reading catalog is empty.")
            return
        
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path as MplPath
        
        sns.set_theme(**self.THEME_CONFIG)
        Path(output_dir).mkdir(exist_ok=True)
        
        if figsize is None:
            figsize = (16, 14)
        
        # Work with reading catalog
        catalog_df = self.reading_catalog.copy()
        
        # Filter by year if specified
        if year is not None:
            catalog_df['finish_date'] = pd.to_datetime(catalog_df['finish'], errors='coerce')
            catalog_df = catalog_df.dropna(subset=['finish_date'])
            catalog_df['finish_year'] = catalog_df['finish_date'].dt.year
            catalog_df = catalog_df[catalog_df['finish_year'] == year]
            
            if catalog_df.empty:
                print(f"No books found for year {year}.")
                return
                
            output_subdir = f'{output_dir}/{year}'
            filename = f'statistics_summary_{year}.png'
            title = f'Reading Statistics Summary ({year})'
        else:
            catalog_df['finish_date'] = pd.to_datetime(catalog_df['finish'], errors='coerce')
            catalog_df = catalog_df.dropna(subset=['finish_date'])
            output_subdir = f'{output_dir}/overall'
            filename = 'statistics_summary.png'
            title = 'Reading Statistics Summary'
        
        Path(output_subdir).mkdir(parents=True, exist_ok=True)
        
        # Calculate statistics
        total_books = len(catalog_df)
        
        # Average pages per book
        if 'pages' in catalog_df.columns:
            catalog_df['pages_num'] = pd.to_numeric(catalog_df['pages'], errors='coerce')
            avg_pages_per_book = catalog_df['pages_num'].mean()
        else:
            avg_pages_per_book = 0
        
        # Average pages per day
        if year is not None:
            days_in_year = 366 if pd.Timestamp(year, 1, 1).is_leap_year else 365
            total_pages_year = catalog_df['pages_num'].sum()
            avg_pages_per_day = total_pages_year / days_in_year
        else:
            # Calculate across all time
            if not catalog_df.empty:
                min_date = catalog_df['finish_date'].min()
                max_date = catalog_df['finish_date'].max()
                total_days = (max_date - min_date).days + 1
                total_pages = catalog_df['pages_num'].sum()
                avg_pages_per_day = total_pages / total_days if total_days > 0 else 0
            else:
                avg_pages_per_day = 0
        
        # Fiction vs Non-Fiction
        if 'fiction' in catalog_df.columns:
            fiction_count = (catalog_df['fiction'] == True).sum()
            nonfiction_count = (catalog_df['fiction'] == False).sum()
            total_categorized = fiction_count + nonfiction_count
            if total_categorized > 0:
                fiction_pct = (fiction_count / total_categorized) * 100
                nonfiction_pct = (nonfiction_count / total_categorized) * 100
            else:
                fiction_pct = nonfiction_pct = 0
        else:
            fiction_pct = nonfiction_pct = 0
        
        # Books per quarter
        if year is not None:
            catalog_df['quarter'] = catalog_df['finish_date'].dt.quarter
            books_per_quarter = catalog_df.groupby('quarter').size().reindex([1, 2, 3, 4], fill_value=0)
        else:
            books_per_quarter = pd.Series([0, 0, 0, 0], index=[1, 2, 3, 4])
        
        # Helper function to generate superellipse/squircle coordinates
        def generate_superellipse(center_x, center_y, width, height, n=4, num_points=500):
            """Generate superellipse coordinates using formula |x/a|^n + |y/b|^n = 1"""
            theta = np.linspace(0, 2*np.pi, num_points)
            a = width / 2
            b = height / 2
            
            # Parametric form
            x = np.sign(np.cos(theta)) * a * np.abs(np.cos(theta))**(2/n)
            y = np.sign(np.sin(theta)) * b * np.abs(np.sin(theta))**(2/n)
            
            return x + center_x, y + center_y
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(-12, 12)
        ax.set_ylim(-10, 12)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 1. CENTRAL SUPERELLIPSE (top center) - Main statistics
        x, y = generate_superellipse(0, 4, 10, 6, n=2.5)
        vertices = np.column_stack([x, y])
        codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(vertices)-2) + [MplPath.CLOSEPOLY]
        path = MplPath(vertices, codes)
        patch = PathPatch(path, facecolor=self.PLOTS_COLORS[10], alpha=0.8, 
                        edgecolor='white', linewidth=3)
        ax.add_patch(patch)
        
        # Add central statistics
        ax.text(0, 6, f'{total_books}', ha='center', va='center',
            fontfamily=self.STYLE_CONFIG['font_family'],
            fontsize=42, fontweight='bold', color='white')
        ax.text(0, 4.8, 'Books Read', ha='center', va='center',
            fontfamily=self.STYLE_CONFIG['font_family'],
            fontsize=16, color='lightgray')
        
        ax.text(-3.5, 2.5, f'{avg_pages_per_book:.0f}', ha='center', va='center',
            fontfamily=self.STYLE_CONFIG['font_family'],
            fontsize=28, fontweight='bold', color='white')
        ax.text(-3.5, 1.5, 'Avg Pages/Book', ha='center', va='center',
            fontfamily=self.STYLE_CONFIG['font_family'],
            fontsize=12, color='lightgray')
        
        ax.text(3.5, 2.5, f'{avg_pages_per_day:.1f}', ha='center', va='center',
            fontfamily=self.STYLE_CONFIG['font_family'],
            fontsize=28, fontweight='bold', color='white')
        ax.text(3.5, 1.5, 'Avg Pages/Day', ha='center', va='center',
            fontfamily=self.STYLE_CONFIG['font_family'],
            fontsize=12, color='lightgray')
        
        # 2. LEFT SQUIRCLE (bottom left) - Books per Quarter
        x, y = generate_superellipse(-5, -3, 7, 7, n=4)
        vertices = np.column_stack([x, y])
        codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(vertices)-2) + [MplPath.CLOSEPOLY]
        path = MplPath(vertices, codes)
        patch = PathPatch(path, facecolor=self.PLOTS_COLORS[2], alpha=0.8, 
                        edgecolor='white', linewidth=3)
        ax.add_patch(patch)
        
        # Add quarterly data
        ax.text(-5, -0.5, 'Books per Quarter', ha='center', va='center',
            fontfamily=self.STYLE_CONFIG['font_family'],
            fontsize=14, fontweight='bold', color='white')
        
        quarter_labels = ['Q1', 'Q2', 'Q3', 'Q4']
        y_positions = [-2, -3, -4, -5]
        for i, (q_label, q_count, y_pos) in enumerate(zip(quarter_labels, books_per_quarter, y_positions)):
            ax.text(-6.5, y_pos, q_label, ha='right', va='center',
                fontfamily=self.STYLE_CONFIG['font_family'],
                fontsize=16, fontweight='bold', color='white')
            ax.text(-5, y_pos, f'{q_count}', ha='center', va='center',
                fontfamily=self.STYLE_CONFIG['font_family'],
                fontsize=20, fontweight='bold', color='white')
        
        # 3. RIGHT SQUIRCLE (bottom right) - Fiction vs Non-Fiction
        x, y = generate_superellipse(5, -3, 7, 7, n=4)
        vertices = np.column_stack([x, y])
        codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(vertices)-2) + [MplPath.CLOSEPOLY]
        path = MplPath(vertices, codes)
        patch = PathPatch(path, facecolor=self.PLOTS_COLORS[4], alpha=0.8, 
                        edgecolor='white', linewidth=3)
        ax.add_patch(patch)
        
        # Add fiction/non-fiction data
        ax.text(5, -0.5, 'Fiction vs Non-Fiction', ha='center', va='center',
            fontfamily=self.STYLE_CONFIG['font_family'],
            fontsize=14, fontweight='bold', color='white')
        
        # Create mini pie chart representation
        if fiction_pct + nonfiction_pct > 0:
            ax.text(5, -2.5, f'{fiction_pct:.0f}%', ha='center', va='center',
                fontfamily=self.STYLE_CONFIG['font_family'],
                fontsize=32, fontweight='bold', color='white')
            ax.text(5, -3.5, 'Fiction', ha='center', va='center',
                fontfamily=self.STYLE_CONFIG['font_family'],
                fontsize=14, color='lightgray')
            
            ax.text(5, -4.8, f'{nonfiction_pct:.0f}%', ha='center', va='center',
                fontfamily=self.STYLE_CONFIG['font_family'],
                fontsize=32, fontweight='bold', color='white')
            ax.text(5, -5.8, 'Non-Fiction', ha='center', va='center',
                fontfamily=self.STYLE_CONFIG['font_family'],
                fontsize=14, color='lightgray')
        else:
            ax.text(5, -3, 'No data', ha='center', va='center',
                fontfamily=self.STYLE_CONFIG['font_family'],
                fontsize=18, color='lightgray', style='italic')
        
        # Add title
        fig.text(0.5, 0.96, title, ha='center', va='top',
                fontfamily=self.STYLE_CONFIG['font_family'],
                fontsize=self.STYLE_CONFIG['title_fontsize'] + 4,
                fontweight=self.STYLE_CONFIG['title_fontweight'],
                color='white')
        
        plt.tight_layout()
        
        save_path = Path(output_subdir) / filename
        plt.savefig(save_path, dpi=self.STYLE_CONFIG['dpi'], 
                bbox_inches='tight', facecolor=fig.get_facecolor())
        
        print(f"Statistics summary saved to {save_path}")
        plt.close()
        return save_path