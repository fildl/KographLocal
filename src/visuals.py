import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import calendar
import numpy as np

class Visualizer:
    """
    Handles generation of interactive Plotly charts.
    """
    
    # Theme Configuration
    THEME_COLORS = {
        'background': '#1c1c1c',
        'paper': '#1c1c1c',
        'text': '#e0e0e0',
        'grid': '#333333',
        'primary': '#00d2d3',    # Cyan
        'secondary': '#ff9ff3',  # Pink
        'accent': '#feca57',     # Yellow
        'subtext': '#aaaaaa',
        'gradient': ['#00d2d3', '#54a0ff', '#5f27cd']
    }

    # Format Colors
    FORMAT_COLORS = {
        'ebook': '#00d2d3',    # Cyan
        'paperback': '#feca57', # Yellow
        'audiobook': '#ff9ff3', # Pink
        'kindle': '#00d2d3',   # Fallback
        'paper': '#feca57'     # Fallback
    }
    
    # Standard Plot Dimensions
    PLOT_WIDTH = 1200
    PLOT_HEIGHT = 700

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self._set_global_theme()

    def _set_global_theme(self):
        """Configure global Plotly defaults."""
        import plotly.io as pio
        import plotly.graph_objects as go
        
        # Modify the default template directly
        pio.templates["kograph_dark"] = go.layout.Template(
            layout=go.Layout(
                paper_bgcolor=self.THEME_COLORS['paper'],
                plot_bgcolor=self.THEME_COLORS['background'],
                font=dict(
                    family="Inter, sans-serif",
                    color=self.THEME_COLORS['text']
                ),
                xaxis=dict(gridcolor=self.THEME_COLORS['grid']),
                yaxis=dict(gridcolor=self.THEME_COLORS['grid']),
                colorway=self.THEME_COLORS['gradient']
            )
        )
        
        # Merge with plotly_dark
        pio.templates.default = "plotly_dark+kograph_dark"

    def plot_weekly_activity(self, year: int = None):
        """
        Stacked Bar chart of total reading hours per week, split by format.
        Args:
            year (int, optional): Filter data for a specific year.
        """
        df = self.data.copy()
        
        # Filter by year if specified
        if year:
            df = df[df['year'] == year]
        
        title = 'Weekly Reading Activity'

        if df.empty:
            print(f"Warning: No data found for year {year}")
            return None

        # Ensure week column exists
        if 'start_datetime' in df.columns:
            df['week'] = df['start_datetime'].dt.to_period('W').dt.start_time
        elif 'date' in df.columns:
            df['start_datetime'] = pd.to_datetime(df['date'])
            df['week'] = df['start_datetime'].dt.to_period('W').dt.start_time
        
        # Aggregate duration and book titles per week and format
        aggregated = df.groupby(['week', 'format']).agg({
            'duration': 'sum',
            'title': lambda x: '<br>'.join(sorted(list(set(x)))[:5]) + ('...' if len(set(x)) > 5 else '')
        }).reset_index()
        
        aggregated['hours'] = aggregated['duration'] / 3600
        aggregated['books_list'] = aggregated['title']
        
        aggregated['formatted_time'] = aggregated.apply(
            lambda x: f"{int(x['duration'] // 3600)}h {int((x['duration'] % 3600) // 60)}m", 
            axis=1
        )
        
        # Create Stacked Bar Plot
        fig = px.bar(
            aggregated, 
            x='week', 
            y='hours',
            color='format', # Stack by format
            title=title,
            labels={'hours': 'Hours Read', 'week': 'Week', 'format': 'Format'},
            custom_data=['formatted_time', 'books_list', 'format'],
            color_discrete_map=self.FORMAT_COLORS # Apply explicit colors
        )
        
        # Styling
        fig.update_layout(
            paper_bgcolor=self.THEME_COLORS['paper'],
            plot_bgcolor=self.THEME_COLORS['background'],
            font_color=self.THEME_COLORS['text'],
            showlegend=True, # Show legend for formats
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            title_x=0.5,
            title_xanchor='center',
            title_y=0.95,
            hovermode="x", # Unified hover might be better for stacked bars? Or sticking to x
            # For stacked bars, "hovermode='x'" shows all stack items at that x.
            width=self.PLOT_WIDTH,
            height=self.PLOT_HEIGHT,
            margin=dict(t=80, l=50, r=50, b=50), # Consistent margins
            xaxis=dict(
                showgrid=False,
                title=None
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=self.THEME_COLORS['grid'],
                title='Reading Time (hours)'
            )
        )
        
        fig.update_traces(
            marker_line_width=0,
            # Use customdata[0] for formatted time, customdata[1] for books
            hovertemplate="<br><b>%{customdata[2]}</b><br><b>Time</b>: %{customdata[0]}<br><b>Books:</b><br>%{customdata[1]}<extra></extra>",
            hoverlabel=dict(bgcolor="black") # Black background
        )
        
        return fig

    def plot_reading_calendar(self, year: int = None):
        """
        3x4 Month Grid Scatter plot for reading habits.
        """
        df = self.data.copy()
        
        if year:
            df = df[df['year'] == year]
        else:
            target_year = df['year'].max()
            df = df[df['year'] == target_year]
            year = target_year

        title = 'Reading Calendar'
        
        # Exclude Paperback data (synthetic daily sessions don't reflect actual habits)
        if 'format' in df.columns:
            df = df[df['format'] != 'paperback']

        if df.empty:
            return None
            
        # Prepare subplots
        fig = make_subplots(
            rows=3, cols=4, 
            subplot_titles=[calendar.month_name[i] for i in range(1, 13)],
            vertical_spacing=0.08,
            horizontal_spacing=0.03
        )

        daily = df.groupby('date').agg({
            'duration': 'sum',
            'title': lambda x: '<br>'.join(sorted(list(set(x)))[:5]) + ('...' if len(set(x)) > 5 else '')
        }).reset_index()
        
        daily['minutes'] = daily['duration'] / 60
        daily['date'] = pd.to_datetime(daily['date'])
        daily['books_list'] = daily['title']
        
        # Max reading for color normalization
        max_reading = daily['minutes'].max() if not daily.empty else 1
        
        # Find global max day for highlighting
        max_day_date = daily.loc[daily['minutes'].idxmax(), 'date'] if not daily.empty else None

        # Iterate through months
        for month in range(1, 13):
            row = (month - 1) // 4 + 1
            col = (month - 1) % 4 + 1
            
            # Generate full dates for this month
            _, num_days = calendar.monthrange(year, month)
            dates = pd.date_range(start=f"{year}-{month:02d}-01", end=f"{year}-{month:02d}-{num_days}")
            
            month_df = pd.DataFrame({'date': dates})
            month_df = month_df.merge(daily[['date', 'minutes', 'books_list']], on='date', how='left').fillna({'minutes': 0, 'books_list': ''})
            
            # Coordinates
            month_df['day_of_week'] = month_df['date'].dt.dayofweek # 0=Mon
            
            # Week of month (0-based index)
            first_day_weekday = month_df.iloc[0]['date'].dayofweek
            month_df['day_idx'] = month_df['date'].dt.day - 1
            month_df['week_of_month'] = (month_df['day_idx'] + first_day_weekday) // 7
            
            # Formatting
            month_df['formatted_time'] = month_df.apply(
                lambda x: f"{int(x['minutes'] // 60)}h {int(x['minutes'] % 60)}m" if x['minutes'] > 0 else "0m", 
                axis=1
            )
            
            month_df['hover_text'] = month_df.apply(
                lambda x: (
                    f"<b>{x['date'].strftime('%b %d')}</b><br>{x['formatted_time']}<br><br><b>Books:</b><br>{x['books_list']}"
                    if x['minutes'] > 0 else
                    f"<b>{x['date'].strftime('%b %d')}</b><br>No Reading"
                ), 
                axis=1
            )
            
            # Show legend (colorbar) only on the last chart (Dec)
            show_scale = (month == 12)
            
            # Split into Active (Reading) and Inactive (Empty)
            active_df = month_df[month_df['minutes'] > 0].copy()
            inactive_df = month_df[month_df['minutes'] == 0].copy()

            # 1. Inactive Trace (No Hover)
            fig.add_trace(
                go.Scatter(
                    x=inactive_df['day_of_week'],
                    y=5 - inactive_df['week_of_month'],
                    mode='markers',
                    marker=dict(
                        size=14,
                        color='#333333', # Empty color
                        line=dict(width=1, color=self.THEME_COLORS['background'])
                    ),
                    hoverinfo='skip',
                    showlegend=False
                ),
                row=row, col=col
            )

            # 2. Active Trace (With Hover)
            if not active_df.empty:
                # Line styling logic: Highlight Max Day
                # We need to compute line colors and widths row by row because Plotly expects arrays or scalar
                # Or we can use apply
                def get_line_style(row_date):
                    if max_day_date and row_date == max_day_date:
                        return self.THEME_COLORS['accent'], 2 # Accent color, thicker
                    return self.THEME_COLORS['text'], 1 # Default, thin

                lines = active_df['date'].apply(get_line_style)
                active_df['line_color'] = lines.apply(lambda x: x[0])
                active_df['line_width'] = lines.apply(lambda x: x[1])
                
                fig.add_trace(
                    go.Scatter(
                        x=active_df['day_of_week'],
                        y=5 - active_df['week_of_month'], 
                        mode='markers',
                        marker=dict(
                            size=14,
                            color=active_df['minutes'],
                            colorscale=[
                                [0, '#333333'], # Should not happen for active
                                [0.01, '#2d3436'],
                                [1, self.THEME_COLORS['primary']]
                            ],
                            cmin=0,
                            cmax=max_reading,
                            showscale=show_scale,
                            colorbar=dict(
                                title=dict(
                                    text="Reading Time (minutes)",
                                    side="right"
                                ),
                                thickness=15,
                                len=0.7,
                                y=0.5
                            ) if show_scale else None,
                            line=dict(
                                width=active_df['line_width'], 
                                color=active_df['line_color']
                            )
                        ),
                        text=active_df['hover_text'],
                        hoverinfo='text',
                        showlegend=False
                    ),
                    row=row, col=col
                )
            
            # Clean up axes for this subplot
            fig.update_xaxes(
                showgrid=False, zeroline=False, showticklabels=False, 
                range=[-0.5, 6.5], row=row, col=col
            )
            fig.update_yaxes(
                showgrid=False, zeroline=False, showticklabels=False, 
                range=[-0.5, 6.5], row=row, col=col
            )
            
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', y=0.98),
            paper_bgcolor=self.THEME_COLORS['paper'],
            plot_bgcolor=self.THEME_COLORS['background'],
            font_color=self.THEME_COLORS['text'],
            width=self.PLOT_WIDTH,
            height=self.PLOT_HEIGHT,
            margin=dict(t=100, l=50, r=50, b=50),
            hoverlabel=dict(bgcolor="black")
        )
        
        return fig

    def plot_time_of_day(self, year: int = None):
        """
        Bar chart showing reading distribution across hours of the day (0-23).
        """
        df = self.data.copy()
        
        if year:
            df = df[df['year'] == year]
        
        title = 'Time of Day Distribution'

        if df.empty:
            return None

        # Exclude Paperback
        if 'format' in df.columns:
            df = df[df['format'] != 'paperback']

        # Group by hour AND format
        # We need to ensure we have all hours for all present formats?
        # Actually px.bar handles missing categories often, but let's aggregate first.
        hourly = df.groupby(['hour', 'format'])['duration'].sum().reset_index()
        
        # Calculate Percentage (relative to Total Duration of displayed formats)
        total_duration = hourly['duration'].sum()
        hourly['percentage'] = (hourly['duration'] / total_duration * 100) if total_duration > 0 else 0
        hourly['hours'] = hourly['duration'] / 3600
        
        # Formatting for tooltip
        hourly['formatted_time'] = hourly.apply(
            lambda x: f"{int(x['duration'] // 3600)}h {int((x['duration'] % 3600) // 60)}m", 
            axis=1
        )
        
        # Create Stacked Bar Plot
        fig = px.bar(
            hourly, 
            x='hour', 
            y='percentage',
            color='format',
            title=title,
            labels={'percentage': 'Percentage (%)', 'hour': 'Hour of Day', 'format': 'Format'},
            custom_data=['formatted_time', 'percentage', 'format'],
            color_discrete_map=self.FORMAT_COLORS
        )
        
        # Styling
        fig.update_layout(
            paper_bgcolor=self.THEME_COLORS['paper'],
            plot_bgcolor=self.THEME_COLORS['background'],
            font_color=self.THEME_COLORS['text'],
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            title_x=0.5,
            title_xanchor='center',
            title_y=0.95,
            hovermode="x", # Shows all stacks for that hour
            width=self.PLOT_WIDTH,
            height=self.PLOT_HEIGHT,
            margin=dict(t=80, l=50, r=50, b=50),
            xaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=1,
                range=[-0.5, 23.5],
                title="Hour of Day",
                showgrid=False
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=self.THEME_COLORS['grid'],
                title='Percentage of Reading Time (%)',
                ticksuffix="%",
                rangemode='tozero'
            )
        )
        
        fig.update_traces(
            marker_line_width=0,
            hovertemplate="<br><b>Format:</b> %{customdata[2]}<br><b>Share</b>: %{y:.1f}%<br><b>Time</b>: %{customdata[0]}<extra></extra>",
            hoverlabel=dict(bgcolor="black")
        )
        
        return fig

    def _calculate_streaks(self, df: pd.DataFrame, min_minutes=10) -> list[int]:
        """
        Calculate list of streak lengths (consecutive days with >= min_minutes reading).
        """
        if df.empty:
            return []

        # daily sums
        daily = df.groupby('date')['duration'].sum()
        # filter days meeting threshold
        valid_days = daily[daily >= min_minutes * 60].index
        valid_days = pd.Series(sorted(valid_days))
        
        if valid_days.empty:
            return []
            
        streaks = []
        current_streak = 1
        
        for i in range(1, len(valid_days)):
            # Check if consecutive day
            if (valid_days.iloc[i] - valid_days.iloc[i-1]).days == 1:
                current_streak += 1
            else:
                streaks.append(current_streak)
                current_streak = 1
        streaks.append(current_streak)
        
        return streaks

    def plot_streaks(self, year: int = None):
        """
        Histogram of reading streaks and summary stats.
        """
        df = self.data.copy()
        if year:
            df = df[df['year'] == year]
        
        title = 'Reading Streaks'
        
        # Exclude Paperback data
        if 'format' in df.columns:
            df = df[df['format'] != 'paperback']

        streaks = self._calculate_streaks(df)
        
        if not streaks:
            return None
            
        # Stats
        longest = max(streaks)
        avg_streak = sum(streaks) / len(streaks)
        
        # Prepare for Histogram
        # We want to count how many times each streak length occurred
        streak_counts = pd.Series(streaks).value_counts().reset_index()
        streak_counts.columns = ['length', 'count']
        streak_counts = streak_counts.sort_values('length')
        
        fig = px.bar(
            streak_counts, 
            x='length', 
            y='count',
            title=title
            # Removed text='count' to hide numbers inside bins
        )
        
        # Annotation text
        stats_text = (
            f"<b>Longest Streak:</b> {longest} days<br>"
            f"<b>Average Streak:</b> {avg_streak:.1f} days<br>"
        )
        
        fig.add_annotation(
            x=0.98,
            y=0.98,
            xref="paper",
            yref="paper",
            text=stats_text,
            showarrow=False,
            font=dict(size=14, color=self.THEME_COLORS['text']),
            align="right",
            bgcolor=self.THEME_COLORS['background'],
            bordercolor=self.THEME_COLORS['subtext'],
            borderwidth=1,
            borderpad=10
        )

        fig.update_layout(
            paper_bgcolor=self.THEME_COLORS['paper'],
            plot_bgcolor=self.THEME_COLORS['background'],
            font_color=self.THEME_COLORS['text'],
            width=self.PLOT_WIDTH,
            height=self.PLOT_HEIGHT,
            margin=dict(t=80, l=50, r=50, b=50),
            title_x=0.5,
            title_xanchor='center',
            title_y=0.95,
            showlegend=False,
            xaxis=dict(
                title='Streak Length (Days)',
                dtick=1, # Show every integer
                gridcolor=self.THEME_COLORS['grid'],
                showgrid=False
            ),
            yaxis=dict(
                title='Count',
                gridcolor=self.THEME_COLORS['grid'],
                showgrid=True
            )
        )
        
        fig.update_traces(
            marker_color=self.THEME_COLORS['secondary'],
            marker_line_width=0,
            # Updated hovertemplate to match weekly_activity style
            hovertemplate="<br><b>Length</b>: %{x} days<br><b>Count</b>: %{y}<extra></extra>",
            hoverlabel=dict(bgcolor="black")
        )
        
        return fig

    def _calculate_daily_streaks_map(self, df: pd.DataFrame, min_minutes=10) -> dict:
        """
        Map each valid reading date to its current streak length.
        Returns: {pd.Timestamp: int_streak_length}
        """
        if df.empty:
            return {}

        daily = df.groupby('date')['duration'].sum()
        valid_dates = sorted(daily[daily >= min_minutes * 60].index)
        
        if not valid_dates:
            return {}
            
        streak_map = {}
        current_streak = []
        
        # We need to iterate and build streaks.
        # This logic is slightly different: we want to assign the streak length 
        # to ALL days in that streak.
        
        # Iterate through dates
        temp_streak = [valid_dates[0]]
        
        for i in range(1, len(valid_dates)):
            curr = valid_dates[i]
            prev = valid_dates[i-1]
            
            if (curr - prev).days == 1:
                # Part of same streak
                temp_streak.append(curr)
            else:
                # Streak ended. Assign lengths.
                length = len(temp_streak)
                for d in temp_streak:
                    streak_map[pd.to_datetime(d)] = length
                # Start new streak
                temp_streak = [curr]
        
        # Final streak
        length = len(temp_streak)
        for d in temp_streak:
            streak_map[pd.to_datetime(d)] = length
            
        return streak_map

    def plot_streak_calendar(self, year: int = None):
        """
        3x4 Month Grid Scatter plot showing streak lengths.
        """
        df = self.data.copy()
        
        if year:
            df = df[df['year'] == year]
        else:
            target_year = df['year'].max()
            df = df[df['year'] == target_year]
            year = target_year
            
        title = 'Streak Calendar'

        # Exclude Paperback data
        if 'format' in df.columns:
            df = df[df['format'] != 'paperback']

        if df.empty:
            return None
        
        # Pre-calc books per day for tooltip
        daily_books = df.groupby('date').agg({
            'title': lambda x: '<br>'.join(sorted(list(set(x)))[:5]) + ('...' if len(set(x)) > 5 else '')
        }).reset_index()
        daily_books['date'] = pd.to_datetime(daily_books['date'])
        daily_books['books_list'] = daily_books['title']

        # Calculate streaks
        streak_map = self._calculate_daily_streaks_map(df)
        if not streak_map:
            return None

        max_streak = max(streak_map.values())

        # Prepare subplots
        fig = make_subplots(
            rows=3, cols=4, 
            subplot_titles=[calendar.month_name[i] for i in range(1, 13)],
            vertical_spacing=0.08,
            horizontal_spacing=0.03
        )

        # Iterate through months
        for month in range(1, 13):
            row = (month - 1) // 4 + 1
            col = (month - 1) % 4 + 1
            
            _, num_days = calendar.monthrange(year, month)
            dates = pd.date_range(start=f"{year}-{month:02d}-01", end=f"{year}-{month:02d}-{num_days}")
            
            month_df = pd.DataFrame({'date': dates})
            
            # Map streak lengths
            month_df['streak'] = month_df['date'].map(streak_map).fillna(0).astype(int)
            
            # Merge book info
            month_df = month_df.merge(daily_books[['date', 'books_list']], on='date', how='left').fillna({'books_list': ''})
            
            # Coordinates
            month_df['day_of_week'] = month_df['date'].dt.dayofweek
            
            first_day_weekday = month_df.iloc[0]['date'].dayofweek
            month_df['day_idx'] = month_df['date'].dt.day - 1
            month_df['week_of_month'] = (month_df['day_idx'] + first_day_weekday) // 7
            
            month_df['hover_text'] = month_df.apply(
                lambda x: (
                    f"<b>{x['date'].strftime('%b %d')}</b><br>Streak: {x['streak']} days<br><br><b>Books:</b><br>{x['books_list']}"
                    if x['streak'] > 0 else 
                    f"<b>{x['date'].strftime('%b %d')}</b><br>No Streak"
                ),
                axis=1
            )
            
            # Show legend only on last
            show_scale = (month == 12)
            
            # Split into Active (Streak) and Inactive (No Streak)
            active_df = month_df[month_df['streak'] > 0].copy()
            inactive_df = month_df[month_df['streak'] == 0].copy()

            # 1. Inactive Trace
            fig.add_trace(
                go.Scatter(
                    x=inactive_df['day_of_week'],
                    y=5 - inactive_df['week_of_month'],
                    mode='markers',
                    marker=dict(
                        size=14,
                        color='#333333',
                        line=dict(width=1, color=self.THEME_COLORS['background'])
                    ),
                    hoverinfo='skip',
                    showlegend=False
                ),
                row=row, col=col
            )

            # 2. Active Trace
            if not active_df.empty:
                active_df['line_color'] = self.THEME_COLORS['text']
                
                fig.add_trace(
                    go.Scatter(
                        x=active_df['day_of_week'],
                        y=5 - active_df['week_of_month'], 
                        mode='markers',
                        marker=dict(
                            size=14,
                            color=active_df['streak'],
                            colorscale=[
                                [0, '#333333'], 
                                [0.01, self.THEME_COLORS['accent']], # Yellow start
                                [1, self.THEME_COLORS['secondary']]  # Pink end
                            ],
                            cmin=0,
                            cmax=max_streak,
                            showscale=show_scale,
                            colorbar=dict(
                                title=dict(
                                    text="Streak Length (days)",
                                    side="right"
                                ),
                                thickness=15,
                                len=0.7,
                                y=0.5
                            ) if show_scale else None,
                            line=dict(width=1, color=active_df['line_color'])
                        ),
                        text=active_df['hover_text'],
                        hoverinfo='text',
                        showlegend=False
                    ),
                    row=row, col=col
                )
            
            fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 6.5], row=row, col=col)
            fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 6.5], row=row, col=col)

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', y=0.98),
            paper_bgcolor=self.THEME_COLORS['paper'],
            plot_bgcolor=self.THEME_COLORS['background'],
            font_color=self.THEME_COLORS['text'],
            width=self.PLOT_WIDTH,
            height=self.PLOT_HEIGHT,
            margin=dict(t=100, l=50, r=50, b=50),
            hoverlabel=dict(bgcolor="black")
        )
        
        return fig

    # Book colors for consistent visualization
    BOOK_COLORS = [
        '#ff6b9d', '#feca57', '#48dbfb', '#0abde3', '#ee5a6f',
        '#ff9ff3', '#54a0ff', '#5f27cd', '#00d2d3', '#ff9f43',
        '#1dd1a1', '#576574', '#c44569', '#f8b500', '#6c5ce7'
    ]

    def _calculate_book_segments(self, df: pd.DataFrame, gap_days=7) -> pd.DataFrame:
        """
        Split book reading into segments if there are gaps > gap_days.
        Returns DataFrame suitable for px.timeline.
        """
        segments = []
        
        # Group by book
        for book_id, book_df in df.groupby('id_book'):
            # Get metadata from first row
            title = book_df['title'].iloc[0]
            fmt = book_df['format'].iloc[0] if 'format' in book_df.columns else 'kindle'
            
            # Get all unique reading dates
            dates = sorted(book_df['date'].unique())
            if not dates:
                continue
                
            dates = [pd.to_datetime(d) for d in dates]
            
            # Global dates for tooltip
            global_start = dates[0]
            global_end = dates[-1]
            
            # Find segments
            current_start = dates[0]
            current_end = dates[0]
            
            for i in range(1, len(dates)):
                diff = (dates[i] - dates[i-1]).days
                
                if diff > gap_days:
                    # End previous segment
                    segments.append({
                        'Title': title,
                        'Start': current_start,
                        'Finish': current_end + pd.Timedelta(days=1), # Add 1 day for visibility
                        'GlobalStart': global_start,
                        'GlobalFinish': global_end,
                        'Format': fmt,
                        'id_book': book_id
                    })
                    # Start new segment
                    current_start = dates[i]
                    current_end = dates[i]
                else:
                    # Extend segment
                    current_end = dates[i]
            
            # Add final segment
            segments.append({
                'Title': title,
                'Start': current_start,
                'Finish': current_end + pd.Timedelta(days=1),
                'GlobalStart': global_start,
                'GlobalFinish': global_end,
                'Format': fmt,
                'id_book': book_id
            })
            
        return pd.DataFrame(segments)

    def plot_book_timeline(self, year: int = None):
        """
        Gantt chart of book reading timeline with smart labeling.
        """
        df = self.data.copy()
        
        if year:
            df = df[df['year'] == year]

        title = 'Reading Timeline'

        if df.empty:
            return None

        segments_df = self._calculate_book_segments(df)
        
        if segments_df.empty:
            return None
            
        # Determine global time range to check boundaries
        min_date = segments_df['Start'].min()
        max_date = segments_df['Finish'].max()
        total_days_span = (max_date - min_date).days
        if total_days_span < 1: total_days_span = 1
        
        # Sort by Start date
        segments_df = segments_df.sort_values('Start', ascending=False)
        
        # Prepare Data Frames
        reading_df = segments_df.copy()
        
        # Create Spans Data (Background)
        # One row per book representing the full duration (Start to Finish)
        spans_df = segments_df[['Title', 'GlobalStart', 'GlobalFinish', 'Format']].drop_duplicates()
        
        # Create VisualFinish for plotting (Finish + 1 day to cover the last day)
        spans_df['VisualFinish'] = spans_df['GlobalFinish'] + pd.Timedelta(days=1)
        
        # Assign colors
        unique_books = segments_df['Title'].unique()
        color_map = {}
        for i, book_title in enumerate(unique_books):
            color_map[book_title] = self.BOOK_COLORS[i % len(self.BOOK_COLORS)]
        
        # 1. Plot Background Spans (Opacity 0.3, Interactive)
        # This layer handles the tooltip and covers the Full Duration (Start to VisualFinish)
        fig = px.timeline(
            spans_df, 
            x_start="GlobalStart", 
            x_end="VisualFinish", 
            y="Title",
            color="Title",
            color_discrete_map=color_map,
            opacity=0.3,
            title=title,
            pattern_shape="Format",
            pattern_shape_map={'ebook': '', 'paperback': '/', 'audiobook': '.'},
            custom_data=['GlobalStart', 'GlobalFinish'] # Explicit columns for tooltip
        )
        
        # 2. Plot Reading Segments (Opacity 1.0, Non-Interactive)
        # This layer shows actual reading sessions. Hover is skipped so mouse hits the background.
        fig_reading = px.timeline(
            reading_df, 
            x_start="Start", 
            x_end="Finish", 
            y="Title",
            color="Title",
            color_discrete_map=color_map,
            opacity=1.0,
            pattern_shape="Format",
            pattern_shape_map={'ebook': '', 'paperback': '/', 'audiobook': '.'}
        )
        
        # Set top layer to ignore hover, allowing fall-through to background
        fig_reading.update_traces(hoverinfo='skip')
        
        # Add reading traces to main fig
        fig.add_traces(fig_reading.data)
        
        # Calculate Annotations (Smart Text Placement per Book)
        annotations = []
        char_days_width = 2.5 
        
        for _, row in spans_df.iterrows():
            title_text = row['Title']
            start = row['GlobalStart']
            end = row['VisualFinish'] # Use VisualFinish for placement logic
            duration = (end - start).days
            
            text_len_days = len(title_text) * char_days_width
            
            # Decide position
            # 1. Inside
            if duration > (text_len_days + total_days_span * 0.02): 
                x_pos = start + (end - start) / 2
                x_anchor = 'center'
                text_color = 'white' 
                show_arrow = False
            else:
                # 2. Try Right
                space_right = (max_date - end).days
                if space_right > (text_len_days + total_days_span * 0.02):
                    x_pos = end + pd.Timedelta(days=total_days_span * 0.01) 
                    x_anchor = 'left'
                    text_color = self.THEME_COLORS['text']
                    show_arrow = False
                else:
                    # 3. Must go Left
                    x_pos = start - pd.Timedelta(days=total_days_span * 0.01)
                    x_anchor = 'right'
                    text_color = self.THEME_COLORS['text']
                    show_arrow = False
            
            annotations.append(dict(
                x=x_pos,
                y=title_text, 
                text=title_text,
                showarrow=show_arrow,
                xanchor=x_anchor,
                yanchor='middle',
                font=dict(color=text_color, size=11),
                bgcolor=None
            ))

        fig.update_layout(
            paper_bgcolor=self.THEME_COLORS['paper'],
            plot_bgcolor=self.THEME_COLORS['background'],
            font_color=self.THEME_COLORS['text'],
            width=self.PLOT_WIDTH + 200, 
            height=max(500, len(unique_books) * 35 + 100),
            margin=dict(t=80, l=50, r=50, b=50), 
            title_x=0.5,
            showlegend=False, 
            xaxis=dict(
                gridcolor=self.THEME_COLORS['grid'],
                title=None
            ),
            yaxis=dict(
                gridcolor=self.THEME_COLORS['grid'],
                title=None,
                showticklabels=False, 
                automargin=True
            ),
            annotations=annotations
        )
        
        # Order
        fig.update_yaxes(categoryorder='array', categoryarray=unique_books[::-1])
        
        # Use Year in axis format for All Time view
        axis_format = "%b %Y" if not year else "%b"
        fig.update_xaxes(tickformat=axis_format)

        # Update Tooltip
        # customdata[0] is GlobalStart, customdata[1] is GlobalFinish
        fig.update_traces(
            marker_line_width=0,
            hovertemplate="<b>%{y}</b><br>Start: %{customdata[0]|%b %d, %Y}<br>End: %{customdata[1]|%b %d, %Y}<extra></extra>",
            hoverlabel=dict(bgcolor="black")
        )
        
        # Custom Legend for Format
        # 1. Hide default legend (titles)
        for trace in fig.data:
            trace.showlegend = False
            
        # 2. Add dummy traces for "Kindle" and "Physical Book"
        fig.add_trace(go.Bar(
            x=[None], y=[None],
            name='Kindle',
            marker=dict(color=self.THEME_COLORS['text'], pattern_shape=''),
            showlegend=True
        ))
        
        fig.add_trace(go.Bar(
            x=[None], y=[None],
            name='Paperback',
            marker=dict(color=self.THEME_COLORS['text'], pattern_shape='/'),
            showlegend=True
        ))
        
        # Dynamic Width for Horizontal Scrolling (Fixed Height)
        pixels_per_day = 4 # ~1500px per year
        dynamic_width = max(self.PLOT_WIDTH, total_days_span * pixels_per_day)
        
        fig.update_layout(
            paper_bgcolor=self.THEME_COLORS['paper'],
            plot_bgcolor=self.THEME_COLORS['background'],
            font_color=self.THEME_COLORS['text'],
            width=dynamic_width,
            height=self.PLOT_HEIGHT, # Fixed height as requested
            autosize=False,
            margin=dict(t=80, l=50, r=50, b=50), # Minimal label margin, relying on on-chart labels
            title=dict(text=title, x=0.5, xanchor='center'), # Title centered on full width
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                bgcolor='rgba(0,0,0,0)'
            ),
            xaxis=dict(
                gridcolor=self.THEME_COLORS['grid'],
                title=None,
                side='top' # Put dates on top for better readability on long scroll? Or keep bottom. Bottom is standard.
            ),
            yaxis=dict(
                gridcolor=self.THEME_COLORS['grid'],
                title=None,
                showticklabels=False, 
                automargin=True
            ),
            annotations=annotations
        )

        return fig

    def plot_daily_pattern(self, year: int = None):
        """
        Bar chart of Average Reading Duration by Day of Week.
        """
        df = self.data.copy()
        if year:
            df = df[df['year'] == year]
            title = f'Daily Reading Pattern ({year})'
        else:
            title = 'Daily Reading Pattern (All Time)'

        if df.empty:
            return None
        
        # 1. Calculate Daily Totals (to account for multiple sessions per day)
        daily_totals = df.groupby(['date', 'day_of_week'])['duration'].sum().reset_index()
        
        # 2. Calculate Average per Day of Week
        weekday_stats = daily_totals.groupby('day_of_week')['duration'].mean().reset_index()
        weekday_stats['minutes'] = weekday_stats['duration'] / 60
        
        # Map 0-6 to Names
        days_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
        weekday_stats['Day'] = weekday_stats['day_of_week'].map(days_map)
        
        fig = px.bar(
            weekday_stats,
            x='Day',
            y='minutes',
            title=title,
            labels={'minutes': 'Average Minutes', 'Day': ''}
        )
        
        fig.update_layout(
            paper_bgcolor=self.THEME_COLORS['paper'],
            plot_bgcolor=self.THEME_COLORS['background'],
            font_color=self.THEME_COLORS['text'],
            width=self.PLOT_WIDTH,
            height=self.PLOT_HEIGHT,
            margin=dict(t=80, l=50, r=50, b=50),
            title_x=0.5,
            showlegend=False,
            xaxis=dict(
                gridcolor=self.THEME_COLORS['grid'],
                showgrid=False
            ),
            yaxis=dict(
                gridcolor=self.THEME_COLORS['grid'],
                title='Average Minutes / Day'
            )
        )
        
        fig.update_traces(
            marker_color=self.THEME_COLORS['primary'],
            marker_line_width=0,
            hovertemplate="<br><b>Day</b>: %{x}<br><b>Avg</b>: %{y:.1f} min<extra></extra>",
            hoverlabel=dict(bgcolor="black")
        )
        
        return fig

    def plot_monthly_pattern(self, year: int = None):
        """
        Bar chart of Reading Hours by Month.
        - Specific Year: Total Hours per Month + Book List.
        - All Time: Average Hours per Month (across years).
        """
        df = self.data.copy()
        
        if df.empty:
            return None
            
        import calendar
        month_order = [calendar.month_abbr[i] for i in range(1, 13)]
        
        if year:
            # Specific Year: Total Hours + Book List
            df = df[df['year'] == year]
            title = f'Monthly Reading Pattern ({year})'
            y_label = 'Total Reading Hours'
            
            # Group by Month -> Sum Duration & Aggregate Books
            monthly_stats = df.groupby('month').agg({
                'duration': 'sum',
                'title': lambda x: '<br>'.join(sorted(list(set(x)))[:5]) + ('...' if len(set(x)) > 5 else '')
            }).reset_index()
            
            monthly_stats['value'] = monthly_stats['duration'] / 3600
            monthly_stats['books_list'] = monthly_stats['title']
            
            tooltip_template = "<br><b>Month</b>: %{x}<br><b>Total</b>: %{y:.1f} hrs<br><br><b>Books:</b><br>%{customdata[0]}<extra></extra>"
            custom_data_cols = ['books_list']
        else:
            # All Time: Average Hours per Month
            title = 'Average Monthly Reading Pattern (All Time)'
            y_label = 'Average Hours'
            tooltip_template = "<br><b>Month</b>: %{x}<br><b>Avg</b>: %{y:.1f} hrs<extra></extra>"
            custom_data_cols = []
            
            # 1. Calculate Monthly Totals for each Year-Month pair
            monthly_year_stats = df.groupby(['year', 'month'])['duration'].sum().reset_index()
            
            # 2. Calculate Average across years for each month
            monthly_stats = monthly_year_stats.groupby('month')['duration'].mean().reset_index()
            monthly_stats['value'] = monthly_stats['duration'] / 3600

        # Map 1-12 to Names
        monthly_stats['Month'] = monthly_stats['month'].apply(lambda x: calendar.month_abbr[x])
        
        fig = px.bar(
            monthly_stats,
            x='Month',
            y='value',
            title=title,
            labels={'value': y_label, 'Month': ''},
            custom_data=custom_data_cols
        )
        
        fig.update_xaxes(categoryorder='array', categoryarray=month_order)
        
        fig.update_layout(
            paper_bgcolor=self.THEME_COLORS['paper'],
            plot_bgcolor=self.THEME_COLORS['background'],
            font_color=self.THEME_COLORS['text'],
            width=self.PLOT_WIDTH,
            height=self.PLOT_HEIGHT,
            margin=dict(t=80, l=50, r=50, b=50),
            title_x=0.5,
            showlegend=False,
            xaxis=dict(
                gridcolor=self.THEME_COLORS['grid'],
                showgrid=False
            ),
            yaxis=dict(
                gridcolor=self.THEME_COLORS['grid'],
                title=y_label
            )
        )
        
        fig.update_traces(
            marker_color=self.THEME_COLORS['accent'], 
            marker_line_width=0,
            hovertemplate=tooltip_template,
            hoverlabel=dict(bgcolor="black")
        )
        
        return fig

    def plot_session_duration(self, year: int = None):
        """
        Analysis of Reading Sessions (Avg Duration).
        Layout:
        - Subplot 1: Avg Session Duration by Day of Week.
        - Subplot 2: Avg Session Duration by Month.
        """
        df = self.data.copy()
        if year:
            df = df[df['year'] == year]
            title = f'Session Analysis ({year})'
        else:
            title = 'Session Analysis (All Time)'

        if df.empty or 'session_id' not in df.columns:
            return None
            
        # 1. Aggregate into Sessions
        # usage events -> single session row
        sessions = df.groupby('session_id').agg({
            'duration': 'sum',
            'start_datetime': 'min' # Use start of session for classification
        }).reset_index()
        
        sessions['minutes'] = sessions['duration'] / 60
        sessions['day_of_week'] = sessions['start_datetime'].dt.dayofweek
        sessions['month'] = sessions['start_datetime'].dt.month
        
        # 2. Avg by Weekday
        weekday_stats = sessions.groupby('day_of_week')['minutes'].agg(['mean', 'count']).reset_index()
        days_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
        weekday_stats['Day'] = weekday_stats['day_of_week'].map(days_map)
        
        # 3. Avg by Month
        month_stats = sessions.groupby('month')['minutes'].agg(['mean', 'count']).reset_index()
        import calendar
        month_stats['Month'] = month_stats['month'].apply(lambda x: calendar.month_abbr[x])
        
        # Prepare Subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Avg Session by Weekday", "Avg Session by Month"),
            horizontal_spacing=0.1
        )
        
        # Trace 1: Weekday
        fig.add_trace(
            go.Bar(
                x=weekday_stats['Day'],
                y=weekday_stats['mean'],
                name="Weekday",
                marker_color=self.THEME_COLORS['primary'],
                hovertemplate="<b>%{x}</b><br>Avg: %{y:.1f} min<br>Sessions: %{customdata}<extra></extra>",
                customdata=weekday_stats['count']
            ),
            row=1, col=1
        )
        
        # Trace 2: Month
        fig.add_trace(
            go.Bar(
                x=month_stats['Month'],
                y=month_stats['mean'],
                name="Month",
                marker_color=self.THEME_COLORS['accent'],
                hovertemplate="<b>%{x}</b><br>Avg: %{y:.1f} min<br>Sessions: %{customdata}<extra></extra>",
                customdata=month_stats['count']
            ),
            row=1, col=2
        )
        
        # Styling
        fig.update_layout(
            title=dict(text=title, x=0.5),
            paper_bgcolor=self.THEME_COLORS['paper'],
            plot_bgcolor=self.THEME_COLORS['background'],
            font_color=self.THEME_COLORS['text'],
            width=self.PLOT_WIDTH,
            height=self.PLOT_HEIGHT,
            margin=dict(t=80, l=50, r=50, b=50),
            showlegend=False,
            yaxis=dict(title='Avg Duration (min)', gridcolor=self.THEME_COLORS['grid']),
            yaxis2=dict(title='Avg Duration (min)', gridcolor=self.THEME_COLORS['grid'])
        )
        
        # Sort Month Axis
        fig.update_xaxes(
            categoryorder='array', 
            categoryarray=[calendar.month_abbr[i] for i in range(1, 13)],
            row=1, col=2
        )
        
    def plot_cumulative_pages(self, year: int = None):
        """
        Stacked Area chart of cumulative pages read over time, split by format.
        """
        df = self.data.copy()
        if 'format' not in df.columns:
            df['format'] = 'kindle'
            
        if year:
            df = df[df['year'] == year]
        
        title = 'Cumulative Pages Read'

        if df.empty:
            return None

        # Ensure pages_read column exists
        if 'pages_read' not in df.columns:
            df['pages_read'] = 0

        # Logic for Audiobooks: 60 seconds = 1 page
        if 'format' in df.columns and 'duration' in df.columns:
            # We use loc to handle the assignment safely
            audio_mask = df['format'] == 'audiobook'
            if audio_mask.any():
                # duration is in seconds. 60s = 1 page.
                df.loc[audio_mask, 'pages_read'] = df.loc[audio_mask, 'duration'] / 60

        # 1. Aggregate Daily Pages by Format
        daily_pages = df.groupby(['date', 'format'])['pages_read'].sum().reset_index()
        
        # 2. Pivot to ensure full date coverage for all formats (fill 0)
        if daily_pages.empty:
            return None

        # Create full date range to handle gaps
        min_date = daily_pages['date'].min()
        max_date = daily_pages['date'].max()
        all_dates = pd.date_range(start=min_date, end=max_date, freq='D').date
        
        pivot_df = daily_pages.pivot(index='date', columns='format', values='pages_read').reindex(all_dates, fill_value=0).fillna(0)
        pivot_df.index.name = 'date'
        
        # 3. Calculate Cumulative Sum
        cumulative_df = pivot_df.cumsum()
        
        # 4. Melt back for Plotly
        plot_df = cumulative_df.reset_index().melt(
            id_vars='date', 
            var_name='Format', 
            value_name='Cumulative Pages'
        )
        
        # Plot
        fig = px.area(
            plot_df,
            x='date',
            y='Cumulative Pages',
            color='Format',
            title=title,
            color_discrete_map=self.FORMAT_COLORS
        )
        
        fig.update_layout(
            paper_bgcolor=self.THEME_COLORS['paper'],
            plot_bgcolor=self.THEME_COLORS['background'],
            font_color=self.THEME_COLORS['text'],
            width=self.PLOT_WIDTH,
            height=self.PLOT_HEIGHT,
            margin=dict(t=80, l=50, r=50, b=50),
            title_x=0.5,
            title_xanchor='center',
            title_y=0.95,
            xaxis=dict(
                gridcolor=self.THEME_COLORS['grid'],
                showgrid=True
            ),
            yaxis=dict(
                gridcolor=self.THEME_COLORS['grid'],
                showgrid=True,
                title='Total Pages'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_traces(
            hoverlabel=dict(bgcolor="black"),
            hovertemplate="<b>%{x}</b><br>%{y:.0f} Pages<extra></extra>"
        )
        
        return fig

    def plot_books_completed(self, year: int = None):
        """
        Bar chart of books completed over time.
        Yearly: Aggregated by Month.
        All Time: Aggregated by Quarter.
        Completion Date = Last date a book was read.
        """
        df = self.data.copy()
        
        # 1. Determine Finish Date and Format for each book
        # Group by book and get the last reading date + format (assuming format constant per book)
        books_finished = df.groupby(['id_book', 'title']).agg({
            'start_datetime': 'max',
            'format': 'first' # Get format
        }).reset_index()
        books_finished.columns = ['id_book', 'title', 'finish_date', 'format']
        
        # 2. Filter by Year
        if year:
            books_finished = books_finished[books_finished['finish_date'].dt.year == year]
            freq = 'MS' # Month Start for cleaner alignment
        else:
            freq = 'QS' # Quarter Start
        
        title_text = 'Books Completed'

        if books_finished.empty and not year:
            if year:
                 pass
            else:
                return None
            
        # 3. Resample by Time AND Format
        # We need to set index to finish_date
        books_finished.set_index('finish_date', inplace=True)
        
        # Helper for titles
        def get_titles(series):
            return '<br>'.join([f"• {t}" for t in series])
        
        # We need to group by format as well for resampling.
        # But resample is time-based. We can group by [pd.Grouper(freq=freq), 'format']
        resampled = books_finished.groupby([pd.Grouper(freq=freq), 'format']).agg({
            'title': ['count', get_titles]
        })
        
        # Flatten columns
        resampled.columns = ['count', 'titles_list']
        resampled = resampled.reset_index()
        
        # Force Full Range if Year is selected (for all formats? logic gets complex with stacking)
        # If we reindex, we lose the format-grouping structure unless we do it per format or cross join.
        # Simpler approach: Just plot what exists. If a month has 0 books, it won't show a bar.
        # If user explicitly wants empty months, we can reindex the time column specifically.
        
        # Prepare X-Axis Column
        x_col = 'finish_date'
        if not year:
            # Custom Quarter Labels: "2025 Q1"
            resampled['quarter_label'] = resampled['finish_date'].apply(lambda d: f"{d.year} Q{d.quarter}")
            x_col = 'quarter_label'
        
        # 4. Plot (Stacked Bar)
        fig = px.bar(
            resampled,
            x=x_col,
            y='count',
            color='format',
            title=title_text,
            labels={'finish_date': 'Date', 'quarter_label': 'Quarter', 'count': 'Books Completed', 'format': 'Format'},
            custom_data=['titles_list'],
            color_discrete_map=self.FORMAT_COLORS
        )
        
        fig.update_layout(
            paper_bgcolor=self.THEME_COLORS['paper'],
            plot_bgcolor=self.THEME_COLORS['background'],
            font_color=self.THEME_COLORS['text'],
            width=self.PLOT_WIDTH,
            height=self.PLOT_HEIGHT,
            margin=dict(t=80, l=50, r=50, b=50),
            title_x=0.5,
            title_xanchor='center',
            title_y=0.95,
            xaxis=dict(
                gridcolor=self.THEME_COLORS['grid'],
                showgrid=False
            ),
            yaxis=dict(
                gridcolor=self.THEME_COLORS['grid'],
                showgrid=True,
                dtick=1 # Ensure integer ticks
            ),
            bargap=0.2,
        )
        
        # Update Bars & Format
        hover_template = "<b>%{x}</b><br>Count: %{y}<br><br>%{customdata[0]}<extra></extra>"
        if year:
            # Yearly View: Date Axis
            fig.update_xaxes(dtick="M1", tickformat="%b")
            fig.update_traces(hovertemplate="<b>%{x|%B %Y}</b><br>Count: %{y}<br><br>%{customdata[0]}<extra></extra>")
        else:
            # All Time View: Categorical Quarter Strings
            fig.update_xaxes(type='category')
            fig.update_traces(hovertemplate="<b>%{x}</b><br>Count: %{y}<br><br>%{customdata[0]}<extra></extra>")
        
        fig.update_traces(
            marker_line_width=0,
            hoverlabel=dict(bgcolor="black")
        )
            
        return fig

    def plot_reading_patterns(self, year: int = None):
        """
        Analysis of Reading Patterns.
        Subplot 1: Daily Reading Pattern (Avg minutes read on a given weekday).
        Subplot 2: Monthly Reading Pattern (Avg minutes read in a month for All Time, or Total for Single Year).
        """
        df = self.data.copy()
        
        # Filter by year if needed
        if year:
            df = df[df['year'] == year]
            title = 'Reading Habits' 
        else:
            title = 'Reading Habits'

        if df.empty:
            return None
            
        # 1. Prepare Daily Data (Total minutes per day)
        daily = df.groupby('date')['duration'].sum().reset_index()
        daily['minutes'] = daily['duration'] / 60
        daily['date'] = pd.to_datetime(daily['date'])
        daily['day_of_week'] = daily['date'].dt.dayofweek
        daily['month'] = daily['date'].dt.month
        
        # --- Subplot 1: Daily Pattern (Avg Minutes per Weekday) ---
        weekday_avg = daily.groupby('day_of_week')['minutes'].mean().reset_index()
        days_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
        weekday_avg['Day'] = weekday_avg['day_of_week'].map(days_map)
        
        # --- Subplot 2: Monthly Pattern ---
        if year:
            # Single Year: Absolute Minutes per Month
            monthly_data = daily.groupby('month')['minutes'].sum().reset_index()
            y_col = 'minutes'
            y_label_month = 'Total Minutes'
            hover_template_month = "<b>%{x}</b><br>Total: %{y:.0f} min<extra></extra>"
        else:
            # All Time: Average Minutes per Month
            # Average across years
            monthly_totals = daily.groupby([daily['date'].dt.to_period('M')])['minutes'].sum().reset_index()
            monthly_totals['month'] = monthly_totals['date'].dt.month
            monthly_data = monthly_totals.groupby('month')['minutes'].mean().reset_index()
            y_col = 'minutes'
            y_label_month = 'Avg Minutes'
            hover_template_month = "<b>%{x}</b><br>Avg: %{y:.1f} min<extra></extra>"

        import calendar
        monthly_data['Month'] = monthly_data['month'].apply(lambda x: calendar.month_abbr[x])

        # Create Subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Daily Reading Pattern", f"Monthly Reading Pattern"),
            horizontal_spacing=0.15
        )
        
        # Trace 1: Weekday
        fig.add_trace(
            go.Bar(
                x=weekday_avg['Day'],
                y=weekday_avg['minutes'],
                name="Weekday",
                marker_color=self.THEME_COLORS['primary'],
                hovertemplate="<b>%{x}</b><br>Avg: %{y:.1f} min<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Trace 2: Month
        fig.add_trace(
            go.Bar(
                x=monthly_data['Month'],
                y=monthly_data[y_col],
                name="Month",
                marker_color=self.THEME_COLORS['accent'],
                hovertemplate=hover_template_month
            ),
            row=1, col=2
        )
        
        # Styling
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', y=0.95),
            paper_bgcolor=self.THEME_COLORS['paper'],
            plot_bgcolor=self.THEME_COLORS['background'],
            font_color=self.THEME_COLORS['text'],
            width=self.PLOT_WIDTH,
            height=self.PLOT_HEIGHT,
            margin=dict(t=80, l=50, r=50, b=50),
            showlegend=False,
            yaxis=dict(title='Avg Minutes', gridcolor=self.THEME_COLORS['grid']),
            yaxis2=dict(title=y_label_month, gridcolor=self.THEME_COLORS['grid'])
        )
        
        # Sort Month Axis
        fig.update_xaxes(
            categoryorder='array', 
            categoryarray=[calendar.month_abbr[i] for i in range(1, 13)],
            row=1, col=2
        )
        
        fig.update_traces(
            marker_line_width=0,
            hoverlabel=dict(bgcolor="black")
        )
        
        return fig
