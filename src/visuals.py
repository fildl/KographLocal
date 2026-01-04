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
        Bar chart of total reading hours per week.
        Args:
            year (int, optional): Filter data for a specific year.
        """
        df = self.data.copy()
        
        # Filter by year if specified
        if year:
            df = df[df['year'] == year]
            title = f'Weekly Reading Activity ({year})'
        else:
            title = 'Weekly Reading Activity (All Time)'

        if df.empty:
            print(f"Warning: No data found for year {year}")
            return None

        df['week'] = df['start_datetime'].dt.to_period('W').dt.start_time
        
        weekly = df.groupby('week')['duration'].sum().reset_index()
        weekly['hours'] = weekly['duration'] / 3600
        
        weekly['formatted_time'] = weekly.apply(
            lambda x: f"{int(x['duration'] // 3600)}h {int((x['duration'] % 3600) // 60)}m", 
            axis=1
        )
        
        # Create plot
        fig = px.bar(
            weekly, 
            x='week', 
            y='hours',
            title=title,
            labels={'hours': 'Hours Read', 'week': 'Week'},
            custom_data=['formatted_time'] # Pass formatted string
        )
        
        # Styling
        fig.update_layout(
            paper_bgcolor=self.THEME_COLORS['paper'],
            plot_bgcolor=self.THEME_COLORS['background'],
            font_color=self.THEME_COLORS['text'],
            showlegend=False,
            title_x=0.5,  # Center title
            hovermode="x", # Simple x hover
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
            marker_color=self.THEME_COLORS['primary'], # Fixed color
            marker_line_width=0,
            # Use customdata[0] for formatted time
            hovertemplate="<br><b>Week</b>: %{x|%b %d}<br><b>Time</b>: %{customdata[0]}<extra></extra>",
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
            title = f'Reading Calendar ({year})'
        else:
            # If all time, default to latest year or warn?
            # User wants visual consistent with PC.
            # Let's default to max year if no year specified for this specific plot type, 
            # because 3x4 grid is inherently yearly.
            target_year = df['year'].max()
            df = df[df['year'] == target_year]
            title = f'Reading Calendar ({target_year})'
            year = target_year

        if df.empty:
            return None

        # Prepare subplots
        fig = make_subplots(
            rows=3, cols=4, 
            subplot_titles=[calendar.month_name[i] for i in range(1, 13)],
            vertical_spacing=0.08,
            horizontal_spacing=0.03
        )

        daily = df.groupby('date')['duration'].sum().reset_index()
        daily['minutes'] = daily['duration'] / 60
        daily['date'] = pd.to_datetime(daily['date'])
        
        # Max reading for color normalization
        max_reading = daily['minutes'].max() if not daily.empty else 1

        # Iterate through months
        for month in range(1, 13):
            row = (month - 1) // 4 + 1
            col = (month - 1) % 4 + 1
            
            # Generate full dates for this month
            _, num_days = calendar.monthrange(year, month)
            dates = pd.date_range(start=f"{year}-{month:02d}-01", end=f"{year}-{month:02d}-{num_days}")

            month_df = pd.DataFrame({'date': dates})
            month_df = month_df.merge(daily, on='date', how='left').fillna(0)
            
            # Coordinates
            month_df['day_of_week'] = month_df['date'].dt.dayofweek # 0=Mon
            
            # Week of month (0-based index)
            first_day_weekday = month_df.iloc[0]['date'].dayofweek
            month_df['day_idx'] = month_df['date'].dt.day - 1
            month_df['week_of_month'] = (month_df['day_idx'] + first_day_weekday) // 7
            
            # Formatting
            month_df['formatted_time'] = month_df.apply(
                lambda x: f"{int(x['minutes'] // 60)}h {int(x['minutes'] % 60)}m", 
                axis=1
            )
            
            month_df['hover_text'] = month_df.apply(
                lambda x: f"<b>{x['date'].strftime('%b %d')}</b><br>{x['formatted_time']}", 
                axis=1
            )
            
            # Show legend (colorbar) only on the last chart (Dec)
            show_scale = (month == 12)
            
            # Outline logic: visible outline for non-zero data
            month_df['line_color'] = month_df['minutes'].apply(
                lambda x: self.THEME_COLORS['text'] if x > 0 else self.THEME_COLORS['background']
            )
            
            fig.add_trace(
                go.Scatter(
                    x=month_df['day_of_week'],
                    y=5 - month_df['week_of_month'], 
                    mode='markers',
                    marker=dict(
                        size=14,
                        color=month_df['minutes'],
                        colorscale=[
                            [0, '#333333'], # Empty
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
                        line=dict(width=1, color=month_df['line_color'])
                    ),
                    text=month_df['hover_text'],
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
            title=dict(text=title, x=0.5),
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
            title = f'Time of Day Distribution ({year})'
        else:
            title = 'Time of Day Distribution (All Time)'

        if df.empty:
            return None

        # Group by hour
        hourly = df.groupby('hour')['duration'].sum().reindex(range(24), fill_value=0).reset_index()
        
        # Calculate Percentage
        total_duration = df['duration'].sum()
        hourly['percentage'] = (hourly['duration'] / total_duration * 100) if total_duration > 0 else 0
        hourly['hours'] = hourly['duration'] / 3600 # Keep for tooltip
        
        # Formatting for tooltip
        hourly['formatted_time'] = hourly.apply(
            lambda x: f"{int(x['duration'] // 3600)}h {int((x['duration'] % 3600) // 60)}m", 
            axis=1
        )
        
        # Create plot
        fig = px.bar(
            hourly, 
            x='hour', 
            y='percentage',
            title=title,
            custom_data=['formatted_time', 'percentage']
        )
        
        # Styling
        fig.update_layout(
            paper_bgcolor=self.THEME_COLORS['paper'],
            plot_bgcolor=self.THEME_COLORS['background'],
            font_color=self.THEME_COLORS['text'],
            showlegend=False,
            title_x=0.5,
            hovermode="x",
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
                ticksuffix="%"
            )
        )
        
        fig.update_traces(
            marker_color=self.THEME_COLORS['primary'],
            marker_line_width=0,
            hovertemplate="<br><b>%{x}:00</b><br><b>Share</b>: %{y:.1f}%<br><b>Time</b>: %{customdata[0]}<extra></extra>",
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
            title = f'Reading Streaks ({year})'
        else:
            title = 'Reading Streaks (All Time)'

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
            title = f'Streak Calendar ({year})'
        else:
            target_year = df['year'].max()
            df = df[df['year'] == target_year]
            title = f'Streak Calendar ({target_year})'
            year = target_year

        if df.empty:
            return None

        # Calculate streaks for this year's data
        # Note: If a streak started in prev year, this naive filter cuts it.
        # Ideally we'd calc streaks on full data then filter map, but for now specific year calc is acceptable.
        streak_map = self._calculate_daily_streaks_map(df)
        if not streak_map:
            return None

        max_streak = max(streak_map.values())

        # Prepare subplots (same as reading calendar)
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
            
            # Coordinates
            month_df['day_of_week'] = month_df['date'].dt.dayofweek
            
            first_day_weekday = month_df.iloc[0]['date'].dayofweek
            month_df['day_idx'] = month_df['date'].dt.day - 1
            month_df['week_of_month'] = (month_df['day_idx'] + first_day_weekday) // 7
            
            month_df['hover_text'] = month_df.apply(
                lambda x: f"<b>{x['date'].strftime('%b %d')}</b><br>Streak: {x['streak']} days" if x['streak'] > 0 else f"<b>{x['date'].strftime('%b %d')}</b><br>No Streak", 
                axis=1
            )
            
            # Show legend only on last
            show_scale = (month == 12)
            
            # Line color
            month_df['line_color'] = month_df['streak'].apply(
                lambda x: self.THEME_COLORS['text'] if x > 0 else self.THEME_COLORS['background']
            )

            fig.add_trace(
                go.Scatter(
                    x=month_df['day_of_week'],
                    y=5 - month_df['week_of_month'], 
                    mode='markers',
                    marker=dict(
                        size=14,
                        color=month_df['streak'],
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
                        line=dict(width=1, color=month_df['line_color'])
                    ),
                    text=month_df['hover_text'],
                    hoverinfo='text',
                    showlegend=False
                ),
                row=row, col=col
            )
            
            fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 6.5], row=row, col=col)
            fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 6.5], row=row, col=col)

        fig.update_layout(
            title=dict(text=title, x=0.5),
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
            title = f'Book Timeline ({year})'
        else:
            title = 'Book Timeline (All Time)'

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
        
        # Assign colors
        unique_books = segments_df['Title'].unique()
        color_map = {}
        for i, book_title in enumerate(unique_books):
            color_map[book_title] = self.BOOK_COLORS[i % len(self.BOOK_COLORS)]
        
        fig = px.timeline(
            segments_df, 
            x_start="Start", 
            x_end="Finish", 
            y="Title",
            color="Title", # Distinct colors per book
            color_discrete_map=color_map,
            title=title,
            pattern_shape="Format", # Distinguish Kindle vs Paperback
            pattern_shape_map={
                'kindle': '',      # Solid
                'paperback': '/'   # Hatched
            }
        )
        
        # Calculate Annotations (Smart Text Placement per Book)
        annotations = []
        
        # Heuristic for char width in "days" units. 
        # With pixels_per_day = 4, and approx 10px per char:
        char_days_width = 2.5 
        
        # Group by Title to calc envelope for labeling
        # Note: distinct books with same title? Unlikely for this user stats, but strictly should use id_book if possible.
        # However, Y-axis is Title, so we group by Title to match the row.
        for title_text, group in segments_df.groupby('Title'):
            start = group['Start'].min()
            end = group['Finish'].max()
            duration = (end - start).days
            
            text_len_days = len(title_text) * char_days_width
            
            # Decide position
            # 1. Inside
            if duration > (text_len_days + total_days_span * 0.02): # Fits with padding?
                x_pos = start + (end - start) / 2
                x_anchor = 'center'
                text_color = 'white' # Assume dark bars (or floating in dark void)
                # If floating in void (gap), white is fine on dark bg. 
                # If on bar, white is fine on color.
                show_arrow = False
            else:
                # 2. Try Right
                space_right = (max_date - end).days
                if space_right > (text_len_days + total_days_span * 0.02):
                    x_pos = end + pd.Timedelta(days=total_days_span * 0.01) # Small padding
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
                y=title_text, # Y is the category name
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
            margin=dict(t=80, l=50, r=50, b=50), # Reduced left margin since labels are on plot
            title_x=0.5,
            showlegend=False, # Hide legend as colors correspond to bars which have labels
            xaxis=dict(
                gridcolor=self.THEME_COLORS['grid'],
                title=None
            ),
            yaxis=dict(
                gridcolor=self.THEME_COLORS['grid'],
                title=None,
                showticklabels=False, # Hide axis labels
                automargin=True
            ),
            annotations=annotations
        )
        
        # Order
        fig.update_yaxes(categoryorder='array', categoryarray=unique_books[::-1])

        fig.update_traces(
            marker_line_width=0,
            hovertemplate="<b>%{y}</b><br>Start: %{base|%b %d}<br>End: %{x|%b %d}<extra></extra>"
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
