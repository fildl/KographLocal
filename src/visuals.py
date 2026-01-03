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
