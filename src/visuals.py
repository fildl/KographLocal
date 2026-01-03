import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

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

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self._set_global_theme()

    def _set_global_theme(self):
        """Configure global Plotly defaults."""
        import plotly.io as pio
        
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
        
        # Create plot
        fig = px.bar(
            weekly, 
            x='week', 
            y='hours',
            title=title,
            labels={'hours': 'Hours Read', 'week': 'Week'},
        )
        
        # Styling
        fig.update_layout(
            paper_bgcolor=self.THEME_COLORS['paper'],
            plot_bgcolor=self.THEME_COLORS['background'],
            font_color=self.THEME_COLORS['text'],
            showlegend=False,
            title_x=0.5,  # Center title
            hovermode="x", # Simple x hover
            xaxis=dict(
                showgrid=False,
                title=None
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=self.THEME_COLORS['grid'],
                title=None
            )
        )
        
        fig.update_traces(
            marker_color=self.THEME_COLORS['primary'], # Fixed color
            marker_line_width=0,
            # Cleaner tooltip: "Week: Oct 10 | Hours: 5.2"
            hovertemplate="<br><b>Week</b>: %{x|%b %d}<br><b>Hours</b>: %{y:.1f}<extra></extra>",
            hoverlabel=dict(bgcolor="black") # Black background
        )
        
        return fig
