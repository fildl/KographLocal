import os
import plotly.io as pio

class DashboardGenerator:
    """
    Generates a single HTML dashboard file aggregating all plots.
    """
    def __init__(self, theme_colors):
        self.theme = theme_colors
        self.plots = {} # title -> html_div
        self.stats = {} # key -> value

    def add_plot(self, key, fig):
        if fig:
            # Generate HTML div for the figure
            # include_plotlyjs='cdn' ensures it's portable but loads fast if cached.
            # Using 'cdn' means one script tag at the top of the body usually handles it.
            # But full_html=False just gives the div. We need to ensure plotly.js is included once.
            self.plots[key] = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

    def add_stat(self, key, value):
        self.stats[key] = value

    def generate(self, year, output_path):
        """Builds the HTML string and writes to file."""

        # Prepare Conditional Sections
        
        # Reading Calendar Section
        reading_calendar_html = ""
        if 'reading_calendar' in self.plots:
            reading_calendar_html = f"""
        <!-- Reading Calendar -->
        <div class="section">
            <div class="section-title">Reading Calendar</div>
            <div class="plot-container">
                {self.plots.get('reading_calendar', '')}
            </div>
        </div>
            """
            
        # Streaks Section (Combined Metrics + Calendar)
        streaks_html = ""
        if 'streaks' in self.plots or 'streak_calendar' in self.plots:
            streaks_content = ""
            if 'streaks' in self.plots:
                streaks_content += f'<div class="plot-container" style="margin-bottom: 40px;">{self.plots["streaks"]}</div>'
            if 'streak_calendar' in self.plots:
                streaks_content += f'<div class="plot-container">{self.plots["streak_calendar"]}</div>'
                
            streaks_html = f"""
        <!-- Streaks -->
        <div class="section">
            <div class="section-title">Streaks</div>
            {streaks_content}
        </div>
            """

        # HTML Template
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reading Report {year}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: {self.theme['background']};
            --paper-color: {self.theme['paper']};
            --text-color: {self.theme['text']};
            --accent-color: {self.theme['accent']};
            --grid-color: {self.theme['grid']};
        }}
        
        body {{
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 40px;
        }}
        
        h1, h2, h3 {{
            color: var(--text-color);
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 60px;
        }}
        
        .header h1 {{
            font-size: 3rem;
            margin: 0;
            font-weight: 800;
            letter-spacing: -1px;
        }}
        
        .header .subtitle {{
            color: var(--accent-color);
            font-size: 1.2rem;
            margin-top: 10px;
            font-weight: 600;
        }}
        
        .container {{
            max_width: 1200px; /* Reduced max-width for single column readability */
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 60px;
        }}
        
        .section {{
            background-color: var(--paper-color);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        
        .section-title {{
            font-size: 1.5rem;
            margin-bottom: 30px;
            border-bottom: 2px solid var(--grid-color);
            padding-bottom: 10px;
            display: inline-block;
        }}
        
        .plot-container {{
            width: 100%;
            border-radius: 10px;
            overflow: hidden;
        }}
        
    </style>
</head>
<body>

    <div class="header">
        <h1>Reading Report {year}</h1>
        <div class="subtitle">Kograph Local Analytics</div>
    </div>

    <div class="container">
        
        <!-- Timeline -->
        <div class="section">
            <div class="section-title">Timeline & History</div>
            <div class="plot-container">
                {self.plots.get('timeline', '<!-- No Data -->')}
            </div>
        </div>

        <!-- Activity & Time -->
        <div class="section">
            <div class="section-title">Activity Patterns</div>
            <div class="plot-container" style="margin-bottom: 40px;">
                {self.plots.get('weekly_activity', '')}
            </div>
            <div class="plot-container">
                {self.plots.get('time_of_day', '')}
            </div>
        </div>
        
        <!-- Reading Patterns -->
        <div class="section">
            <div class="section-title">Reading Habits</div>
            <div class="plot-container" style="margin-bottom: 40px;">
                {self.plots.get('daily_pattern', '')}
            </div>
            <div class="plot-container" style="margin-bottom: 40px;">
                {self.plots.get('monthly_pattern', '')}
            </div>
            <div class="plot-container">
                {self.plots.get('session_duration', '')}
            </div>
        </div>

        {reading_calendar_html}

        {streaks_html}
        
    </div>
    
    <div style="text-align: center; margin-top: 60px; color: #666; font-size: 0.9rem;">
        Generated by Kograph Local
    </div>
    
</body>
</html>
        """
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"  ✓ Dashboard Saved: {output_path}")
