import pandas as pd
import numpy as np
from datetime import timedelta

class DataProcessor:
    """
    Handles cleaning, merging, and feature engineering of reading data.
    """
    
    def __init__(self, page_stats: pd.DataFrame, books: pd.DataFrame):
        self.page_stats = page_stats
        self.books = books
        self.merged_df = None
        self.sessions_df = None

    def process(self):
        """Run the full processing pipeline."""
        self._clean_and_merge()
        self._enrich_data()
        self._create_sessions()
        return self.merged_df

    def _clean_and_merge(self):
        """Merge page stats with book info and clean basic errors."""
        # Convert timestamps
        self.page_stats['start_datetime'] = pd.to_datetime(self.page_stats['start_time'], unit='s')
        
        # Merge with book info
        # Using inner join to discard orphan stats (books deleted from device)
        self.merged_df = self.page_stats.merge(
            self.books[['id', 'title', 'authors', 'pages', 'language']], 
            left_on='id_book', 
            right_on='id', 
            how='inner'
        )
        
        # Drop redundant columns
        self.merged_df.drop(columns=['id', 'start_time', 'total_pages'], inplace=True, errors='ignore')
        
        # Clean language code (e.g., "it-IT" -> "it")
        if 'language' in self.merged_df.columns:
            self.merged_df['language'] = self.merged_df['language'].str.split('-').str[0]

        # Fix known Koreader bugs
        # 1. Negative durations or zero duration
        self.merged_df = self.merged_df[self.merged_df['duration'] > 0].copy()
        
        # 2. Ghost sessions (optional: filter extremely short reads if needed, keeping > 1s for now)
        
        # Sort for sessionization
        self.merged_df.sort_values(['id_book', 'start_datetime'], inplace=True)

    def _enrich_data(self):
        """Add time-based features."""
        dt = self.merged_df['start_datetime']
        self.merged_df['date'] = dt.dt.date
        self.merged_df['year'] = dt.dt.year
        self.merged_df['month'] = dt.dt.month
        self.merged_df['day_of_week'] = dt.dt.dayofweek  # 0=Monday
        self.merged_df['hour'] = dt.dt.hour
        self.merged_df['minute'] = dt.dt.minute

    def _create_sessions(self, gap_minutes=20):
        """
        Group individual page turns into reading sessions.
        A new session starts if the gap between page turns > gap_minutes.
        """
        # Calculate time diff between consecutive rows
        # We need to group by book first to avoid mixing sessions across books
        # But for global sessions, we might just sort by time generally?
        # Standard approach: Sort by time globally for "User Sessions", 
        # but Koreader tracks *per book*. Let's stick to global time for "Life Sessions".
        
        df = self.merged_df.sort_values('start_datetime').copy()
        
        df['prev_time'] = df['start_datetime'].shift(1)
        df['prev_book'] = df['id_book'].shift(1)
        
        # Calculate gap in minutes
        df['time_diff'] = (df['start_datetime'] - df['prev_time']).dt.total_seconds() / 60
        
        # New session if:
        # 1. Gap is larger than threshold
        # 2. Book changed (Optional, but usually implies a break or switch)
        # 3. First record
        condition = (df['time_diff'] > gap_minutes) | (df['id_book'] != df['prev_book']) | (df['time_diff'].isna())
        
        df['session_id'] = condition.cumsum()
        
        self.merged_df = df.drop(columns=['prev_time', 'prev_book', 'time_diff'])
        
        # TODO: Calculate aggregations per session if needed later (e.g. session length)

    def get_cleaned_data(self):
        return self.merged_df
