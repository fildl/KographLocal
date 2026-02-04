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
        """Run the full processing pipeline for Kindle data."""
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
            
        # Add format column
        self.merged_df['format'] = 'ebook'
        
        # Add pages_read column (1 row = 1 page read event)
        self.merged_df['pages_read'] = 1

        # Fix known Koreader bugs
        # 1. Negative durations or zero duration
        self.merged_df = self.merged_df[self.merged_df['duration'] > 0].copy()
        
        # Filter: Exclude books with < 5 minutes total reading time (noise/accidental opens)
        book_durations = self.merged_df.groupby('id_book')['duration'].sum()
        valid_books = book_durations[book_durations >= 300].index # 300 seconds = 5 mins
        
        dropped_count = self.merged_df['id_book'].nunique() - len(valid_books)
        if dropped_count > 0:
            print(f"Dropped {dropped_count} books with < 5 minutes reading time.")
            
        self.merged_df = self.merged_df[self.merged_df['id_book'].isin(valid_books)].copy()
        
        # 2. Ghost sessions (optional: filter extremely short reads if needed, keeping > 1s for now)
        
        # Sort for sessionization
        self.merged_df.sort_values(['id_book', 'start_datetime'], inplace=True)

    def get_data_with_paper_books(self, csv_path):
        """
        Return a copy of the data combined with paper books from CSV.
        Does not modify inner state.
        """
        if self.merged_df is None:
            return None
            
        combined_df = self.merged_df.copy()
        
        try:
            # Read CSV with flexible whitespace handling
            paper_df = pd.read_csv(csv_path, skipinitialspace=True)
            paper_df.columns = paper_df.columns.str.strip()
            
            synthetic_rows = []
            
            for idx, row in paper_df.iterrows():
                try:
                    start_date = pd.to_datetime(row['start_date'])
                    end_date = pd.to_datetime(row['end_date'])
                    
                    if pd.isna(start_date) or pd.isna(end_date):
                        continue
                        
                    pages = float(row['pages']) if pd.notna(row['pages']) else 0
                    if pages <= 0: continue
                    
                    # Estimate total duration: 2 minutes per page
                    total_seconds = pages * 2 * 60
                    
                    # Date range
                    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                    days_count = len(date_range)
                    
                    if days_count == 0: continue
                    
                    if days_count == 0: continue
                    
                    daily_seconds = total_seconds / days_count
                    daily_pages = pages / days_count
                    
                    # Generate a unique pseudo-ID (negative to distinct from Kindle IDs)
                    pseudo_id = -(abs(hash(row['title'])) % 1000000)
                    
                    for d in date_range:
                        # Create a "session" at 12:00 PM for each day
                        session_time = d + pd.Timedelta(hours=12)
                        
                        synthetic_rows.append({
                            'id_book': pseudo_id,
                            'duration': daily_seconds,
                            'pages_read': daily_pages,
                            'start_datetime': session_time,
                            'title': row['title'],
                            'authors': row['authors'],
                            'pages': row['pages'],
                            'language': row['language'] if 'language' in row else 'en',
                            'format': 'paperback',
                            # Add enriched columns manually or re-enrich?
                            # Re-enriching is safer usually, or just add what we strictly need for visuals
                            'date': session_time.date(),
                            'year': session_time.year,
                            'month': session_time.month,
                            'day_of_week': session_time.dayofweek,
                            'hour': session_time.hour,
                            'minute': session_time.minute
                        })
                        
                except Exception as e:
                    print(f"Error processing paper book row {idx}: {e}")
                    continue
            
            if synthetic_rows:
                synthetic_df = pd.DataFrame(synthetic_rows)
                # Concatenate with main dataframe
                combined_df = pd.concat([combined_df, synthetic_df], ignore_index=True)
                # Re-sort
                combined_df.sort_values('start_datetime', inplace=True)
                print(f"Added {len(paper_df)} paper books to combined dataset.")
                
            return combined_df
                
        except Exception as e:
            print(f"Failed to load paper books from {csv_path}: {e}")
            return combined_df

    def get_data_with_audio_books(self, csv_path, current_combined_df=None):
        """
        Parse audio books CSV and merge into the dataset.
        Format expectation: title, authors, total_duration, date, end_time, progress
        'progress' is interpreted as session duration (H:MM).
        """
        target_df = current_combined_df if current_combined_df is not None else self.merged_df.copy()
        
        try:
            audio_df = pd.read_csv(csv_path, skipinitialspace=True)
            audio_df.columns = audio_df.columns.str.strip()
            
            new_rows = []
            
            for idx, row in audio_df.iterrows():
                try:
                    # Parse Date
                    date_str = str(row['date']).strip()
                    end_time_str = str(row['end_time']).strip()
                    duration_str = str(row['progress']).strip() # "0:22" -> 22 mins
                    
                    # Parse Duration (H:MM)
                    try:
                        h, m = map(int, duration_str.split(':'))
                        duration_seconds = h * 3600 + m * 60
                    except ValueError:
                        # Fallback for M:SS if needed, assuming H:MM based on "0:22" = 22 mins
                        # If split gives 3 parts H:MM:SS handle that too
                        parts = list(map(int, duration_str.split(':')))
                        if len(parts) == 3:
                            duration_seconds = parts[0]*3600 + parts[1]*60 + parts[2]
                        elif len(parts) == 2:
                            duration_seconds = parts[0]*3600 + parts[1]*60
                        else:
                            duration_seconds = 0
                            
                    if duration_seconds <= 0: continue
                    
                    # Parse End Time to calculate Start Time
                    # Combined string: "2026-02-02 13:29"
                    end_dt = pd.to_datetime(f"{date_str} {end_time_str}")
                    start_dt = end_dt - pd.Timedelta(seconds=duration_seconds)
                    
                    # Generate Pseudo ID
                    pseudo_id = -(abs(hash(row['title'] + "audio")) % 1000000) - 2000000 # Offset from paper books
                    
                    new_rows.append({
                        'id_book': pseudo_id,
                        'duration': duration_seconds,
                        'pages_read': 0, # Audio doesn't track pages
                        'start_datetime': start_dt,
                        'title': row['title'],
                        'authors': row['authors'],
                        'pages': 0, # Could try to estimate from total_duration?
                        'language': 'en', # Default or parse if available
                        'format': 'audiobook',
                        'date': start_dt.date(),
                        'year': start_dt.year,
                        'month': start_dt.month,
                        'day_of_week': start_dt.dayofweek,
                        'hour': start_dt.hour,
                        'minute': start_dt.minute
                    })

                except Exception as e:
                    print(f"Error processing audiobook row {idx}: {e}")
                    continue
            
            if new_rows:
                new_df = pd.DataFrame(new_rows)
                target_df = pd.concat([target_df, new_df], ignore_index=True)
                target_df.sort_values('start_datetime', inplace=True)
                print(f"Added {len(new_df)} audiobook sessions.")
                
            return target_df

        except Exception as e:
            print(f"Failed to load audiobooks from {csv_path}: {e}")
            return target_df

    def _enrich_data(self):
        """Add time-based features."""
        dt = self.merged_df['start_datetime']
        self.merged_df['date'] = dt.dt.date
        self.merged_df['year'] = dt.dt.year
        self.merged_df['month'] = dt.dt.month
        self.merged_df['day_of_week'] = dt.dt.dayofweek  # 0=Monday
        self.merged_df['hour'] = dt.dt.hour
        self.merged_df['minute'] = dt.dt.minute

    def _create_sessions(self, gap_minutes=5):
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
