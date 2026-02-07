import pandas as pd
import numpy as np
from datetime import timedelta
import difflib
import re
try:
    from numbers_parser import Document
except ImportError:
    Document = None

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
        
        # Apply manual time corrections
        self._apply_time_corrections()

    def _apply_time_corrections(self, csv_path='data/time_corrections.csv'):
        """
        Apply manual time offsets from a CSV file to correct Kindle clock errors.
        Format: start_datetime, end_datetime, offset_minutes
        """
        try:
            import os
            if not os.path.exists(csv_path):
                return

            corrections_df = pd.read_csv(csv_path, comment='#', skipinitialspace=True)
            corrections_df.columns = corrections_df.columns.str.strip()
            
            if corrections_df.empty:
                return

            # Parse datetimes (allow NaT for end_datetime)
            corrections_df['start_datetime'] = pd.to_datetime(corrections_df['start_datetime'])
            corrections_df['end_datetime'] = pd.to_datetime(corrections_df['end_datetime'])
            
            # Apply each correction rule
            count = 0
            for _, row in corrections_df.iterrows():
                try:
                    start_window = row['start_datetime']
                    end_window = row['end_datetime']
                    offset_min = float(row['offset_minutes'])
                    
                    if offset_min == 0:
                        continue
                        
                    # Filter rows in window
                    # If end_window is NaT (empty in CSV), treat as open-ended (until forever)
                    if pd.isna(end_window):
                        mask = (self.merged_df['start_datetime'] >= start_window)
                    else:
                        mask = (self.merged_df['start_datetime'] >= start_window) & \
                               (self.merged_df['start_datetime'] <= end_window)
                    
                    # Ensure we only correct 'ebook' format (though at this stage usually only ebooks exist)
                    if 'format' in self.merged_df.columns:
                        mask = mask & (self.merged_df['format'] == 'ebook')

                    if mask.any():
                        self.merged_df.loc[mask, 'start_datetime'] += pd.Timedelta(minutes=offset_min)
                        count += mask.sum()
                except Exception as e:
                    print(f"Error applying correction row: {e}")
                    
            if count > 0:
                print(f"Applied time corrections to {count} reading sessions.")
                
        except Exception as e:
            print(f"Failed to load time corrections from {csv_path}: {e}")

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
                    
                    # Estimate total duration: 1 minute per page (60 seconds)
                    total_seconds = pages * 60
                    
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
                
        except Exception as e:
            print(f"Failed to load paper books from {csv_path}: {e}")
            return combined_df
    
    def _find_best_match(self, target_title, candidates):
        """
        Find best match for target_title in candidates list using multiple strategies.
        Returns the matching candidate string or None.
        """
        t_title = str(target_title).strip()
        t_lower = t_title.lower()
        
        # 1. Exact Match
        for c in candidates:
            if str(c).lower().strip() == t_lower:
                return c
                
                
                
        # 2. Fuzzy Match (difflib)
        matches = difflib.get_close_matches(t_title, [str(c) for c in candidates], n=1, cutoff=0.9)
        if matches:
            return matches[0]
            
        # 3. Token Set Match
        def get_tokens(text):
             return set(re.sub(r'[^a-z0-9\s]', '', str(text).lower()).split())
             
        t_tokens = get_tokens(t_title)
        if not t_tokens: return None
        
        best_match = None
        best_score = 0
        
        for c in candidates:
            c_tokens = get_tokens(c)
            if not c_tokens: continue
            
            common = t_tokens.intersection(c_tokens)
            if not common: continue
            
            jaccard = len(common) / len(t_tokens.union(c_tokens))
            
            if jaccard > 0.8 and jaccard > best_score:
                best_score = jaccard
                best_match = c
                
        return best_match

    def get_data_with_metadata(self, metadata_path, current_combined_df=None):
        """
        Enrich dataset with metadata from a Numbers file (e.g., Country, Purchase Date).
        """
        target_df = current_combined_df if current_combined_df is not None else self.merged_df.copy()
        
        if not Document:
            print("numbers-parser not installed. Skipping metadata enrichment.")
            return target_df, None

        try:
            doc = Document(metadata_path)
            sheets = doc.sheets
            if not sheets:
                return target_df
            
            # Assume first table of first sheet or look for a specific one
            # Based on inspection, data is in the first table
            table = sheets[0].tables[0]
            data = table.rows(values_only=True)
            
            if not data:
                return target_df
                
            # Convert to DataFrame
            # First row is header
            meta_df = pd.DataFrame(data[1:], columns=data[0])
            
            # Normalization mapping based on inspection
            # Found columns: title, author, nationality, purchase
            # Map to: 'title', 'author', 'country', 'purchase_date'
            
            # Clean column names (strip whitespace and lower case just in case)
            meta_df.columns = meta_df.columns.str.strip().str.lower()
            
            rename_map = {
                'title': 'title_match',
                'nationality': 'author_country',
                'purchase': 'purchase_date'
            }
            
            # Filter only relevant columns if they exist
            cols_to_keep = [c for c in rename_map.keys() if c in meta_df.columns]
            if 'title' not in cols_to_keep:
                print("Metadata file missing 'title' column.")
                return target_df, None
                
            meta_df = meta_df[cols_to_keep].copy()
            meta_df.rename(columns=rename_map, inplace=True)
            
            # Prepare for Fuzzy Matching
            # We want to map Target Title -> Meta Title
            target_titles = target_df['title'].unique()
            meta_titles = meta_df['title_match'].dropna().unique()
            
            title_map = {}
            
            # Helper for token matching
            title_map = {}
            for t_title in target_titles:
                match = self._find_best_match(t_title, meta_titles)
                if match:
                    title_map[t_title] = match
                    
            # Map the 'title_match' column in target_df
            target_df['title_match'] = target_df['title'].map(title_map)
            
            # Drop duplicates in metadata to avoid explosion
            meta_df.drop_duplicates(subset=['title_match'], inplace=True)
            
            # Merge on the matched title
            target_df = target_df.merge(
                meta_df[['title_match', 'author_country', 'purchase_date']],
                on='title_match',
                how='left'
            )
            
            target_df.drop(columns=['title_match'], inplace=True)
            
            # Convert purchase_date to datetime
            if 'purchase_date' in target_df.columns:
                target_df['purchase_date'] = pd.to_datetime(target_df['purchase_date'], errors='coerce')
                
            print(f"Enriched {target_df['author_country'].notna().sum()} rows with metadata.")
            
            # Also convert meta_df purchase_date for external usage
            if 'purchase_date' in meta_df.columns:
                meta_df['purchase_date'] = pd.to_datetime(meta_df['purchase_date'], errors='coerce')
            
            return target_df, meta_df

        except Exception as e:
            print(f"Failed to load metadata from {metadata_path}: {e}")
            return target_df, None

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
            
            # Helper to parse duration string "H:MM" or "M:SS" to seconds
            def parse_duration(dur_str):
                try:
                    dur_str = str(dur_str).strip()
                    parts = list(map(int, dur_str.split(':')))
                    if len(parts) == 3: # H:MM:SS
                        return parts[0]*3600 + parts[1]*60 + parts[2]
                    elif len(parts) == 2: # H:MM usually, but could be M:SS? 
                        # User example "0:22" for 22 mins implies H:MM format.
                        return parts[0]*3600 + parts[1]*60
                    return 0
                except:
                    return 0

            # Pre-process numeric data
            audio_df['progress_seconds'] = audio_df['progress'].apply(parse_duration)
            
            # Construct full datetime for sorting
            # Combine date and end_time
            audio_df['end_dt'] = pd.to_datetime(audio_df['date'].astype(str) + ' ' + audio_df['end_time'].astype(str))
            
            # Sort by title and time to ensure correct diffing
            audio_df.sort_values(by=['title', 'end_dt'], inplace=True)
            
            # Calculate session duration (diff of cumulative progress)
            # Group by title to track progress per book
            audio_df['prev_progress'] = audio_df.groupby('title')['progress_seconds'].shift(1).fillna(0)
            
            # Calculate duration
            audio_df['duration_seconds'] = audio_df['progress_seconds'] - audio_df['prev_progress']
            
            # Handle cases where progress might reset (negative duration -> treat as new start/session from 0)
            # If duration < 0, it means we jumped back or restarted. 
            # In that case, the session duration is just the current progress (read from 0 to X).
            audio_df.loc[audio_df['duration_seconds'] < 0, 'duration_seconds'] = audio_df['progress_seconds']
            
            # Calculate start_dt
            audio_df['start_dt'] = audio_df['end_dt'] - pd.to_timedelta(audio_df['duration_seconds'], unit='s')
            
            new_rows = []
            
            for idx, row in audio_df.iterrows():
                try:
                    duration_seconds = row['duration_seconds']
                    if duration_seconds <= 0: continue
                    
                    # Generate Pseudo ID
                    pseudo_id = -(abs(hash(row['title'] + "audio")) % 1000000) - 2000000 
                    
                    new_rows.append({
                        'id_book': pseudo_id,
                        'duration': duration_seconds,
                        'pages_read': 0, 
                        'start_datetime': row['start_dt'],
                        'title': row['title'],
                        'authors': row['authors'],
                        'pages': 0, 
                        'language': 'en',
                        'format': 'audiobook',
                        'date': row['start_dt'].date(),
                        'year': row['start_dt'].year,
                        'month': row['start_dt'].month,
                        'day_of_week': row['start_dt'].dayofweek,
                        'hour': row['start_dt'].hour,
                        'minute': row['start_dt'].minute
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

    def get_paper_books_from_numbers(self, metadata_path, current_combined_df=None):
        """
        Extract paper books from Numbers file and merge meaningful sessions.
        Replaces get_data_with_paper_books.
        """
        combined_df = current_combined_df if current_combined_df is not None else self.merged_df.copy()
        
        if not Document:
            print("numbers-parser not installed.")
            return combined_df
            
        try:
            doc = Document(metadata_path)
            sheets = doc.sheets
            if not sheets: return combined_df
            
            table = sheets[0].tables[0]
            data = table.rows(values_only=True)
            if not data: return combined_df
            
            # Create DF from Numbers Data
            raw_df = pd.DataFrame(data[1:], columns=data[0])
            raw_df.columns = raw_df.columns.str.strip().str.lower()
            
            # Filter for Paper Books: 'hardcover' or 'paperback'
            if 'format' not in raw_df.columns:
                print("Numbers file missing 'format' column.")
                return combined_df
                
            # Filter for books to import:
            # 1. Paper/Hardcover (Always import)
            # 2. eBook (Import ONLY if not present in current_combined_df to avoid duplicates with Kindle sync)
            
            format_col = raw_df['format'].str.lower()
            manual_formats = ['hardcover', 'paperback', 'ebook']
            
            # Filter rows with relevant formats
            relevant_df = raw_df[format_col.isin(manual_formats)].copy()
            
            if relevant_df.empty:
                return combined_df

            # Pre-calculate existing titles for deduplication
            existing_titles = []
            if not combined_df.empty and 'title' in combined_df.columns:
                existing_titles = combined_df['title'].dropna().unique().tolist()
                
            # Filter for relevant formats
            target_formats = ['hardcover', 'paperback', 'ebook']
            
            # Normalize format column in raw_df for filtering
            # We use a temporary column for filtering but keep original logic if needed
            # logic: row['format']
            
            # Map columns: title, author, pages, start, finish
            rename_map = {
                'title': 'title',
                'author': 'authors',
                'pages': 'pages',
                'start': 'start_date',
                'finish': 'end_date',
                'language': 'language'
                # 'format' is kept as is (but we need to handle it manually)
            }
            
            synthetic_rows = []
            
            for idx, row in raw_df.iterrows():
                try:
                    # 1. Check Format
                    fmt_raw = str(row.get('format', '')).lower().strip()
                    if fmt_raw not in target_formats:
                        continue
                        
                    # 2. Check Title & Dates
                    title = row.get('title')
                    if pd.isna(title): continue
                    
                    start_date = pd.to_datetime(row.get('start'), errors='coerce')
                    end_date = pd.to_datetime(row.get('finish'), errors='coerce')
                    
                    if pd.isna(start_date) or pd.isna(end_date):
                        continue
                    if end_date < start_date: continue

                    # 3. Deduplication Logic (Universal)
                    # Check if title already exists in Kindle/Audio data
                    # If so, we assume the DB version is the one we want to keep (e.g. read on Kindle)
                    match = self._find_best_match(title, existing_titles)
                    if match:
                        continue
                            
                    # 4. Prepare Data
                    pages = float(row.get('pages', 0)) if pd.notna(row.get('pages')) else 0
                    if pages <= 0: pages = 200
                    
                    # 5. Determine App Format
                    # App uses lowercase 'paperback', 'ebook' usually?
                    # Visuals.py expects: 'ebook', 'paperback', 'audiobook'.
                    # So we map Hardcover/Paperback -> paperback, eBook -> ebook.
                    
                    if fmt_raw in ['hardcover', 'paperback']:
                        app_format = 'paperback'
                    else:
                        app_format = 'ebook'

                    # 6. Create Synthetic Sessions
                    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                    days_count = len(date_range)
                    if days_count == 0: days_count = 1
                    
                    daily_pages = pages / days_count
                    daily_seconds = daily_pages * 60
                    
                    pseudo_id = -(abs(hash(title)) % 1000000)
                    
                    for d in date_range:
                        session_time = d + pd.Timedelta(hours=12)
                        session_end = session_time + pd.Timedelta(seconds=daily_seconds)

                        synthetic_rows.append({
                            'id_book': pseudo_id,
                            'duration': daily_seconds,
                            'pages_read': daily_pages,
                            'start_datetime': session_time,
                            'end_datetime': session_end,
                            'title': title,
                            'authors': row.get('author', 'Unknown'),
                            'pages': pages,
                            'language': row.get('language', 'en'),
                            'format': app_format, 
                            'date': session_time.date(),
                            'year': session_time.year,
                            'month': session_time.month,
                            'day_of_week': session_time.dayofweek,
                            'hour': session_time.hour,
                            'minute': session_time.minute
                        })
                        
                except Exception as e:
                    print(f"Error processing numbers row {idx}: {e}")
                    continue
                    
            if synthetic_rows:
                new_df = pd.DataFrame(synthetic_rows)
                combined_df = pd.concat([combined_df, new_df], ignore_index=True)
                combined_df.sort_values('start_datetime', inplace=True)
                print(f"Imported {len(synthetic_rows)} synthetic sessions from Numbers (Manual Import).")
                
            return combined_df
        except Exception as e:
            print(f"Failed to load manual books from Numbers: {e}")
            return combined_df
