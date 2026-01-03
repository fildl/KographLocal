import pandas as pd
from datetime import datetime, timedelta
import os
import sqlite3
import numpy as np
import shutil
from pathlib import Path
from plots_kindle import KindlePlots
from convert_numbers_file import load_reading_catalog
from plots_numbers import NumbersPlots

class KindleReadingAnalyzer:
    def __init__(self, sqlite_path='statistics.sqlite3'):
        self.sqlite_path = sqlite_path
        self.reading_data = None
        self.conversion_data = None
        # Create output directory if it doesn't exist
        os.makedirs('output', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
        # Check for new database and copy if available
        self._update_database()
    
    def _update_database(self):
        """Check for new statistics.sqlite3 and copy it if available, otherwise use existing one"""
        # Source directory (Kindle device)
        source_dir = Path("/Volumes/Kindle/koreader/settings")
        source_file = source_dir / "statistics.sqlite3"
        
        # Destination directory
        dest_dir = Path("data")
        dest_file = dest_dir / "statistics.sqlite3"
        
        try:
            # Check if new database exists on Kindle
            if source_file.exists():
                print(f"Found new database on Kindle: {source_file}")
                
                # Check if we need to update (compare modification times)
                if not dest_file.exists() or source_file.stat().st_mtime > dest_file.stat().st_mtime:
                    shutil.copy2(source_file, dest_file)
                    print(f"✓ Updated database from Kindle to: {dest_file}")
                else:
                    print("✓ Local database is already up to date")
            else:
                print("Kindle not connected or database not found on device")
                
                # Check if we have a local copy
                if dest_file.exists():
                    print(f"✓ Using existing local database: {dest_file}")
                else:
                    raise FileNotFoundError("No database found locally or on Kindle device")
            
            # Update sqlite_path to use the data directory
            self.sqlite_path = str(dest_file)
            print(f"Database ready: {self.sqlite_path}")
            
        except Exception as e:
            print(f"Error updating database: {e}")
            # Try to use existing local database as fallback
            if dest_file.exists():
                print(f"Falling back to existing local database: {dest_file}")
                self.sqlite_path = str(dest_file)
            else:
                raise FileNotFoundError("No database available for analysis")
    
    def load_and_process_data(self):
        """Extract data from SQLite and process to create merged reading data"""
        # Extract data from SQLite database
        page_stats, books = self._extract_from_sqlite()
        # Convert Unix timestamp to readable datetime
        page_stats['start_datetime'] = pd.to_datetime(page_stats['start_time'], unit='s')
        # Merge the datasets to get book titles with reading data
        reading_data = page_stats.merge(books[['id', 'title', 'authors', 'pages', 'language']],
                                      left_on='id_book', right_on='id', how='left')
        # Sort the dataframe by id_book and page
        reading_data = reading_data.sort_values(['id_book', 'page'])
        # Remove unnecessary columns
        reading_data = reading_data.drop(columns=['id', 'start_time', 'total_pages'])
        # Normalize language format (convert "it-IT" to "it")
        reading_data['language'] = reading_data['language'].str.split('-').str[0]
        
        # Group by book and page, merge duplicate entries
        kindle_data = reading_data.groupby(['id_book', 'page', 'title', 'authors', 'pages', 'language']).agg({
            'duration': 'sum',
            'start_datetime': 'min'
        }).reset_index()
        
        # Add format column for Kindle books
        kindle_data['format'] = 'kindle'
        
        # Load and process paper books
        paper_data = self._load_and_process_paper_books()
        
        # Combine both datasets
        self.reading_data = pd.concat([kindle_data, paper_data], ignore_index=True)
        self.reading_data = self.reading_data.sort_values(['format', 'id_book', 'page'])
        
        return self.reading_data
    
    def _load_conversion_data(self):
        """Load page conversion data"""
        try:
            conversion_df = pd.read_csv('data/paper_to_kindle_conversion.csv')
            # Clean column names (remove extra spaces)
            conversion_df.columns = conversion_df.columns.str.strip()
            
            # Check actual column names and rename if needed
            #print(f"Conversion file columns: {list(conversion_df.columns)}")
            
            # Rename columns to match expected names
            column_mapping = {}
            for col in conversion_df.columns:
                clean_col = col.strip().lower()
                if 'kindle' in clean_col:
                    column_mapping[col] = 'kindle'
                elif 'paperback' in clean_col:
                    column_mapping[col] = 'paperback'
                elif 'title' in clean_col:
                    column_mapping[col] = 'title'
            
            conversion_df = conversion_df.rename(columns=column_mapping)
            
            # Clean the data - replace empty strings/spaces with NaN
            conversion_df['kindle'] = conversion_df['kindle'].replace(r'^\s*$', np.nan, regex=True)
            conversion_df['paperback'] = conversion_df['paperback'].replace(r'^\s*$', np.nan, regex=True)
            
            # Convert to numeric
            conversion_df['kindle'] = pd.to_numeric(conversion_df['kindle'], errors='coerce')
            conversion_df['paperback'] = pd.to_numeric(conversion_df['paperback'], errors='coerce')
            
            # Clean and prepare conversion data - only keep rows with both values
            conversion_df = conversion_df.dropna(subset=['kindle', 'paperback'])
            conversion_df['title'] = conversion_df['title'].str.strip()
            self.conversion_data = conversion_df
            #print(f"Loaded {len(conversion_df)} books with conversion data")
            return conversion_df
        except FileNotFoundError:
            print("Warning: paper_to_kindle_conversion.csv not found. Using original page counts.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading conversion data: {e}")
            return pd.DataFrame()
    
    def _calculate_conversion_ratio(self):
        """Calculate average conversion ratio from available data"""
        if self.conversion_data is not None and not self.conversion_data.empty:
            ratios = self.conversion_data['kindle'] / self.conversion_data['paperback']
            avg_ratio = ratios.mean()
            #print(f"Average Kindle/Paperback page ratio: {avg_ratio:.3f}")
            return avg_ratio
        return 1.0  # Default to 1:1 if no conversion data
    
    def _load_and_process_paper_books(self):
        """Load paper books from CSV file and create synthetic reading data"""
        try:
            # Load paper books data from CSV file
            paper_books = pd.read_csv('data/paper_books.csv')
            
            # Clean column names (remove any hidden characters/whitespace)
            paper_books.columns = paper_books.columns.str.strip()
            
            # Check if we have any data
            if paper_books.empty:
                print("No paper books data found in paper_books.csv")
                return pd.DataFrame()
            
            # Ensure required columns exist
            required_columns = ['title', 'authors', 'pages', 'start_date', 'end_date']
            missing_columns = [col for col in required_columns if col not in paper_books.columns]
            if missing_columns:
                print(f"Missing required columns in paper_books.csv: {missing_columns}")
                print(f"Available columns: {list(paper_books.columns)}")
                return pd.DataFrame()
            
            # Convert date columns
            paper_books['start_date'] = pd.to_datetime(paper_books['start_date'])
            paper_books['end_date'] = pd.to_datetime(paper_books['end_date'])
            
            # Ensure language column exists
            if 'language' not in paper_books.columns:
                paper_books['language'] = 'unknown'
            
            # Load conversion data
            self._load_conversion_data()
            avg_ratio = self._calculate_conversion_ratio()
            
            all_paper_data = []
            
            # Get max id_book from Kindle data by accessing SQLite directly
            max_kindle_id = self._get_max_kindle_id()
            
            for idx, book in paper_books.iterrows():
                book_id = max_kindle_id + idx + 1
                
                # Convert pages if conversion data available
                kindle_equivalent_pages = self._get_kindle_equivalent_pages(book['title'], book['pages'], avg_ratio)
                
                # Generate synthetic reading data
                synthetic_data = self._generate_synthetic_reading_data(
                    book_id=book_id,
                    title=book['title'],
                    authors=book['authors'],
                    total_pages=int(kindle_equivalent_pages),
                    start_date=book['start_date'],
                    end_date=book['end_date'],
                    language=book['language']
                )
                
                all_paper_data.extend(synthetic_data)
            
            # Convert to DataFrame
            paper_df = pd.DataFrame(all_paper_data)
            paper_df['format'] = 'paperback'
            
            print(f"Generated synthetic data for {len(paper_books)} paper books")
            print(f"Paper book IDs start from: {max_kindle_id + 1}")
            print()
            return paper_df
            
        except FileNotFoundError:
            print("Warning: paper_books.csv not found. Skipping paper books integration.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading paper books from CSV file: {e}")
            return pd.DataFrame()

    def _get_max_kindle_id(self):
        """Get maximum id_book from Kindle database"""
        try:
            conn = sqlite3.connect(self.sqlite_path)
            result = conn.execute("SELECT MAX(id) FROM book").fetchone()
            conn.close()
            max_id = result[0] if result[0] is not None else 0
            #print(f"Max Kindle book ID: {max_id}")
            return max_id
        except Exception as e:
            print(f"Error getting max Kindle ID: {e}")
            return 1000  # Fallback to a high number
    
    def _get_kindle_equivalent_pages(self, title, original_pages, avg_ratio):
        """Get Kindle equivalent pages for a book"""
        if self.conversion_data is not None and not self.conversion_data.empty:
            # Look for exact match
            match = self.conversion_data[self.conversion_data['title'].str.contains(title.split(',')[0], case=False, na=False)]
            if not match.empty:
                return match.iloc[0]['kindle']
        
        # Use average ratio if no specific conversion found
        return int(original_pages * avg_ratio)
    
    def _generate_synthetic_reading_data(self, book_id, title, authors, total_pages, start_date, end_date, language):
        """Generate synthetic page-by-page reading data for a paper book"""
        total_days = (end_date - start_date).days + 1
        
        # Calculate reading sessions (not every day)
        reading_days = self._generate_reading_schedule(start_date, end_date, total_pages)
        
        # Distribute pages across reading sessions
        pages_per_session = self._distribute_pages_across_sessions(total_pages, len(reading_days))
        
        synthetic_data = []
        current_page = 1
        
        for session_idx, (session_date, pages_in_session) in enumerate(zip(reading_days, pages_per_session)):
            # Generate reading time for this session (5-15 minutes per "Kindle page")
            session_duration = np.random.normal(10 * 60, 3 * 60, pages_in_session)  # seconds
            session_duration = np.clip(session_duration, 2 * 60, 20 * 60)  # 2-20 minutes per page
            
            # Add some variation to start time within the day
            session_start = session_date + timedelta(
                hours=np.random.randint(8, 22),  # Reading between 8 AM and 10 PM
                minutes=np.random.randint(0, 60)
            )
            
            for page_idx in range(int(pages_in_session)):
                if current_page <= total_pages:
                    page_start_time = session_start + timedelta(minutes=page_idx * 10)  # 10 min per page avg
                    
                    synthetic_data.append({
                        'id_book': book_id,
                        'page': current_page,
                        'title': title,
                        'authors': authors,
                        'pages': total_pages,
                        'duration': int(session_duration[page_idx]),
                        'start_datetime': page_start_time,
                        'language': language
                    })
                    current_page += 1
        
        return synthetic_data
    
    def _generate_reading_schedule(self, start_date, end_date, total_pages):
        """Generate realistic reading schedule"""
        total_days = (end_date - start_date).days + 1
        
        # Estimate reading days (assuming not reading every day)
        # More pages = more reading days, but with some randomness
        if total_pages < 200:
            reading_frequency = 0.6  # Read 60% of days
        elif total_pages < 400:
            reading_frequency = 0.7
        else:
            reading_frequency = 0.8
        
        estimated_reading_days = max(int(total_days * reading_frequency), 3)  # At least 3 sessions
        
        # Generate random reading dates
        all_dates = [start_date + timedelta(days=x) for x in range(total_days)]
        reading_dates = sorted(np.random.choice(all_dates, 
                                              size=min(estimated_reading_days, total_days), 
                                              replace=False))
        
        # Ensure we have start and end dates
        if start_date not in reading_dates:
            reading_dates[0] = start_date
        if end_date not in reading_dates:
            reading_dates[-1] = end_date
        
        return sorted(reading_dates)
    
    def _distribute_pages_across_sessions(self, total_pages, num_sessions):
        """Distribute pages across reading sessions"""
        if num_sessions == 1:
            return [total_pages]
        
        # Generate random distribution with some bias toward consistent sessions
        base_pages = total_pages // num_sessions
        remainder = total_pages % num_sessions
        
        pages_per_session = [base_pages] * num_sessions
        
        # Distribute remainder randomly
        remainder_indices = np.random.choice(num_sessions, remainder, replace=False)
        for idx in remainder_indices:
            pages_per_session[idx] += 1
        
        # Add some variation (±20% of base)
        for i in range(num_sessions):
            variation = int(pages_per_session[i] * 0.2 * np.random.uniform(-1, 1))
            pages_per_session[i] = max(1, pages_per_session[i] + variation)
        
        # Adjust to ensure total is correct
        current_total = sum(pages_per_session)
        if current_total != total_pages:
            diff = total_pages - current_total
            pages_per_session[-1] += diff
            pages_per_session[-1] = max(1, pages_per_session[-1])
        
        return pages_per_session
    
    def save_merged_data(self, output_path='output/kindle_reading_merged.csv'):
        """Save the merged dataframe to CSV"""
        if self.reading_data is not None:
            self.reading_data.to_csv(output_path, index=False)
            print(f"Saved merged data to {output_path}")
            print()
            #print(f"Total records: {len(self.reading_data)}")
            #print(f"Kindle books: {len(self.reading_data[self.reading_data['format'] == 'kindle']['id_book'].unique())}")
            #print(f"Paper books: {len(self.reading_data[self.reading_data['format'] == 'paperback']['id_book'].unique())}")
        else:
            print("No data to save. Please run load_and_process_data() first.")

    def _extract_from_sqlite(self):
        """Extract page_stat_data and book tables from SQLite database"""
        try:
            # Connect to SQLite database
            conn = sqlite3.connect(self.sqlite_path)
            # Extract page_stat_data table
            page_stats = pd.read_sql_query("SELECT * FROM page_stat_data", conn)
            # Extract book table
            books = pd.read_sql_query("SELECT * FROM book", conn)
            # Save as CSV files for backup/reference
            page_stats.to_csv('data/page_stat_data.csv', index=False)
            books.to_csv('data/book.csv', index=False)
            print(f"Extracted {len(page_stats)} page stat records and {len(books)} books from database")
            print()
            conn.close()
            return page_stats, books
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
            raise
        except Exception as e:
            print(f"Error extracting data: {e}")
            raise

if __name__ == "__main__":
    # Initialize with SQLite database path
    analyzer = KindleReadingAnalyzer('data/statistics.sqlite3')
    # Extract from SQLite and process data
    analyzer.load_and_process_data()
    analyzer.save_merged_data()

    plots = KindlePlots(analyzer.reading_data)

    analysis_year = datetime.now().year
    print(f"Generating plots for year: {analysis_year}")
    
    plots.create_weekly_reading_time()
    plots.create_weekly_reading_time(year=analysis_year)
    
    plots.create_reading_streaks()
    plots.create_reading_streaks(year=analysis_year)

    plots.create_yearly_reading_calendar(year=analysis_year)

    plots.create_yearly_streak_calendar(year=analysis_year)

    plots.create_hourly_session_distribution()
    plots.create_hourly_session_distribution(year=analysis_year)

    plots.plot_average_daily_monthly()
    plots.plot_average_daily_monthly(year=analysis_year)

    plots.create_session_length_distribution()
    plots.create_session_length_distribution(year=analysis_year)

    plots.create_book_completion_timeline()
    plots.create_book_completion_timeline(year=analysis_year)

    plots.create_reading_pace_timeline()
    plots.create_reading_pace_timeline(year=analysis_year)
    
    # Load and create Numbers plots
    reading_catalog = load_reading_catalog()
    if not reading_catalog.empty:
        numbers_plots = NumbersPlots(reading_catalog)
        
        numbers_plots.create_yearly_books_count()

        numbers_plots.create_monthly_books_count()
        numbers_plots.create_monthly_books_count(year=analysis_year)

        numbers_plots.create_yearly_books_bought_vs_read(year_start=2020)

        numbers_plots.create_statistics_summary()
    else:
        print("No reading catalog data available for Numbers plots")