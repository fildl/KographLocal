import sqlite3
import pandas as pd
from pathlib import Path
import shutil
import os

class DatabaseManager:
    """
    Manages connection to the Koreader statistics database.
    Handles locating, copying, and querying the sqlite3 file.
    """
    
    def __init__(self, db_path: str = "data/statistics.sqlite3"):
        self.db_path = Path(db_path)
        self.conn = None
        self._update_database()
        
    def _update_database(self):
        """Check for new statistics.sqlite3 on Kindle and copy if available."""
        # Source directory (Kindle device)
        source_path = Path("/Volumes/Kindle/koreader/settings/statistics.sqlite3")
        
        try:
            # Check if Kindle is connected and file exists
            if source_path.exists():
                print(f"Found database on Kindle: {source_path}")
                
                # Check if we need to update (compare modification times or if local missing)
                if not self.db_path.exists() or source_path.stat().st_mtime > self.db_path.stat().st_mtime:
                    os.makedirs(self.db_path.parent, exist_ok=True)
                    shutil.copy2(source_path, self.db_path)
                    print(f"✓ Updated database from Kindle to: {self.db_path}")
                else:
                    print("✓ Local database is up to date with Kindle")
            else:
                # Kindle not connected
                pass 
                
        except Exception as e:
            print(f"Warning: Could not check/update from Kindle: {e}")

    def connect(self):
        """Establish connection to the SQLite database."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database file not found at {self.db_path}")
            
        try:
            self.conn = sqlite3.connect(self.db_path)
            # Enable accessing columns by name
            self.conn.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
            raise

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()

    def get_raw_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract the main tables: page_stat_data and book.
        
        Returns:
            tuple: (page_stats_df, books_df)
        """
        if not self.conn:
            self.connect()
            
        try:
            # Query page statistics
            page_query = "SELECT * FROM page_stat_data"
            page_stats = pd.read_sql_query(page_query, self.conn)
            
            # Query book metadata
            book_query = "SELECT * FROM book"
            books = pd.read_sql_query(book_query, self.conn)
            
            return page_stats, books
            
        except Exception as e:
            print(f"Error extracting data: {e}")
            raise
        finally:
            self.close()

    @staticmethod
    def ensure_db_exists(source_path: str = None, target_path: str = "data/statistics.sqlite3"):
        """
        Utilities to help user copy their DB if it doesn't exist.
        """
        target = Path(target_path)
        if target.exists():
            return True
            
        if source_path:
            source = Path(source_path)
            if source.exists():
                os.makedirs(target.parent, exist_ok=True)
                shutil.copy2(source, target)
                return True
                
        return False
