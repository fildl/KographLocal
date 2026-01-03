from src.db import DatabaseManager
from src.processing import DataProcessor
import pandas as pd

def main():
    print("--- Phase 1 Verification ---")
    
    # 1. Test connection
    print("Connecting to database...")
    db = DatabaseManager()
    try:
        page_stats, books = db.get_raw_data()
        print(f"✓ Data extracted: {len(page_stats)} page stats, {len(books)} books")
    except Exception as e:
        print(f"❌ Database error: {e}")
        return

    # 2. Test Processing
    print("Processing data...")
    processor = DataProcessor(page_stats, books)
    df = processor.process()
    
    print(f"✓ Data processed. Rows: {len(df)}")
    print(f"✓ Columns: {df.columns.tolist()}")
    
    # 3. Basic Stats
    total_hours = df['duration'].sum() / 3600
    print(f"\n--- Stats ---\nTotal Reading Time: {total_hours:.2f} hours")
    print(f"Total Books Read (touched): {df['id_book'].nunique()}")
    print(f"Total Sessions: {df['session_id'].nunique()}")
    print(f"Date Range: {df['start_datetime'].min()} to {df['start_datetime'].max()}")

if __name__ == "__main__":
    main()
