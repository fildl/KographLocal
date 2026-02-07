
import sys
import pandas as pd
import difflib

# Add src to path
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from processing import DataProcessor
from db import DatabaseManager

def verify_matching():
    print("--- Verifying Fuzzy Matching ---")
    
    # 1. Load Base Data
    db = DatabaseManager()
    try:
        raw_data, books = db.get_raw_data()
    except Exception as e:
        print(f"Error loading DB data: {e}")
        return

    processor = DataProcessor(raw_data, books)
    kindle_data = processor.process()
    
    
    # 2. Add Paper/Audio Books to have full 'target' set
    paper_path = 'data/paper_books.csv'
    audio_path = 'data/audio_books.csv'
    
    # Start with Kindle data
    target_data = kindle_data.copy()
    
    if os.path.exists(paper_path):
        target_data = processor.get_data_with_paper_books(paper_path)
    if os.path.exists(audio_path):
        target_data = processor.get_data_with_audio_books(audio_path, current_combined_df=target_data)
        
    print(f"Total Unique Books in App: {target_data['title'].nunique()}")
    
    # 3. Load Metadata
    metadata_path = '/Users/filippodiludovico/Library/Mobile Documents/com~apple~Numbers/Documents/reading.numbers'
    if not os.path.exists(metadata_path):
        print("Metadata file not found.")
        return

    # Call the enrichment method which applies matching
    enriched_df = processor.get_data_with_metadata(metadata_path, current_combined_df=target_data)
    
    if 'author_country' not in enriched_df.columns:
        print("Metadata enrichment failed or produced no matches.")
        return

    # 4. Analyze Matches
    # Identify which titles got a match
    unique_books = enriched_df[['title', 'author_country']].drop_duplicates()
    matched = unique_books[unique_books['author_country'].notna()]
    unmatched = unique_books[unique_books['author_country'].isna()]
    
    print(f"Matched Books: {len(matched)}")
    print(f"Unmatched Books: {len(unmatched)}")
    
    print("\n--- MATCHED BOOKS ---")
    if not matched.empty:
        # Show what it matched to (Title -> Country)
        print(matched[['title', 'author_country']].to_string(index=False))
    else:

        print("No matches found.")
        
    print("\n--- UNMATCHED BOOKS ---")
    if not unmatched.empty:
        print(unmatched['title'].to_string(index=False))
    else:

        print("No unmatched books.")

if __name__ == "__main__":
    verify_matching()
