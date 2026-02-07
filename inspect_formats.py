
from numbers_parser import Document
import pandas as pd

def inspect_formats(file_path):
    print(f"Reading file: {file_path}")
    try:
        doc = Document(file_path)
        sheets = doc.sheets
        if not sheets: return
        
        table = sheets[0].tables[0]
        data = table.rows(values_only=True)
        
        if not data: return

        # Load into DF
        df = pd.DataFrame(data[1:], columns=data[0])
        df.columns = df.columns.str.strip().str.lower()
        
        if 'format' in df.columns:
            print("--- Unique Formats Found ---")
            print(df['format'].unique())
            
            print("\n--- Rows with format 'Paperback' (sample) ---")
            print(df[df['format'] == 'Paperback'].head())
        else:
            print("Column 'format' not found.")
            print("Columns found:", df.columns.tolist())

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    file_path = "/Users/filippodiludovico/Library/Mobile Documents/com~apple~Numbers/Documents/reading.numbers"
    inspect_formats(file_path)
