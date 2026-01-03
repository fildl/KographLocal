import pandas as pd
from numbers_parser import Document
import os

def load_reading_catalog():
    """Load reading catalog from Numbers file and return as DataFrame"""
    # Path to your Numbers file
    file_path = "/Users/filippodiludovico/Library/Mobile Documents/com~apple~Numbers/Documents/reading.numbers"
    
    try:
        # Load the Numbers document
        print("Loading Numbers document...")
        doc = Document(file_path)
        
        # Get the first sheet
        sheet = doc.sheets[0]
        print(f"Sheet name: {sheet.name}")
        
        # Get the first table from the first sheet
        table = sheet.tables[0]
        print(f"Table name: {table.name}")
        print(f"Table dimensions: {table.num_rows} rows x {table.num_cols} columns")
        
        # Extract all rows
        rows = list(table.rows())
        
        if not rows:
            print("No data found in the table")
            return pd.DataFrame()
        
        # Extract actual values from cell objects
        # First row as column headers, rest as data
        headers = []
        for i, cell in enumerate(rows[0]):
            if cell is not None and hasattr(cell, 'value') and cell.value is not None:
                headers.append(str(cell.value))
            else:
                headers.append(f"Column_{i}")
        
        data_rows = []
        for row in rows[1:]:  # Skip header row
            data_row = []
            for cell in row:
                if cell is not None and hasattr(cell, 'value'):
                    data_row.append(cell.value)
                else:
                    data_row.append(None)
            data_rows.append(data_row)
        
        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=headers)
        
        # Convert year to integer (handling NaN and non-numeric values)
        def safe_int_convert(value):
            if pd.isna(value) or value is None:
                return None
            try:
                return int(float(value))  # Convert to float first, then int
            except (ValueError, TypeError):
                return None  # Return None for non-numeric values
        
        if 'year' in df.columns:
            df['year'] = df['year'].apply(safe_int_convert)
        if 'pages' in df.columns:
            df['pages'] = df['pages'].apply(safe_int_convert)
        
        # Convert Duration from datetime to minutes
        def convert_duration_to_minutes(x):
            if pd.isna(x) or x is None:
                return None
            try:
                return x.hour * 60 + x.minute
            except AttributeError:
                return None
        
        if 'duration' in df.columns:
            df['duration'] = df['duration'].apply(convert_duration_to_minutes)
        
        print("\nDataFrame created successfully!")
        """
        print(f"Shape: {df.shape}")
        print("\nColumn names:")
        for i, col in enumerate(df.columns):
            print(f"{i}: {col}")
        
        print("\nFirst few rows:")
        print(df.head())
        """
        
        # Save the DataFrame for future use
        os.makedirs('output', exist_ok=True)
        df.to_csv('output/reading_catalog.csv', index=False)
        print("\nDataFrame saved to 'output/reading_catalog.csv'")
        print()
        
        return df
        
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        print("Please check if the file path is correct and the file exists.")
        return pd.DataFrame()
    except ImportError:
        print("numbers-parser library not found. Please install it with:")
        print("pip install numbers-parser")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Error type: {type(e).__name__}")
        return pd.DataFrame()

if __name__ == "__main__":
    # For testing purposes
    df = load_reading_catalog()
    if not df.empty:
        print("Successfully loaded reading catalog!")
    else:
        print("Failed to load reading catalog.")