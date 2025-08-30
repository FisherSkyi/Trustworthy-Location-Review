import os
import pandas as pd

# Data Loading Function
def load_and_preprocess_dataset(file_path):
    """Load dataset, select and rename necessary columns."""
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} rows from {file_path}")
            
            # Keep only 'text' and 'rating' columns
            df = df[['text', 'rating']]
            
            # Rename 'text' to 'review_text'
            df = df.rename(columns={'text': 'review_text'})
            
            return df
        else:
            print(f"File not found: {file_path}")
            return None
    except Exception as e:
        print(f"Error loading or processing file: {e}")
        return None


