import pandas as pd

# Load the Excel file
df = pd.read_excel('flickr8k/image_captions.xlsx')

# Count unique entries in the 'Filename' column
unique_count = df['Filename'].nunique()

print(f"Number of unique filenames: {unique_count}")