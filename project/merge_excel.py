import pandas as pd

# Load both Excel files
old_df = pd.read_excel('captioned_images_old.xlsx')  # Contains 'Filename' and 'old_model_caption'
new_df = pd.read_excel('captioned_images_new.xlsx')  # Contains 'Filename' and 'new_model_caption'

# Merge on 'Filename'
merged_df = pd.merge(old_df, new_df, on='Filename', how='inner')

# Save the result to a new Excel file
merged_df.to_excel('merged_captions.xlsx', index=False)