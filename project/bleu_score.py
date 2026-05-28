import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from openpyxl import Workbook

# Load the two Excel files
ref_df = pd.read_excel("flickr8k/image_captions.xlsx")
gen_df = pd.read_excel("flickr8k/captioned_images_new.xlsx")

# Ensure columns are named consistently
ref_df.columns = ["image", "caption"]
gen_df.columns = ["image", "caption"]

# Merge on image filename
merged = pd.merge(ref_df, gen_df, on="image", suffixes=("_ref", "_gen"))

# Prepare BLEU score calculation
smooth_fn = SmoothingFunction().method1
bleu_scores = []

for _, row in merged.iterrows():
    reference = row["caption_ref"].split()
    candidate = row["caption_gen"].split()
    score = sentence_bleu([reference], candidate, smoothing_function=smooth_fn)
    bleu_scores.append((row["image"], score))

# Save results to new Excel file
wb = Workbook()
ws = wb.active
ws.title = "BLEU Scores"

# Write header
ws.append(["image", "BLEU_score"])

# Write rows
for image, score in bleu_scores:
    ws.append([image, score])

wb.save("flickr8k/bleu_scores.xlsx")

print("BLEU scores saved to flickr8k/bleu_scores.xlsx")