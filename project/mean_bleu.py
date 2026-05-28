import pandas as pd

# Load the BLEU scores file
df = pd.read_excel("flickr8k/bleu_scores_blip.xlsx")

# Ensure the columns are named correctly
df.columns = ["image", "BLEU_score"]

# Group by image and compute mean BLEU score
mean_scores = df.groupby("image", as_index=False)["BLEU_score"].mean()

# Save to a new Excel file
mean_scores.to_excel("flickr8k/bleu_scores_blip_mean.xlsx", index=False)

print("Mean BLEU scores saved to flickr8k/bleu_scores_blip_mean.xlsx")