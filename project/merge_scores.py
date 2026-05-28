import pandas as pd

# Read the CNN-LSTM mean BLEU scores
cnn_df = pd.read_excel("flickr8k/bleu_scores_cnn_lstm_mean.xlsx")
cnn_df.columns = ["image", "BLEU_score_cnn_lstm"]

# Read the BLIP mean BLEU scores
blip_df = pd.read_excel("flickr8k/bleu_scores_blip_mean.xlsx")
blip_df.columns = ["image", "BLEU_score_blip"]

# Merge on image filename
merged = pd.merge(cnn_df, blip_df, on="image", how="inner")

# Save to new Excel file
merged.to_excel("flickr8k/bleu_scores_comparison.xlsx", index=False)

print("Merged BLEU scores saved to flickr8k/bleu_scores_comparison.xlsx")