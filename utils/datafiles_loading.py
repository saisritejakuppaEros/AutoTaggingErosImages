import pandas as pd

# read CSV
df = pd.read_csv("/data/corerndimage/image_tagging/AutoTaggingErosImages/all_themes_dedup.csv")

# first 10 rows
print(df.head(10))
print(df.iloc[0])


# json_path = '/data/corerndimage/v2/target_processed/00290/tagging/results.json'
# json_path = '/data/corerndimage/v2/target_processed/00290/segmentation/results.json'

# import json

# with open(json_path, 'r') as f:
#     data = json.load(f)

# print(data[:100])