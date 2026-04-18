import pandas as pd
import numpy as np

# Load the dataset
df=pd.read_csv('data/raw/bots_vs_users.csv')
# Explore the dataset
print(df.head())
print(df.info())

# Feature Selection
Features=[
    "posts_count",
    "avg_likes",
    "links_ratio",
    "hashtags_ratio",
    "avg_text_length",
    "avg_comments",
    "posting_frequency_days",
    "avg_text_uniqueness",
    "phone_numbers_ratio",
    "ads_ratio"
]
Target="target"

df_selected=df[Features+[Target]]

# Handle missing values
df_selected=df_selected.dropna(subset=Features)
print(df_selected.isnull().sum())

# Split the dataset into features and target variable
X=df_selected[Features]
y=df_selected[Target]

# Save the preprocessed data
df_selected.to_csv("data/processed/cleaned_data.csv", index=False)
