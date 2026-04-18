import pandas as pd
import numpy as np

# Load the dataset
df=pd.read_csv('data/raw/bothawk_data.csv')
# Explore the dataset
print(df.head())
print(df.info())
print(df['label'].value_counts())
# Missing value analysis
missing_values=df.isnull().sum()
print("Missing values in each column:\n", missing_values)

#drop irrelevant columns
df = df.drop(columns=['actor_id', 'name', 'email'], errors='ignore')
print("Irrelevant columns dropped: 'actor_id', 'name', 'email' (if they existed).")

#encode target variable
df['label']=(df['label']=='Bot').astype(int)
print("Target variable 'label' encoded: 'Bot' as 1 and 'Human' as 0.")

#handle missing values using median imputation
if df.isnull().sum().sum()>0:
    df.fillna(df.median(numeric_only=True), inplace=True)
print("Missing values handled using median imputation.")

# ── CHECK LOW-VARIANCE / SPARSE FEATURES 
print("\nChecking low-signal columns:")

low_signal_cols = ['login', 'bio', 'tag']

for col in low_signal_cols:
    if col in df.columns:
        unique_vals = df[col].value_counts(normalize=True)
        print(f"\n{col} distribution:")
        print(unique_vals.head())

        # If one value dominates (>95%), drop it
        if unique_vals.iloc[0] > 0.95:
            df.drop(columns=[col], inplace=True)
            print(f"→ Dropped {col} (low variance)")

#handle outliers using Winsorization    
winsor_cols = [
    'Number of followers', 'Number of following', 'Number of Activity',
    'Number of Issue', 'Number of Pull Request', 'Number of Repository',
    'Number of Commit', 'Number of Connection Account', 'Median Response Time'
]
for col in winsor_cols:
    if col in df.columns:
        cap=df[col].quantile(0.99)
        count_outliers=(df[col]>cap).sum()
        df[col]=df[col].clip(upper=cap)
        print(f'{col}: Capped {count_outliers} outliers at {cap}')
print("Outlier handling completed using Winsorization for specified columns.")

# check for skewness and apply log transformation if necessary
skew_vals=df.skew(numeric_only=True).sort_values(ascending=False)
print("\nSkewness of numeric features:\n", skew_vals)
skew_threshold=2
for col in skew_vals.index:
    #  Skip target column
    if col == 'label':
        continue
    
    # Skip binary columns
    if df[col].nunique() <= 2:
        continue
    if abs(skew_vals[col]) > skew_threshold:
        df[col]=df[col].apply(lambda x: np.log1p(x) if x>=0 else x)
        print(f"Applied log transformation to {col} (skewness: {skew_vals[col]:.2f})")
print("Skewness handling completed for features with skewness > 2.")

# feature engineering: create new features
df['issue_activity_ratio'] = df['Number of Issue'] / (df['Number of Activity'] + 1)
df['pr_activity_ratio'] = df['Number of Pull Request'] / (df['Number of Activity'] + 1)
df['commit_per_repo'] = df['Number of Commit'] / (df['Number of Repository'] + 1)
df['follower_following_ratio'] = df['Number of followers'] / (df['Number of following'] + 1)
df['activity_per_day'] = df['Number of Activity'] / (df['Number of Active day'] + 1)
print("Feature engineering completed. New features added: 'issue_activity_ratio', 'pr_activity_ratio', 'commit_per_repo', 'follower_following_ratio', 'activity_per_day'.")

# Save the preprocessed dataset
df.to_csv('data/processed/preprocessed_data.csv', index=False)
print("Preprocessing completed and saved to 'data/processed/preprocessed_data.csv'")