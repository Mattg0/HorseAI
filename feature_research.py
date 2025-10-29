import pandas as pd
import json

with open('x_predict_feature.json') as f:
    data = json.load(f)
df = pd.DataFrame(data)

# Missing value report
missing_report = {
    'feature': [],
    'missing_count': [],
    'missing_pct': [],
    'missing_in_recent': []  # Critical for recency
}

for col in df.columns:
    missing_count = df[col].isna().sum()
    missing_pct = (missing_count / len(df)) * 100
    
    missing_report['feature'].append(col)
    missing_report['missing_count'].append(missing_count)
    missing_report['missing_pct'].append(missing_pct)

missing_df = pd.DataFrame(missing_report).sort_values('missing_pct', ascending=False)
print(missing_df[missing_df['missing_pct'] > 0])