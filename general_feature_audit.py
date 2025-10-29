#!/usr/bin/env python3
"""
General Feature Audit for TabNet Features

Analyzes X_general_features.json which contains TabNet features from multiple races.
Adapted from feature_audit.py to handle the new JSON structure.
"""
import json
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

def load_data(filepath='X_general_features.json'):
    """
    Load and flatten the new JSON structure into a DataFrame.

    New structure:
    {
      "export_timestamp": "...",
      "total_races": 5,
      "total_horses": 87,
      "races": [
        {
          "race_id": "...",
          "hippo": "...",
          "date": "...",
          "partants": 16,
          "X_features": [{...}, {...}, ...],  # List of horse features
          "feature_names": ["age", "cotedirect", ...],
          "feature_count": 47
        },
        ...
      ]
    }
    """
    print(f"Loading data from {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)

    print(f"Export timestamp: {data.get('export_timestamp', 'unknown')}")
    print(f"Total races: {data.get('total_races', 0)}")
    print(f"Total horses: {data.get('total_horses', 0)}")

    # Flatten all races into a single DataFrame
    all_features = []
    races = data.get('races', [])

    for race in races:
        race_id = race.get('race_id', 'unknown')
        hippo = race.get('hippo', '')
        date = race.get('date', '')
        X_features = race.get('X_features', [])

        # Add race metadata to each horse's features
        for horse_features in X_features:
            horse_features['_race_id'] = race_id
            horse_features['_hippo'] = hippo
            horse_features['_date'] = date
            all_features.append(horse_features)

    df = pd.DataFrame(all_features)

    print(f"\nFlattened to DataFrame: {len(df)} horses, {len(df.columns)} columns")
    print(f"Feature columns (excluding metadata): {len(df.columns) - 3}")

    return df

def analyze_missing(df):
    """Analyze missing values in features."""
    # Exclude metadata columns
    feature_cols = [col for col in df.columns if not col.startswith('_')]
    df_features = df[feature_cols]

    missing = df_features.isnull().sum()
    missing_pct = (missing / len(df_features)) * 100
    result = pd.DataFrame({'count': missing, 'pct': missing_pct})
    result = result[result['count'] > 0].sort_values('pct', ascending=False)
    high_missing = result[result['pct'] > 50].index.tolist()
    return result, high_missing

def analyze_constant(df):
    """Detect constant and low variance features."""
    feature_cols = [col for col in df.columns if not col.startswith('_')]
    df_features = df[feature_cols]

    constant = []
    low_variance = []
    for col in df_features.select_dtypes(include=[np.number]).columns:
        nunique = df_features[col].nunique()
        if nunique == 1:
            constant.append(col)
        elif nunique / len(df_features) < 0.01:
            most_common = df_features[col].mode()[0] if len(df_features[col].mode()) > 0 else None
            freq = (df_features[col] == most_common).sum() / len(df_features) if most_common is not None else 0
            low_variance.append({
                'feature': col,
                'unique_pct': (nunique/len(df_features))*100,
                'most_common': most_common,
                'freq': freq
            })
    return constant, low_variance

def detect_leakage(df):
    """Detect potential data leakage features."""
    feature_cols = [col for col in df.columns if not col.startswith('_')]
    leakage_keywords = ['cl', 'ecar', 'place', 'rank', 'result', 'finish', 'arrived', 'position']
    potential_leakage = [col for col in feature_cols if any(kw in col.lower() for kw in leakage_keywords)]
    return potential_leakage

def analyze_distribution(df):
    """Analyze distribution and detect highly skewed features."""
    feature_cols = [col for col in df.columns if not col.startswith('_')]
    df_features = df[feature_cols]

    skewed = []
    for col in df_features.select_dtypes(include=[np.number]).columns:
        data = df_features[col].dropna()
        if len(data) > 0:
            skewness = stats.skew(data)
            kurtosis = stats.kurtosis(data)
            if abs(skewness) > 5:
                skewed.append({
                    'feature': col,
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'min': data.min(),
                    'max': data.max(),
                    'mean': data.mean(),
                    'median': data.median()
                })
    return skewed

def detect_outliers(df):
    """Detect features with high outlier percentages."""
    feature_cols = [col for col in df.columns if not col.startswith('_')]
    df_features = df[feature_cols]

    outlier_features = []
    for col in df_features.select_dtypes(include=[np.number]).columns:
        data = df_features[col].dropna()
        if len(data) > 0:
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 3 * IQR
            upper = Q3 + 3 * IQR
            outliers = ((data < lower) | (data > upper)).sum()
            outlier_pct = (outliers / len(data)) * 100
            if outlier_pct > 5:
                outlier_features.append({
                    'feature': col,
                    'outlier_pct': outlier_pct,
                    'outlier_count': outliers,
                    'lower_bound': lower,
                    'upper_bound': upper
                })
    return outlier_features

def analyze_correlation(df):
    """Detect highly correlated feature pairs."""
    feature_cols = [col for col in df.columns if not col.startswith('_')]
    df_features = df[feature_cols]

    numeric_df = df_features.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.95:
                high_corr.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    return high_corr

def analyze_scaling(df):
    """Analyze feature scaling needs."""
    feature_cols = [col for col in df.columns if not col.startswith('_')]
    df_features = df[feature_cols]

    scaling_info = []
    for col in df_features.select_dtypes(include=[np.number]).columns:
        data = df_features[col].dropna()
        if len(data) > 0:
            scaling_info.append({
                'feature': col,
                'min': data.min(),
                'max': data.max(),
                'range': data.max() - data.min(),
                'mean': data.mean(),
                'std': data.std()
            })
    scaling_df = pd.DataFrame(scaling_info).sort_values('range', ascending=False)
    return scaling_df

def check_logical_consistency(df):
    """Check for logical inconsistencies in features."""
    feature_cols = [col for col in df.columns if not col.startswith('_')]
    df_features = df[feature_cols]

    issues = []

    # Age check
    if 'age' in df_features.columns:
        invalid_age = ((df_features['age'] < 2) | (df_features['age'] > 15)).sum()
        if invalid_age > 0:
            issues.append(f"age: {invalid_age} values outside 2-15 range")

    # Ratio/percentage checks
    for col in df_features.columns:
        if 'ratio' in col.lower() or 'pct' in col.lower() or 'rate' in col.lower():
            data = df_features[col].dropna()
            if len(data) > 0:
                if data.min() < 0 or data.max() > 100:
                    if data.max() > 1 and data.max() <= 100:
                        continue
                    issues.append(f"{col}: ratio/percentage outside valid range (min={data.min():.2f}, max={data.max():.2f})")

        # Negative value checks (except cote fields)
        if col not in ['cotedirect', 'cote']:
            data = df_features[col].dropna()
            if len(data) > 0 and pd.api.types.is_numeric_dtype(df_features[col]):
                neg_count = (data < 0).sum()
                if neg_count > 0:
                    issues.append(f"{col}: {neg_count} negative values (may be invalid)")

    return issues

def analyze_zeros(df):
    """Analyze zero patterns in features."""
    feature_cols = [col for col in df.columns if not col.startswith('_')]
    df_features = df[feature_cols]

    zero_patterns = []
    for col in df_features.select_dtypes(include=[np.number]).columns:
        zero_count = (df_features[col] == 0).sum()
        zero_pct = (zero_count / len(df_features)) * 100
        if zero_pct == 100:
            zero_patterns.append({'feature': col, 'zero_pct': zero_pct, 'status': 'ALL_ZEROS'})
        elif zero_pct > 50:
            zero_patterns.append({'feature': col, 'zero_pct': zero_pct, 'status': 'SUSPICIOUS'})
    return zero_patterns

def analyze_by_race(df):
    """Analyze feature consistency across races."""
    if '_race_id' not in df.columns:
        return None

    feature_cols = [col for col in df.columns if not col.startswith('_')]
    race_analysis = []

    for race_id in df['_race_id'].unique():
        race_df = df[df['_race_id'] == race_id][feature_cols]
        race_analysis.append({
            'race_id': race_id,
            'horses': len(race_df),
            'missing_pct': race_df.isnull().sum().sum() / (len(race_df) * len(feature_cols)) * 100,
            'zero_pct': (race_df == 0).sum().sum() / (len(race_df) * len(feature_cols)) * 100
        })

    return pd.DataFrame(race_analysis)

def main():
    print("="*80)
    print("GENERAL FEATURE AUDIT (TabNet Features)")
    print("="*80)

    # Load data from new JSON structure
    try:
        df = load_data()
    except FileNotFoundError:
        print("\n❌ Error: X_general_features.json not found!")
        print("Run predictions first to generate the feature file.")
        return
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        return

    print(f"\nDataset: {len(df)} horses, {len(df.columns)} total columns")

    # Count feature vs metadata columns
    feature_cols = [col for col in df.columns if not col.startswith('_')]
    metadata_cols = [col for col in df.columns if col.startswith('_')]
    print(f"  - Feature columns: {len(feature_cols)}")
    print(f"  - Metadata columns: {len(metadata_cols)} {metadata_cols}")

    report_lines = []
    report_lines.append("="*80)
    report_lines.append("GENERAL FEATURE AUDIT REPORT (TabNet Features)")
    report_lines.append("="*80)
    report_lines.append(f"\nDataset: {len(df)} horses, {len(feature_cols)} features\n")

    # Race-level analysis
    print("\n" + "="*80)
    print("RACE-LEVEL ANALYSIS")
    print("="*80)
    race_df = analyze_by_race(df)
    if race_df is not None:
        print(f"\nRaces analyzed: {len(race_df)}")
        print(race_df.to_string())
        report_lines.append("\nRACE-LEVEL ANALYSIS\n")
        report_lines.append(race_df.to_string())

    # 1. Missing Values
    print("\n" + "="*80)
    print("1. MISSING VALUES ANALYSIS")
    print("="*80)
    missing_df, high_missing = analyze_missing(df)
    if len(missing_df) > 0:
        print(f"\nFeatures with missing values: {len(missing_df)}")
        print(missing_df.head(10))
        if high_missing:
            print(f"\n⚠️  High missing (>50%): {high_missing}")
    else:
        print("✓ No missing values")
    report_lines.append("\n1. MISSING VALUES\n")
    report_lines.append(missing_df.to_string() if len(missing_df) > 0 else "No missing values")

    # 2. Constant Features
    print("\n" + "="*80)
    print("2. CONSTANT/LOW VARIANCE FEATURES")
    print("="*80)
    constant, low_var = analyze_constant(df)
    if constant:
        print(f"\n⚠️  Constant features (remove): {constant}")
    if low_var:
        print(f"\nLow variance features (<1% unique):")
        for item in low_var[:5]:
            print(f"  {item['feature']}: {item['unique_pct']:.2f}% unique, {item['freq']*100:.1f}% = {item['most_common']}")
    report_lines.append("\n2. CONSTANT FEATURES\n")
    report_lines.append(f"Constant: {constant}\n")
    report_lines.append(f"Low variance: {[x['feature'] for x in low_var]}")

    # 3. Leakage
    print("\n" + "="*80)
    print("3. DATA LEAKAGE DETECTION")
    print("="*80)
    leakage = detect_leakage(df)
    if leakage:
        print(f"⚠️  Potential leakage: {leakage}")
    else:
        print("✓ No obvious leakage")
    report_lines.append("\n3. POTENTIAL LEAKAGE\n")
    report_lines.append(str(leakage))

    # 4. Distribution
    print("\n" + "="*80)
    print("4. DISTRIBUTION ISSUES")
    print("="*80)
    skewed = analyze_distribution(df)
    if skewed:
        print(f"\nHighly skewed (|skew|>5): {len(skewed)} features")
        for item in skewed[:5]:
            print(f"  {item['feature']}: skew={item['skewness']:.2f}, range=[{item['min']:.2f}, {item['max']:.2f}]")
    report_lines.append("\n4. SKEWED FEATURES\n")
    report_lines.append(str([x['feature'] for x in skewed]))

    # 5. Outliers
    print("\n" + "="*80)
    print("5. OUTLIER DETECTION")
    print("="*80)
    outliers = detect_outliers(df)
    if outliers:
        print(f"\nFeatures with >5% outliers: {len(outliers)}")
        for item in outliers[:5]:
            print(f"  {item['feature']}: {item['outlier_pct']:.1f}% outliers ({item['outlier_count']} records)")
    report_lines.append("\n5. OUTLIER HEAVY FEATURES\n")
    report_lines.append(str([x['feature'] for x in outliers]))

    # 6. Correlation
    print("\n" + "="*80)
    print("6. MULTICOLLINEARITY")
    print("="*80)
    high_corr = analyze_correlation(df)
    if high_corr:
        print(f"\nHighly correlated pairs (>0.95): {len(high_corr)}")
        for item in high_corr[:5]:
            print(f"  {item['feature1']} ↔ {item['feature2']}: {item['correlation']:.3f}")
    report_lines.append("\n6. HIGHLY CORRELATED PAIRS\n")
    report_lines.append(str([[x['feature1'], x['feature2']] for x in high_corr]))

    # 7. Scaling
    print("\n" + "="*80)
    print("7. FEATURE SCALING")
    print("="*80)
    scaling_df = analyze_scaling(df)
    print("\nTop 5 features by range:")
    print(scaling_df[['feature', 'min', 'max', 'range', 'std']].head())

    # 8. Logical Consistency
    print("\n" + "="*80)
    print("8. LOGICAL CONSISTENCY")
    print("="*80)
    consistency = check_logical_consistency(df)
    if consistency:
        print("\n⚠️  Issues:")
        for issue in consistency[:10]:
            print(f"  {issue}")
    else:
        print("✓ No obvious issues")
    report_lines.append("\n8. LOGICAL ISSUES\n")
    report_lines.append('\n'.join(consistency) if consistency else "None")

    # 9. Zero Patterns
    print("\n" + "="*80)
    print("9. ZERO/NULL PATTERNS")
    print("="*80)
    zeros = analyze_zeros(df)
    if zeros:
        print(f"\nFeatures with suspicious zero patterns: {len(zeros)}")
        for item in zeros[:10]:
            print(f"  {item['feature']}: {item['zero_pct']:.1f}% zeros ({item['status']})")
    report_lines.append("\n9. ZERO PATTERNS\n")
    report_lines.append(str([x['feature'] for x in zeros]))

    # Save report
    with open('general_feature_audit_report.txt', 'w') as f:
        f.write('\n'.join(report_lines))

    # Save JSON summary
    summary = {
        'constant_features': constant,
        'high_missing_features': high_missing,
        'zero_variance_features': [x['feature'] for x in zeros if x['status'] == 'ALL_ZEROS'],
        'highly_correlated_pairs': [[x['feature1'], x['feature2']] for x in high_corr],
        'requires_log_transform': [x['feature'] for x in skewed],
        'outlier_heavy_features': [x['feature'] for x in outliers],
        'potential_leakage': leakage
    }
    with open('general_feature_audit_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save recommendations CSV
    recommendations = []
    for feat in constant:
        recommendations.append({
            'feature_name': feat,
            'issue_type': 'constant',
            'severity': 'high',
            'recommendation': 'Remove (no variance)'
        })
    for feat in high_missing:
        recommendations.append({
            'feature_name': feat,
            'issue_type': 'missing',
            'severity': 'high',
            'recommendation': 'Remove or impute (>50% missing)'
        })
    for item in zeros:
        if item['status'] == 'ALL_ZEROS':
            recommendations.append({
                'feature_name': item['feature'],
                'issue_type': 'all_zeros',
                'severity': 'high',
                'recommendation': 'Remove (not calculated)'
            })
    for item in skewed:
        recommendations.append({
            'feature_name': item['feature'],
            'issue_type': 'skewed',
            'severity': 'medium',
            'recommendation': 'Apply log/sqrt transform'
        })
    for pair in high_corr:
        recommendations.append({
            'feature_name': f"{pair['feature1']},{pair['feature2']}",
            'issue_type': 'multicollinear',
            'severity': 'medium',
            'recommendation': 'Remove one feature'
        })
    for feat in leakage:
        recommendations.append({
            'feature_name': feat,
            'issue_type': 'leakage',
            'severity': 'high',
            'recommendation': 'Verify not using future data'
        })

    pd.DataFrame(recommendations).to_csv('general_feature_recommendations.csv', index=False)

    print("\n" + "="*80)
    print("✓ Reports saved:")
    print("  - general_feature_audit_report.txt")
    print("  - general_feature_audit_summary.json")
    print("  - general_feature_recommendations.csv")
    print("="*80)

if __name__ == '__main__':
    main()
