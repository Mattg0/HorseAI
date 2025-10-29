#!/usr/bin/env python3
import json
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

def load_data(filepath='X_predict_feature.json'):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def analyze_missing(df):
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    result = pd.DataFrame({'count': missing, 'pct': missing_pct})
    result = result[result['count'] > 0].sort_values('pct', ascending=False)
    high_missing = result[result['pct'] > 50].index.tolist()
    return result, high_missing

def analyze_constant(df):
    constant = []
    low_variance = []
    for col in df.select_dtypes(include=[np.number]).columns:
        nunique = df[col].nunique()
        if nunique == 1:
            constant.append(col)
        elif nunique / len(df) < 0.01:
            most_common = df[col].mode()[0] if len(df[col].mode()) > 0 else None
            freq = (df[col] == most_common).sum() / len(df) if most_common is not None else 0
            low_variance.append({'feature': col, 'unique_pct': (nunique/len(df))*100, 'most_common': most_common, 'freq': freq})
    return constant, low_variance

def detect_leakage(df):
    leakage_keywords = ['cl', 'ecar', 'place', 'rank', 'result', 'finish', 'arrived', 'position']
    potential_leakage = [col for col in df.columns if any(kw in col.lower() for kw in leakage_keywords)]
    return potential_leakage

def analyze_distribution(df):
    skewed = []
    for col in df.select_dtypes(include=[np.number]).columns:
        data = df[col].dropna()
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
    outlier_features = []
    for col in df.select_dtypes(include=[np.number]).columns:
        data = df[col].dropna()
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
    numeric_df = df.select_dtypes(include=[np.number])
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
    scaling_info = []
    for col in df.select_dtypes(include=[np.number]).columns:
        data = df[col].dropna()
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
    issues = []
    if 'age' in df.columns:
        invalid_age = ((df['age'] < 2) | (df['age'] > 15)).sum()
        if invalid_age > 0:
            issues.append(f"age: {invalid_age} values outside 2-15 range")

    for col in df.columns:
        if 'ratio' in col.lower() or 'pct' in col.lower() or 'rate' in col.lower():
            data = df[col].dropna()
            if len(data) > 0:
                if data.min() < 0 or data.max() > 100:
                    if data.max() > 1 and data.max() <= 100:
                        continue
                    issues.append(f"{col}: ratio/percentage outside valid range (min={data.min():.2f}, max={data.max():.2f})")

        if col not in ['cotedirect', 'cote']:
            data = df[col].dropna()
            if len(data) > 0 and pd.api.types.is_numeric_dtype(df[col]):
                neg_count = (data < 0).sum()
                if neg_count > 0:
                    issues.append(f"{col}: {neg_count} negative values (may be invalid)")

    return issues

def analyze_zeros(df):
    zero_patterns = []
    for col in df.select_dtypes(include=[np.number]).columns:
        zero_count = (df[col] == 0).sum()
        zero_pct = (zero_count / len(df)) * 100
        if zero_pct == 100:
            zero_patterns.append({'feature': col, 'zero_pct': zero_pct, 'status': 'ALL_ZEROS'})
        elif zero_pct > 50:
            zero_patterns.append({'feature': col, 'zero_pct': zero_pct, 'status': 'SUSPICIOUS'})
    return zero_patterns

def main():
    print("="*80)
    print("FEATURE AUDIT")
    print("="*80)

    df = load_data()
    print(f"\nDataset: {len(df)} records, {len(df.columns)} features")

    report_lines = []
    report_lines.append("="*80)
    report_lines.append("FEATURE AUDIT REPORT")
    report_lines.append("="*80)
    report_lines.append(f"\nDataset: {len(df)} records, {len(df.columns)} features\n")

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
    with open('feature_audit_report.txt', 'w') as f:
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
    with open('feature_audit_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save recommendations CSV
    recommendations = []
    for feat in constant:
        recommendations.append({'feature_name': feat, 'issue_type': 'constant', 'severity': 'high', 'recommendation': 'Remove (no variance)'})
    for feat in high_missing:
        recommendations.append({'feature_name': feat, 'issue_type': 'missing', 'severity': 'high', 'recommendation': 'Remove or impute (>50% missing)'})
    for item in zeros:
        if item['status'] == 'ALL_ZEROS':
            recommendations.append({'feature_name': item['feature'], 'issue_type': 'all_zeros', 'severity': 'high', 'recommendation': 'Remove (not calculated)'})
    for item in skewed:
        recommendations.append({'feature_name': item['feature'], 'issue_type': 'skewed', 'severity': 'medium', 'recommendation': 'Apply log/sqrt transform'})
    for pair in high_corr:
        recommendations.append({'feature_name': f"{pair['feature1']},{pair['feature2']}", 'issue_type': 'multicollinear', 'severity': 'medium', 'recommendation': 'Remove one feature'})
    for feat in leakage:
        recommendations.append({'feature_name': feat, 'issue_type': 'leakage', 'severity': 'high', 'recommendation': 'Verify not using future data'})

    pd.DataFrame(recommendations).to_csv('feature_recommendations.csv', index=False)

    print("\n" + "="*80)
    print("✓ Reports saved:")
    print("  - feature_audit_report.txt")
    print("  - feature_audit_summary.json")
    print("  - feature_recommendations.csv")
    print("="*80)

if __name__ == '__main__':
    main()
