#!/usr/bin/env python3
"""
Debug script to analyze TabNet feature preparation and NULL values.
This script simulates the prediction pipeline and reports on feature availability.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.env_setup import AppConfig
from core.orchestrators.embedding_feature import FeatureEmbeddingOrchestrator
from core.calculators.static_feature_calculator import FeatureCalculator


def load_sample_race_data(db_path: str, limit: int = 5) -> tuple:
    """Load a sample race from the database for testing - USES DAILY_RACE (prediction data).

    Returns:
        Tuple of (race_df with race-level fields added, race_data dict)
    """
    import sqlite3
    import json

    conn = sqlite3.connect(db_path)

    # CRITICAL FIX: Load MULTIPLE races to show feature variation across races
    query = """
    SELECT comp, participants, typec, dist, natpis, meteo, temperature,
           forceVent, directionVent, corde, jour, hippo, quinte
    FROM daily_race
    LIMIT ?
    """

    cursor = conn.cursor()
    cursor.execute(query, (limit,))
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        raise ValueError("No races found in daily_race table")

    print(f"Loading {len(rows)} races for realistic feature variation analysis...")

    # Combine all races into a single DataFrame
    all_race_dfs = []
    all_race_data = []

    for row in rows:
        # Unpack row
        (comp, participants_json, typec, dist, natpis, meteo, temperature,
         forceVent, directionVent, corde, jour, hippo, quinte) = row

        # Create race_data dict
        race_data = {
            'comp': comp,
            'typec': typec,
            'dist': dist,
            'natpis': natpis,
            'meteo': meteo,
            'temperature': temperature,
            'forceVent': forceVent,
            'directionVent': directionVent,
            'corde': corde,
            'jour': jour,
            'hippo': hippo,
            'quinte': quinte
        }
        all_race_data.append(race_data)

        # Expand participants JSON
        participants = json.loads(participants_json)

        # Convert to DataFrame
        race_df = pd.DataFrame(participants)

        # CRITICAL FIX: Add race-level fields to each participant
        race_attributes = ['typec', 'dist', 'natpis', 'meteo', 'temperature',
                           'forceVent', 'directionVent', 'corde', 'jour',
                           'hippo', 'quinte']

        for field in race_attributes:
            if field in race_data and race_data[field] is not None:
                race_df[field] = race_data[field]

        # Add comp to DataFrame
        race_df['comp'] = comp

        # Remove target column for prediction simulation
        for col in ['final_position', 'cl', 'narrivee']:
            if col in race_df.columns:
                race_df = race_df.drop(col, axis=1)

        all_race_dfs.append(race_df)

    # Combine all races
    combined_df = pd.concat(all_race_dfs, ignore_index=True)

    print(f"Combined {len(rows)} races: {len(combined_df)} total horses")
    print(f"Distance variation: {combined_df['dist'].nunique()} unique distances")
    print(f"Age variation: {combined_df['age'].nunique()} unique ages")
    print(f"Temperature variation: {combined_df['temperature'].nunique()} unique temperatures")

    # DEBUG: Show critical field values from raw data (first race, first horse)
    print(f"\nRAW DATA CHECK (first race, first horse):")
    first_participants = json.loads(rows[0][1])
    if len(first_participants) > 0:
        first_horse = first_participants[0]
        print(f"  cotedirect: {first_horse.get('cotedirect')}")
        print(f"  coteprob: {first_horse.get('coteprob')}")
        print(f"  gainsAnneeEnCours: {first_horse.get('gainsAnneeEnCours')}")
        print(f"  age: {first_horse.get('age')}")
        print(f"  equipment - oeil: {first_horse.get('oeil')}, oeilFirstTime: {first_horse.get('oeilFirstTime')}")

    return combined_df, all_race_data[0]  # Return combined df but first race metadata


def analyze_tabnet_features():
    """
    Main analysis function that simulates TabNet prediction pipeline
    and reports on feature availability and NULL values.
    """
    print("=" * 80)
    print("TABNET FEATURE PREPARATION ANALYSIS")
    print("=" * 80)

    # Initialize config and orchestrator
    config = AppConfig()
    db_path = config.get_sqlite_dbpath(config._config.base.active_db)

    print(f"\n1. Database: {db_path}")

    orchestrator = FeatureEmbeddingOrchestrator(
        sqlite_path=db_path,
        verbose=True  # Enable verbose to see debug output
    )

    # Load TabNet feature columns from saved model (use latest model)
    # Try to find latest TabNet model automatically
    from utils.model_manager import get_model_manager
    model_manager = get_model_manager()
    model_paths = model_manager.get_all_model_paths()

    if 'tabnet' in model_paths:
        tabnet_model_path = Path(model_paths['tabnet'])
    else:
        # Fallback to manual path
        tabnet_model_path = Path("models/2025-10-01/2years_154355")

    tabnet_config_path = tabnet_model_path / "tabnet_config.json"

    if not tabnet_config_path.exists():
        print(f"ERROR: TabNet config not found at {tabnet_config_path}")
        return

    with open(tabnet_config_path, 'r') as f:
        tabnet_config = json.load(f)
        expected_features = tabnet_config.get('feature_columns', [])

    print(f"   Using TabNet model: {tabnet_model_path}")

    print(f"\n2. Expected TabNet Features: {len(expected_features)}")

    # Load sample race data for prediction (use multiple races to show variation)
    print(f"\n3. Loading sample race data...")
    race_df, race_data = load_sample_race_data(db_path, limit=50)  # Load 50 races for realistic analysis

    print(f"\n4. Input race data shape: {race_df.shape}")
    print(f"   Columns: {len(race_df.columns)}")

    # DEBUG: Check critical fields in raw DataFrame
    print(f"\n   CRITICAL FIELDS IN RAW DATAFRAME:")
    for feat in ['cotedirect', 'coteprob', 'gainsAnneeEnCours', 'dist', 'temperature']:
        if feat in race_df.columns:
            values = race_df[feat].head(3).tolist()
            unique = race_df[feat].nunique()
            print(f"   {feat}: {values} ({unique} unique)")
        else:
            print(f"   {feat}: MISSING")

    # Step 1: Apply FeatureCalculator (same as prediction)
    print(f"\n5. Applying FeatureCalculator...")
    df_with_features = FeatureCalculator.calculate_all_features(race_df)
    print(f"   After FeatureCalculator: {df_with_features.shape}")
    print(f"   Columns added: {len(df_with_features.columns) - len(race_df.columns)}")

    # DEBUG: Check critical fields after FeatureCalculator
    print(f"\n   CRITICAL FIELDS AFTER FeatureCalculator:")
    for feat in ['cotedirect', 'coteprob', 'gainsAnneeEnCours', 'dist', 'temperature']:
        if feat in df_with_features.columns:
            values = df_with_features[feat].head(3).tolist()
            unique = df_with_features[feat].nunique()
            print(f"   {feat}: {values} ({unique} unique)")
        else:
            print(f"   {feat}: MISSING")

    # Step 2: Apply orchestrator's TabNet preparation
    print(f"\n6. Applying orchestrator.prepare_tabnet_features()...")
    complete_df = orchestrator.prepare_tabnet_features(
        df_with_features,
        use_cache=False
    )
    print(f"   After prepare_tabnet_features: {complete_df.shape}")
    print(f"   Columns in complete_df: {list(complete_df.columns)[:20]}...")  # Show first 20

    # DEBUG: Check critical fields after prepare_tabnet_features
    print(f"\n   CRITICAL FIELDS AFTER prepare_tabnet_features:")
    for feat in ['cotedirect', 'coteprob', 'gainsAnneeEnCours', 'dist', 'temperature']:
        if feat in complete_df.columns:
            values = complete_df[feat].head(3).tolist()
            unique = complete_df[feat].nunique()
            min_val = complete_df[feat].min()
            max_val = complete_df[feat].max()
            print(f"   {feat}: {values} ({unique} unique, range [{min_val:.1f}, {max_val:.1f}])")
        else:
            print(f"   {feat}: MISSING")

    # Step 3: Check which expected features are available
    print(f"\n7. Analyzing feature availability...")

    available_features = [f for f in expected_features if f in complete_df.columns]
    missing_features = [f for f in expected_features if f not in complete_df.columns]

    print(f"   Available: {len(available_features)}/{len(expected_features)} ({len(available_features)/len(expected_features)*100:.1f}%)")
    print(f"   Missing: {len(missing_features)}")

    if missing_features:
        print(f"\n   Missing features:")
        for feat in missing_features:
            print(f"      - {feat}")

    # Step 4: Analyze NULL values in available features
    print(f"\n8. NULL VALUE ANALYSIS FOR EACH FEATURE")
    print("=" * 80)
    print(f"{'#':<4} {'Feature':<40} {'NULL Count':<12} {'NULL %':<10} {'Dtype':<10}")
    print("-" * 80)

    null_summary = []

    for i, feat in enumerate(expected_features, 1):
        if feat in complete_df.columns:
            null_count = complete_df[feat].isna().sum()
            null_pct = (null_count / len(complete_df)) * 100
            dtype = str(complete_df[feat].dtype)

            null_summary.append({
                'feature': feat,
                'null_count': null_count,
                'null_pct': null_pct,
                'dtype': dtype,
                'status': 'available'
            })

            # Color code based on NULL percentage
            status_icon = "✅" if null_count == 0 else ("⚠️" if null_pct < 50 else "❌")

            print(f"{status_icon} {i:<3} {feat:<40} {null_count:<12} {null_pct:<9.1f}% {dtype:<10}")
        else:
            null_summary.append({
                'feature': feat,
                'null_count': len(complete_df),
                'null_pct': 100.0,
                'dtype': 'MISSING',
                'status': 'missing'
            })
            print(f"❌ {i:<3} {feat:<40} {'MISSING':<12} {'100.0%':<10} {'N/A':<10}")

    # Step 5: Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    features_with_no_nulls = sum(1 for f in null_summary if f['null_count'] == 0 and f['status'] == 'available')
    features_with_some_nulls = sum(1 for f in null_summary if 0 < f['null_count'] < len(complete_df) and f['status'] == 'available')
    features_all_nulls = sum(1 for f in null_summary if f['null_count'] == len(complete_df) and f['status'] == 'available')
    features_missing = sum(1 for f in null_summary if f['status'] == 'missing')

    print(f"Total expected features: {len(expected_features)}")
    print(f"  ✅ Complete (0 NULLs): {features_with_no_nulls}")
    print(f"  ⚠️  Partial NULLs: {features_with_some_nulls}")
    print(f"  ❌ All NULLs: {features_all_nulls}")
    print(f"  ❌ Missing columns: {features_missing}")

    # Step 6: Check if this would cause TabNet to fail
    print(f"\n" + "=" * 80)
    print("PREDICTION READINESS")
    print("=" * 80)

    usable_features = [f for f in null_summary if f['status'] == 'available' and f['null_count'] < len(complete_df)]
    feature_match_ratio = len(usable_features) / len(expected_features)

    print(f"Usable features: {len(usable_features)}/{len(expected_features)} ({feature_match_ratio*100:.1f}%)")

    if feature_match_ratio < 0.7:
        print(f"⚠️  WARNING: Feature match ratio {feature_match_ratio:.2f} < 0.70")
        print(f"   TabNet prediction will return NULL!")
    else:
        print(f"✅ Feature match ratio sufficient for prediction")

    # Step 7: Show sample data for debugging
    print(f"\n" + "=" * 80)
    print("SAMPLE FEATURE VALUES (First Horse)")
    print("=" * 80)

    if len(complete_df) > 0:
        for feat in expected_features[:10]:  # Show first 10 features
            if feat in complete_df.columns:
                value = complete_df[feat].iloc[0]
                print(f"  {feat}: {value}")

    # Step 8: Analyze feature completeness and quality
    print(f"\n" + "=" * 80)
    print("FEATURE COMPLETENESS ANALYSIS")
    print("=" * 80)

    # Check for missing musique features
    musique_features = [f for f in expected_features if 'che_' in f or 'joc_' in f]
    available_musique = [f for f in musique_features if f in complete_df.columns]
    missing_musique = [f for f in musique_features if f not in complete_df.columns]

    # Check for missing Phase 1 features
    phase1_features = ['career_strike_rate', 'earnings_per_race', 'earnings_trend',
                       'last_race_position_normalized', 'last_race_odds_normalized',
                       'last_race_field_size_factor', 'distance_consistency',
                       'vha_normalized', 'claiming_tax_trend', 'class_stability']
    available_phase1 = [f for f in phase1_features if f in expected_features and f in complete_df.columns]
    missing_phase1 = [f for f in phase1_features if f in expected_features and f not in complete_df.columns]

    # Check for missing Phase 2 features
    phase2_features = ['class_drop_pct', 'purse_ratio', 'moving_up_in_class',
                       'speed_figure_proxy', 'market_confidence_shift', 'trainer_change',
                       'distance_comfort', 'recence_x_class_drop']
    available_phase2 = [f for f in phase2_features if f in expected_features and f in complete_df.columns]
    missing_phase2 = [f for f in phase2_features if f in expected_features and f not in complete_df.columns]

    # Basic features
    basic_features = [f for f in expected_features if f not in musique_features + phase1_features + phase2_features]
    available_basic = [f for f in basic_features if f in complete_df.columns]
    missing_basic = [f for f in basic_features if f not in complete_df.columns]

    # Print category completeness
    print(f"\n1. Musique Features (Cheval/Jockey performance metrics):")
    print(f"   Available: {len(available_musique)}/{len(musique_features)} ({len(available_musique)/len(musique_features)*100:.1f}%)")
    if missing_musique:
        print(f"   Missing: {missing_musique[:3]}{'...' if len(missing_musique) > 3 else ''}")

    print(f"\n2. Phase 1 Features (Advanced derived features):")
    print(f"   Available: {len(available_phase1)}/{len(phase1_features)} ({len(available_phase1)/len(phase1_features)*100:.1f}%)")
    if missing_phase1:
        print(f"   Missing: {missing_phase1[:3]}{'...' if len(missing_phase1) > 3 else ''}")

    print(f"\n3. Phase 2 Features (Class/Market dynamics):")
    print(f"   Available: {len(available_phase2)}/{len(phase2_features)} ({len(available_phase2)/len(phase2_features)*100:.1f}%)")
    if missing_phase2:
        print(f"   Missing: {missing_phase2[:3]}{'...' if len(missing_phase2) > 3 else ''}")

    print(f"\n4. Basic Features (Core racing metrics):")
    print(f"   Available: {len(available_basic)}/{len(basic_features)} ({len(available_basic)/len(basic_features)*100:.1f}%)")
    if missing_basic:
        print(f"   Missing: {missing_basic[:3]}{'...' if len(missing_basic) > 3 else ''}")

    # Step 9: Feature value distribution analysis
    print(f"\n" + "=" * 80)
    print("FEATURE VALUE DISTRIBUTION ANALYSIS")
    print("=" * 80)

    # Analyze value distributions for key features
    zero_value_features = []
    constant_value_features = []
    extreme_value_features = []

    for feat in available_features:
        if feat in complete_df.columns:
            values = complete_df[feat]

            # Check for all zeros
            if (values == 0).all():
                zero_value_features.append(feat)
            # Check for constant values
            elif values.nunique() == 1:
                constant_value_features.append(feat)
            # Check for extreme values (>1000 or <-1000)
            elif pd.api.types.is_numeric_dtype(values):
                if (values.abs() > 1000).any():
                    max_val = values.max()
                    min_val = values.min()
                    extreme_value_features.append(f"{feat} (range: {min_val:.0f} to {max_val:.0f})")

    if zero_value_features:
        print(f"\n⚠️  Features with all zero values ({len(zero_value_features)}):")
        for feat in zero_value_features[:5]:
            print(f"   - {feat}")
        if len(zero_value_features) > 5:
            print(f"   ... and {len(zero_value_features) - 5} more")

    if constant_value_features:
        print(f"\n⚠️  Features with constant values ({len(constant_value_features)}):")
        for feat in constant_value_features[:5]:
            print(f"   - {feat} = {complete_df[feat].iloc[0]}")
        if len(constant_value_features) > 5:
            print(f"   ... and {len(constant_value_features) - 5} more")

    if extreme_value_features:
        print(f"\n⚠️  Features with extreme values ({len(extreme_value_features)}):")
        for feat_info in extreme_value_features[:5]:
            print(f"   - {feat_info}")
        if len(extreme_value_features) > 5:
            print(f"   ... and {len(extreme_value_features) - 5} more")

    if not (zero_value_features or constant_value_features or extreme_value_features):
        print("\n✅ All features have reasonable value distributions")

    # Step 10: Feature correlation analysis (for musique features)
    print(f"\n" + "=" * 80)
    print("MUSIQUE FEATURE CORRELATION (Sample)")
    print("=" * 80)

    if available_musique:
        print(f"\nAnalyzing correlation between cheval and jockey musique features...")

        # Compare che_ and joc_ features
        che_avg_pos = complete_df['che_global_avg_pos'].mean() if 'che_global_avg_pos' in complete_df.columns else None
        joc_avg_pos = complete_df['joc_global_avg_pos'].mean() if 'joc_global_avg_pos' in complete_df.columns else None

        che_recent = complete_df['che_global_recent_perf'].mean() if 'che_global_recent_perf' in complete_df.columns else None
        joc_recent = complete_df['joc_global_recent_perf'].mean() if 'joc_global_recent_perf' in complete_df.columns else None

        if che_avg_pos is not None and joc_avg_pos is not None:
            print(f"  Avg position - Cheval: {che_avg_pos:.2f}, Jockey: {joc_avg_pos:.2f}")
        if che_recent is not None and joc_recent is not None:
            print(f"  Recent perf - Cheval: {che_recent:.2f}, Jockey: {joc_recent:.2f}")

    # Final summary
    print(f"\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    total_issues = len(zero_value_features) + len(constant_value_features) + len(extreme_value_features)
    total_missing = len(missing_musique) + len(missing_phase1) + len(missing_phase2) + len(missing_basic)

    if total_missing == 0 and total_issues == 0:
        print("✅ EXCELLENT: All features present with good value distributions")
        print("✅ TabNet prediction should work perfectly")
    elif total_missing == 0:
        print(f"✅ GOOD: All features present")
        print(f"⚠️  WARNING: {total_issues} features have distribution issues (may reduce accuracy)")
        print("✅ TabNet prediction will work but may have reduced accuracy")
    else:
        print(f"⚠️  WARNING: {total_missing} features missing")
        print(f"⚠️  WARNING: {total_issues} features have distribution issues")

        if feature_match_ratio >= 0.7:
            print("⚠️  TabNet prediction will work but accuracy may be reduced")
        else:
            print("❌ TabNet prediction will fail (need ≥70% feature match)")

    print("\n" + "=" * 80)

    # Step 11: Save detailed feature report to CSV
    print(f"\n" + "=" * 80)
    print("SAVING FEATURE REPORT")
    print("=" * 80)

    # Create comprehensive feature report
    feature_report = []

    for i, feat in enumerate(expected_features, 1):
        if feat in complete_df.columns:
            values = complete_df[feat]

            # Calculate statistics
            null_count = values.isna().sum()
            null_pct = (null_count / len(complete_df)) * 100
            dtype = str(values.dtype)

            # Value distribution analysis
            is_all_zero = (values == 0).all()
            is_constant = values.nunique() == 1
            has_extreme = False
            value_range = ""

            if pd.api.types.is_numeric_dtype(values):
                min_val = values.min()
                max_val = values.max()
                mean_val = values.mean()
                std_val = values.std()

                if (values.abs() > 1000).any():
                    has_extreme = True
                    value_range = f"{min_val:.0f} to {max_val:.0f}"
                else:
                    value_range = f"{min_val:.2f} to {max_val:.2f}"
            else:
                min_val = max_val = mean_val = std_val = None
                value_range = "N/A"

            # Determine feature category
            if 'che_' in feat or 'joc_' in feat:
                category = "Musique"
            elif feat in phase1_features:
                category = "Phase1"
            elif feat in phase2_features:
                category = "Phase2"
            else:
                category = "Basic"

            # Quality assessment
            quality_issues = []
            if is_all_zero:
                quality_issues.append("All zeros")
            if is_constant:
                quality_issues.append("Constant value")
            if has_extreme:
                quality_issues.append("Extreme values")

            quality_status = "OK" if not quality_issues else "; ".join(quality_issues)

            feature_report.append({
                'rank': i,
                'feature_name': feat,
                'category': category,
                'status': 'Available',
                'dtype': dtype,
                'null_count': null_count,
                'null_pct': f"{null_pct:.1f}%",
                'min': min_val if min_val is not None else "N/A",
                'max': max_val if max_val is not None else "N/A",
                'mean': f"{mean_val:.2f}" if mean_val is not None else "N/A",
                'std': f"{std_val:.2f}" if std_val is not None else "N/A",
                'value_range': value_range,
                'unique_values': values.nunique(),
                'quality_status': quality_status
            })
        else:
            # Feature is missing
            feature_report.append({
                'rank': i,
                'feature_name': feat,
                'category': "Unknown",
                'status': 'Missing',
                'dtype': 'N/A',
                'null_count': len(complete_df),
                'null_pct': '100.0%',
                'min': 'N/A',
                'max': 'N/A',
                'mean': 'N/A',
                'std': 'N/A',
                'value_range': 'N/A',
                'unique_values': 0,
                'quality_status': 'Missing'
            })

    # Convert to DataFrame
    report_df = pd.DataFrame(feature_report)

    # Save to CSV
    output_path = Path("analysis_output")
    output_path.mkdir(exist_ok=True)

    csv_file = output_path / "tabnet_feature_report.csv"
    report_df.to_csv(csv_file, index=False)

    print(f"\n✅ Feature report saved to: {csv_file}")
    print(f"   Rows: {len(report_df)}")
    print(f"   Columns: {len(report_df.columns)}")

    # Also create a summary report
    summary_report = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_expected_features': len(expected_features),
        'available_features': len(available_features),
        'missing_features': len(missing_features),
        'feature_match_ratio': f"{feature_match_ratio:.2%}",
        'zero_value_features': len(zero_value_features),
        'constant_value_features': len(constant_value_features),
        'extreme_value_features': len(extreme_value_features),
        'total_quality_issues': total_issues,
        'musique_available': f"{len(available_musique)}/{len(musique_features)}",
        'phase1_available': f"{len(available_phase1)}/{len(phase1_features)}",
        'phase2_available': f"{len(available_phase2)}/{len(phase2_features)}",
        'basic_available': f"{len(available_basic)}/{len(basic_features)}",
        'prediction_readiness': 'READY' if feature_match_ratio >= 0.7 else 'NOT READY',
        'quality_rating': 'EXCELLENT' if total_issues == 0 else 'GOOD' if total_issues < 10 else 'FAIR'
    }

    summary_df = pd.DataFrame([summary_report])
    summary_csv = output_path / "tabnet_feature_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"✅ Summary report saved to: {summary_csv}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    from datetime import datetime
    analyze_tabnet_features()