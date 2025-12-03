#!/usr/bin/env python3
import pandas as pd
import sqlite3
from pathlib import Path

def verify_no_within_race_variance(df: pd.DataFrame, race_id_col='comp'):
    failed = []
    for race_id in df[race_id_col].unique():
        race_df = df[df[race_id_col] == race_id]

        for col in ['victoirescheval', 'placescheval', 'coursescheval', 'gainsCarriere']:
            if col in race_df.columns:
                unique_vals = race_df[col].nunique()
                if unique_vals > 1:
                    failed.append(f"{col} in race {race_id}")
                    print(f"⚠️  {col} varies in race {race_id}: {race_df[col].unique()}")

    if failed:
        print(f"\n❌ FAILED: {len(failed)} features vary within races")
        return False
    else:
        print("\n✅ PASSED: No within-race variance")
        return True

def verify_temporal_vs_raw(db_path: str):
    from core.calculators.temporal_feature_calculator import TemporalFeatureCalculator

    calc = TemporalFeatureCalculator(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT jour FROM historical_races ORDER BY jour DESC LIMIT 5")
    dates = [r[0] for r in cursor.fetchall()]
    conn.close()

    for date in dates:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT participants FROM historical_races WHERE jour = ? LIMIT 1
        """, (date,))
        row = cursor.fetchone()
        conn.close()

        if row:
            import json
            participants = json.loads(row[0])
            for p in participants[:3]:
                horse_id = p.get('idche')
                raw_victoires = p.get('victoirescheval', 0)

                temporal = calc.calculate_historical_career_stats(horse_id, date)

                if temporal['victoirescheval'] >= raw_victoires:
                    print(f"⚠️  Horse {horse_id}: temporal={temporal['victoirescheval']} >= raw={raw_victoires}")

    print("\n✅ Temporal calculation complete")

if __name__ == "__main__":
    print("="*60)
    print("VERIFYING LEAKAGE FIX")
    print("="*60)

    if Path('X_predict_feature.json').exists():
        import json
        with open('X_predict_feature.json') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        verify_no_within_race_variance(df)

    db_path = 'data/hippique2.db'
    if Path(db_path).exists():
        verify_temporal_vs_raw(db_path)
