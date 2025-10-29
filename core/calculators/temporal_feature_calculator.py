import sqlite3
import json
from typing import Dict, List
import pandas as pd

class TemporalFeatureCalculator:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.cache = {}
        self.call_count = 0

    def batch_calculate_all_horses(self, df: pd.DataFrame, preserve_existing: bool = False) -> pd.DataFrame:
        """
        Calculate temporal stats for all horses in batch (MUCH faster than one-by-one).

        Args:
            df: DataFrame with race participant data
            preserve_existing: If True, only calculate stats for rows with missing values.
                              Use True for prediction to preserve current database values.
        """
        result_df = df.copy()

        if 'idche' not in df.columns or 'jour' not in df.columns:
            print("  ⚠️  Missing idche or jour columns, skipping temporal calculations")
            return result_df

        # For prediction: preserve existing stats from database
        if preserve_existing:
            preserve_fields = ['victoirescheval', 'placescheval', 'coursescheval', 'gainsCarriere']
            has_existing = all(
                field in df.columns and df[field].notna().any()
                for field in preserve_fields
            )
            if has_existing:
                print(f"  ✅ Preserving existing career stats from database (prediction mode)")
                return result_df

        unique_horses = df['idche'].unique()
        unique_dates = df['jour'].unique()

        print(f"  📦 Batch loading historical data for {len(unique_horses)} horses...")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all historical races ONCE
        max_date = max(unique_dates)
        cursor.execute("""
            SELECT jour, participants
            FROM historical_races
            WHERE jour < ?
            ORDER BY jour
        """, (max_date,))

        all_races = cursor.fetchall()
        conn.close()

        print(f"  📊 Loaded {len(all_races)} historical races, building horse stats...")

        # Build cumulative stats for each horse by date
        horse_stats_by_date = {}

        for race_date, participants_json in all_races:
            try:
                participants = json.loads(participants_json)
                for horse in participants:
                    horse_id = horse.get('idche')
                    if horse_id not in unique_horses:
                        continue

                    if horse_id not in horse_stats_by_date:
                        horse_stats_by_date[horse_id] = {}

                    # Get previous stats for this horse
                    prev_dates = sorted([d for d in horse_stats_by_date[horse_id].keys() if d < race_date])
                    if prev_dates:
                        prev_stats = horse_stats_by_date[horse_id][prev_dates[-1]]
                        victoires = prev_stats['victoires']
                        places = prev_stats['places']
                        courses = prev_stats['courses']
                        gains = prev_stats['gains']
                    else:
                        victoires = places = courses = 0
                        gains = 0.0

                    # Add this race
                    courses += 1
                    place = horse.get('place', '')

                    if str(place) == '1':
                        victoires += 1
                        places += 1
                    elif str(place).isdigit() and int(place) <= 3:
                        places += 1

                    gain_str = horse.get('gain', '0')
                    try:
                        gains += float(str(gain_str).replace(' ', '').replace(',', ''))
                    except:
                        pass

                    horse_stats_by_date[horse_id][race_date] = {
                        'victoires': victoires,
                        'places': places,
                        'courses': courses,
                        'gains': gains
                    }
            except:
                continue

        print(f"  ✅ Built stats for {len(horse_stats_by_date)} horses, applying to dataframe...")

        # Apply to dataframe
        for idx, row in result_df.iterrows():
            horse_id = row['idche']
            race_date = row['jour']

            stats = {'victoires': 0, 'places': 0, 'courses': 0, 'gains': 0.0}

            if horse_id in horse_stats_by_date:
                # Get stats from before this race date
                available_dates = sorted([d for d in horse_stats_by_date[horse_id].keys() if d < race_date])
                if available_dates:
                    stats = horse_stats_by_date[horse_id][available_dates[-1]]

            ratio_victoires = stats['victoires'] / stats['courses'] if stats['courses'] > 0 else 0.0
            ratio_places = stats['places'] / stats['courses'] if stats['courses'] > 0 else 0.0

            result_df.at[idx, 'victoirescheval'] = stats['victoires']
            result_df.at[idx, 'placescheval'] = stats['places']
            result_df.at[idx, 'coursescheval'] = stats['courses']
            result_df.at[idx, 'gainsCarriere'] = stats['gains']
            result_df.at[idx, 'ratio_victoires'] = ratio_victoires
            result_df.at[idx, 'ratio_places'] = ratio_places

        print(f"  ✅ Temporal calculations complete!")
        return result_df

    def calculate_historical_career_stats(self, horse_id: int, race_date: str) -> Dict[str, float]:
        """Legacy single-horse method (SLOW - use batch_calculate_all_horses instead)"""
        self.call_count += 1
        if self.call_count % 100 == 0:
            print(f"  ⚠️  Using slow method: {self.call_count} calls (use batch_calculate_all_horses!)")

        cache_key = (horse_id, race_date)
        if cache_key in self.cache:
            return self.cache[cache_key]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT participants
            FROM historical_races
            WHERE jour < ?
        """, (race_date,))

        rows = cursor.fetchall()
        conn.close()

        victoires = 0
        places = 0
        courses = 0
        gains = 0.0

        for row in rows:
            try:
                participants = json.loads(row[0])
                for horse in participants:
                    if horse.get('idche') == horse_id:
                        courses += 1
                        place = horse.get('place', '')

                        if str(place) == '1':
                            victoires += 1
                            places += 1
                        elif str(place).isdigit() and int(place) <= 3:
                            places += 1

                        gain_str = horse.get('gain', '0')
                        try:
                            gains += float(str(gain_str).replace(' ', '').replace(',', ''))
                        except:
                            pass
            except:
                continue

        ratio_victoires = victoires / courses if courses > 0 else 0.0
        ratio_places = places / courses if courses > 0 else 0.0

        result = {
            'victoirescheval': victoires,
            'placescheval': places,
            'coursescheval': courses,
            'gainsCarriere': gains,
            'ratio_victoires': ratio_victoires,
            'ratio_places': ratio_places
        }

        self.cache[cache_key] = result
        return result
