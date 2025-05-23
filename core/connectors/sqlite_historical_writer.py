# data_writers/sqlite_writer.py
import sqlite3
import json
from datetime import datetime


def write_races_to_sqlite(sqlite_db, course_data, participants_data):
    """
    Write race data to SQLite database.

    Args:
        sqlite_db: Path to SQLite database
        course_data: DataFrame with course information
        participants_data: DataFrame with participant information
    """
    conn = sqlite3.connect(sqlite_db)
    cursor = conn.cursor()

    # Prepare the insert statement
    insert_query = '''
    INSERT OR REPLACE INTO historical_races 
    (comp, jour, reunion, prix, quinte, hippo, meteo, dist, corde, natpis, 
     pistegp, typec, partant, temperature, forceVent, directionVent, 
     nebulosite, participants, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    '''

    # Group participants by race ID
    grouped_participants = participants_data.groupby('id')

    # Iterate through each course
    for index, course_row in course_data.iterrows():
        comp = course_row['id']  # Race identifier

        # Get participants for this race
        if comp in grouped_participants.groups:
            course_participants = grouped_participants.get_group(comp)
            course_participants = course_participants.drop(columns=['id'])
            # Convert participants to JSON
            from core.transformers.historical_race_transformer import convert_decimal
            participants_json = json.dumps(
                course_participants.to_dict('records'),
                default=convert_decimal,
                ensure_ascii=False
            )
        else:
            participants_json = json.dumps([])

        # Get values with defaults
        cursor.execute(insert_query, (
            comp,
            course_row.get('jour', 'N/A'),
            course_row.get('reun', 'N/A'),
            course_row.get('prix', 'N/A'),
            bool(course_row.get('quinte')),
            course_row.get('hippo', 'N/A'),
            course_row.get('meteo', 'N/A'),
            course_row.get('dist', 'N/A'),
            course_row.get('corde', 'N/A'),
            course_row.get('natpis', 'N/A'),
            course_row.get('pistegp', 'N/A'),
            course_row.get('typec', 'N/A'),
            course_row.get('partant', 'N/A'),
            course_row.get('temperature', 'N/A'),
            course_row.get('forceVent', 'N/A'),
            course_row.get('directionVent', 'N/A'),
            course_row.get('nebulositeLibelleCourt', 'N/A'),
            participants_json,
            datetime.now()
        ))

    conn.commit()
    conn.close()
    print("Race data inserted into SQLite successfully.")


def write_results_to_sqlite(sqlite_db, result_data):
    """
    Write race results to SQLite database.

    Args:
        sqlite_db: Path to SQLite database
        result_data: DataFrame with race results
    """
    conn = sqlite3.connect(sqlite_db)
    cursor = conn.cursor()

    for index, result in result_data.iterrows():
        comp = result['comp']
        ordre_arrivee = result['ordre_arrivee']
        created_at = datetime.now()

        cursor.execute("""
            INSERT OR REPLACE INTO race_results (comp, ordre_arrivee, created_at)
            VALUES (?, ?, ?)
        """, (comp, ordre_arrivee, created_at))

    conn.commit()
    conn.close()
    print("Results data inserted into SQLite successfully.")