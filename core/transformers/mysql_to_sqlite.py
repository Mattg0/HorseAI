import pandas as pd

from core.connectors.mysql_connector import  connect_to_mysql
from utils.env_setup import get_sqlite_dbpath
import mysql.connector
import sqlite3
from datetime import datetime
import json
from collections import defaultdict
from decimal import Decimal
from core.calculators.static_feature_calculator import FeatureCalculator

def fetch_mysql_race_data(conn):
    mysql_query = """
    SELECT caractrap.id, caractrap.jour,caractrap.reun,caractrap.prix,caractrap.partant,caractrap.quinte, caractrap.hippo, caractrap.meteo, caractrap.dist,
           caractrap.corde, caractrap.natpis, caractrap.pistegp,caractrap.typec,
           caractrap.temperature, caractrap.forceVent, caractrap.directionVent,
           caractrap.nebulositeLibelleCourt, cachedate.idche, cachedate.cheval,cachedate.cl,cachedate.cotedirect,cachedate.coteprob,
           cachedate.numero,cachedate.vha,cachedate.poidmont,cachedate.recence,cachedate.gainsAnneeEnCours, cachedate.musiqueche, cachedate.idJockey, musiquejoc, cachedate.idEntraineur, cachedate.age,
           cachedate.nbVictCouple, cachedate.nbPlaceCouple, cachedate.victoirescheval, cachedate.placescheval, cachedate.TxVictCouple,
           cachedate.pourcVictChevalHippo, cachedate.pourcPlaceChevalHippo, cachedate.pourcVictJockHippo, cachedate.pourcPlaceJockHippo,cachedate.coursescheval
    FROM caractrap
    INNER JOIN cachedate ON caractrap.id = cachedate.comp 
    
    """
    cursor = conn.cursor()
    cursor.execute(mysql_query)
    data = cursor.fetchall()
    columns = [column[0] for column in cursor.description]
    df_course_data = pd.DataFrame(data,columns=columns)

    return df_course_data


def convert_decimal(value):
    """Convert Decimal to float or return the value as is."""
    if isinstance(value, Decimal):
        return float(value)
    return value


def transform_race_data(df):
    """Transform the fetched data into a structured format."""
    features = FeatureCalculator.calculate_all_features(df)
    course_infos = df.copy()
    course_infos_columns= ['jour','quinte','hippo','reun','prix','partant','meteo','dist','corde','natpis','pistegp','typec','temperature','forceVent','directionVent','nebulositeLibelleCourt']
    course_infos = df.loc[:, df.columns.isin(course_infos_columns + ['id'])]
    course_infos = course_infos.drop_duplicates(subset=['id'])


    # Drop unwanted columns from features to get participants
    participants = features.drop(columns=[col for col in features.columns if col in course_infos_columns])


    return course_infos, participants
def format_resultats(df_raw_data):
    """Transform the fetched data into a structured format."""
    course_results = defaultdict(list)

    # Process each row of data
    for index,row in df_raw_data.iterrows():
        comp = row.get('id')  # ID de la course
        cl = row.get('cl')
        numero = int(row.get('numero'))   # Numéro du cheval
        idche = int(row.get('idche'))   # ID du cheval

        # Ajouter le résultat à la liste pour cette course
        course_results[comp].append({
            'narrivee': cl,
            'cheval': numero,
            'idche': idche
        })

    # Sérialiser les résultats pour chaque course
    serialized_results = []
    for comp, results in course_results.items():
 # Trier les résultats par ordre d'arrivée (cl)
        results.sort(key=lambda x: (isinstance(x['narrivee'], str), x['narrivee']))

        serialized_results.append({
            'comp': comp,
            'ordre_arrivee': json.dumps(results)  # Sérialiser en JSON
        })

    return serialized_results

def insert_resultats_into_sqlite (sqlite_db,result_data):
    conn = sqlite3.connect(sqlite_db)
    cursor = conn.cursor()

    for index,result in result_data.iterrows():
        comp = result['comp']
        ordre_arrivee = result['ordre_arrivee']
        created_at = datetime.now()  # Utiliser la date et l'heure actuelles

        cursor.execute("""
            INSERT OR REPLACE INTO race_results (comp, ordre_arrivee, created_at)
            VALUES (?, ?, ?)
        """, (comp, ordre_arrivee, created_at))

    conn.commit()
    conn.close()
def insert_data_into_sqlite(sqlite_db, course_data,participants_data):
    """Insert transformed data into SQLite database."""
    conn = sqlite3.connect(sqlite_db)
    cursor = conn.cursor()

    # Prepare the insert statement
    insert_query = '''
    INSERT OR REPLACE INTO historical_races (comp,jour,reunion,prix, quinte,hippo, meteo, dist, corde, natpis, pistegp,typec, partant, temperature, forceVent, directionVent, nebulosite, participants,created_at)
    VALUES (?, ?,?,?,?,?,?, ?,?, ?, ?, ?, ?, ?, ?, ?,?, ?,?)
    '''

    grouped_participants = participants_data.groupby('id')

    # Iterate through each course
    for index, course_row in course_data.iterrows():
        comp = course_row['id']  # Assuming 'id' is the course identifier
        # Get participants for this course
        if comp in grouped_participants.groups:
            course_participants = grouped_participants.get_group(comp)
            course_participants = course_participants.drop(columns=['id'])
            # Convert participants to list of dictionaries and then to JSON
            participants_json = json.dumps(course_participants.to_dict('records'), default=convert_decimal,ensure_ascii=False)
        else:
            participants_json = json.dumps([])

        # Remplacer les valeurs vides par 'N/A'
        jour = course_row.get('jour', 'N/A')
        hippo = course_row.get('hippo', 'N/A')
        reun = course_row.get('reun','N/A')
        prix = course_row.get('prix','N/A')
        quinte = bool(course_row.get('qinte'))
        meteo = course_row.get('meteo', 'N/A')
        dist = course_row.get('dist', 'N/A')
        corde = course_row.get('corde', 'N/A')
        partant = course_row.get('partant','N/A')
        natpis = course_row.get('natpis', 'N/A')
        pistegp = course_row.get('pistegp', 'N/A')
        typec = course_row.get('typec', 'N/A')
        temperature = course_row.get('temperature', 'N/A')
        forceVent = course_row.get('forceVent', 'N/A')
        directionVent = course_row.get('directionVent', 'N/A')
        nebulosite = course_row.get('nebulositeLibelleCourt', 'N/A')
        created_at = datetime.now()
        cursor.execute(insert_query, (
            comp,
            jour,# Insert comp_id here
            reun,
            prix,
            quinte,
            hippo,
            meteo,
            dist,
            corde,
            natpis,
            pistegp,
            typec,
            partant,
            temperature,
            forceVent,
            directionVent,
            nebulosite,
            participants_json,
            created_at
        ))

    conn.commit()
    conn.close()
    print("Data inserted into SQLite successfully.")





def main(dbname=None):
    mysql = connect_to_mysql(dbname)
    sqlite_db_path= get_sqlite_dbpath()
    df_raw_data = fetch_mysql_race_data(mysql)
    course_infos, participants = transform_race_data(df_raw_data)
    insert_data_into_sqlite(sqlite_db_path,course_infos,participants)
    results = pd.DataFrame(format_resultats(df_raw_data))
    insert_resultats_into_sqlite(sqlite_db_path,results)

if __name__ == "__main__":
    main('pturf2025')