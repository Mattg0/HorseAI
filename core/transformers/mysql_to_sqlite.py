import pandas as pd

from core.connectors.mysql_connector import  connect_to_mysql,execute_query
import mysql.connector
import sqlite3
import datetime
import json
from collections import defaultdict
from decimal import Decimal
from core.calculators.static_feature_calculator import FeatureCalculator

def fetch_mysql_race_data(conn):
    mysql_query = """
    SELECT caractrap.id, caractrap.jour,caractrap.quinte, caractrap.hippo, caractrap.meteo, caractrap.dist,
           caractrap.corde, caractrap.natpis, caractrap.pistegp, caractrap.arriv,caractrap.typec,
           caractrap.temperature, caractrap.forceVent, caractrap.directionVent,
           caractrap.nebulositeLibelleCourt, cachedate.idche, cachedate.cheval,cachedate.cotedirect,cachedate.coteprob,
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


def transform_data(df):
    """Transform the fetched data into a structured format."""

    features = FeatureCalculator.calculate_all_features(df)



    return df

def format_resulats(data,column):
    """Transform the fetched data into a structured format."""
    course_results = defaultdict(list)

    # Process each row of data
    for row in data:
        comp = int(row[0])  # ID de la course
        cl = int(row[1])  # Ordre d'arrivée
        numero = int(row[2])  # Numéro du cheval
        idche = int(row[3])  # ID du cheval

        # Ajouter le résultat à la liste pour cette course
        course_results[comp].append({
            'narrivee': cl,
            'cheval': numero,
            'idche': idche
        })

    # Sérialiser les résultats pour chaque course
    serialized_results = []
    for comp, results in course_results.items():
        def transform_narrivee(value):
            try:
                return int(value)  # Essayer de convertir en entier
            except (ValueError, TypeError):  # En cas d'erreur, retourner 99
                return 99

        # Appliquer la transformation à chaque résultat
        for result in results:
            result['narrivee'] = transform_narrivee(result['narrivee'])

        # Trier les résultats par ordre d'arrivée (cl)
        results.sort(key=lambda x: (x['narrivee'] != 99, x['narrivee']))

        serialized_results.append({
            'comp': comp,
            'ordre_arrivee': json.dumps(results)  # Sérialiser en JSON
        })

    return serialized_results

def insert_resultats_into_sqlite (sqlite_db,result_data):
    conn = sqlite3.connect(sqlite_db)
    cursor = conn.cursor()

    for result in result_data:
        comp = result['comp']
        ordre_arrivee = result['ordre_arrivee']
        created_at = datetime.now()  # Utiliser la date et l'heure actuelles

        cursor.execute("""
            INSERT INTO resultats (comp, ordre_arrivee, created_at)
            VALUES (?, ?, ?)
        """, (comp, ordre_arrivee, created_at))

    conn.commit()
    conn.close()
def insert_data_into_sqlite(sqlite_db, course_data):
    """Insert transformed data into SQLite database."""
    conn = sqlite3.connect(sqlite_db)
    cursor = conn.cursor()

    # Prepare the insert statement
    insert_query = '''
    INSERT INTO Course (comp,jour, hippodrome, meteo, dist, corde, natpis, pistegp,typec, temperature, forceVent, directionVent, nebulosite, participants)
    VALUES (?, ?,?,?, ?,?, ?, ?, ?, ?, ?, ?, ?, ?)
    '''

    for course_key,course in course_data.items():
        course_info = course['course_info']
        participants_json = json.dumps(course['participants'], default=convert_decimal)  # Serialize participants to JSON

        # Remplacer les valeurs vides par 'N/A'
        comp = course_info['comp']
        jour = course_info['jour']
        hippo = course_info['hippo'] if course_info['hippo'] else 'N/A'
        meteo = course_info['meteo'] if course_info['meteo'] else 'N/A'
        dist = course_info['dist'] if course_info['dist'] else 'N/A'
        corde = course_info['corde'] if course_info['corde'] else 'N/A'
        natpis = course_info['natpis'] if course_info['natpis'] else 'N/A'
        pistegp = course_info['pistegp'] if course_info['pistegp'] else 'N/A'
        typec = course_info['typec'] if course_info['typec'] else 'N/A'
        temperature = course_info['temperature'] if course_info['temperature'] else 'N/A'
        forceVent = course_info['forceVent'] if course_info['forceVent'] else 'N/A'
        directionVent = course_info['directionVent'] if course_info['directionVent'] else 'N/A'
        nebulosite = course_info['nebulosite'] if course_info['nebulosite'] else 'N/A'

        cursor.execute(insert_query, (
            comp,
            jour,# Insert comp_id here
            hippo,
            meteo,
            dist,
            corde,
            natpis,
            pistegp,
            typec,
            temperature,
            forceVent,
            directionVent,
            nebulosite,
            participants_json
        ))

    conn.commit()
    conn.close()
    print("Data inserted into SQLite successfully.")


def main(dbname=None):
    mysql = connect_to_mysql(dbname)
    df_raw_data = fetch_mysql_race_data(mysql)
    df_course_data = transform_data(df_raw_data)

         # Insert data into SQLite
        #insert_data_into_sqlite(course_data)


if __name__ == "__main__":
    main()