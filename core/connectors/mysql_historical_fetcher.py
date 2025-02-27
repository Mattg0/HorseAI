# data_fetchers/mysql_fetcher.py

import pandas as pd
from core.connectors.mysql_connector import connect_to_mysql


def fetch_race_data(conn=None, close_conn=False):
    """
    Fetch race data from MySQL database.

    Args:
        conn: MySQL connection (will create one if None)
        close_conn: Whether to close the connection after fetching

    Returns:
        DataFrame containing race data
    """
    if conn is None:
        conn = connect_to_mysql()
        close_conn = True

    mysql_query = """
    SELECT caractrap.id, caractrap.jour, caractrap.reun, caractrap.prix, caractrap.partant, 
           caractrap.quinte, caractrap.hippo, caractrap.meteo, caractrap.dist,
           caractrap.corde, caractrap.natpis, caractrap.pistegp, caractrap.typec,
           caractrap.temperature, caractrap.forceVent, caractrap.directionVent,
           caractrap.nebulositeLibelleCourt, cachedate.idche, cachedate.cheval, cachedate.cl, 
           cachedate.cotedirect, cachedate.coteprob, cachedate.numero, cachedate.handicapDistance,
           cachedate.handicapPoids, 
           cachedate.poidmont, cachedate.recence, cachedate.gainsAnneeEnCours, 
           cachedate.musiqueche, cachedate.idJockey, musiquejoc, cachedate.idEntraineur,cachedate.proprietaire, 
           cachedate.age, cachedate.nbVictCouple, cachedate.nbPlaceCouple, 
           cachedate.victoirescheval, cachedate.placescheval, cachedate.TxVictCouple,
           cachedate.pourcVictChevalHippo, cachedate.pourcPlaceChevalHippo, 
           cachedate.pourcVictJockHippo, cachedate.pourcPlaceJockHippo, cachedate.coursescheval
    FROM caractrap
    INNER JOIN cachedate ON caractrap.id = cachedate.comp 
    """

    cursor = conn.cursor()
    cursor.execute(mysql_query)
    data = cursor.fetchall()
    columns = [column[0] for column in cursor.description]
    df_course_data = pd.DataFrame(data, columns=columns)

    if close_conn:
        conn.close()

    return df_course_data