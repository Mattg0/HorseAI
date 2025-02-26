import mysql.connector
import sqlite3
from utils.env_setup import get_mysql_config

# Load the MYSQL object from env_setup and initialize MySQL connection


def connect_to_mysql(db_name = None):
    dbconfig = get_mysql_config(db_name)
    try:
        conn = mysql.connector.connect(
            host=dbconfig.host,
            user=dbconfig.user,
            password=dbconfig.password,
            database=dbconfig.dbname
        )
        print("Connected to MySQL database.")
        return conn
    except mysql.connector.Error as err:
        print(f"An error occurred while connecting to MySQL: {err}")
        return None
# Connect to the MySQL database using the configuration from config.yaml file

def execute_query(conn, query):
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.commit()
        cursor.close()
        return results
    except mysql.connector.Error as err:
        print(f"An error occurred while executing the query: {err}")
        return None



if __name__ == "__main__":
    mysql_conn = connect_to_mysql()
    query = "SELECT * FROM cachedate LIMIT 0,1;"
    test=execute_query(mysql_conn,query)
    print(test)
    mysql_conn.close()

