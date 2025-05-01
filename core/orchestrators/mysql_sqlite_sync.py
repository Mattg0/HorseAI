from core.connectors.mysql_historical_fetcher import fetch_race_data
from core.transformers.historical_race_transformer import transform_race_data, transform_results
from core.connectors.sqlite_historical_writer import write_races_to_sqlite, write_results_to_sqlite
from utils.env_setup import AppConfig

from core.connectors.mysql_connector import connect_to_mysql


def sync_data(mysql_dbname=None):
    """
    Main function to synchronize data from MySQL to SQLite.

    Args:
        mysql_dbname: Optional name of MySQL database
    """
    config = AppConfig()
    # Connect to data sources
    mysql_conn = connect_to_mysql(mysql_dbname)
    sqlite_db_path = config.get_active_db_path()


    # Fetch data
    df_raw_data = fetch_race_data(mysql_conn)

    # Transform race data
    course_infos, participants = transform_race_data(df_raw_data)

    # Transform results data
    results = transform_results(df_raw_data)

    # Write data to SQLite
    write_races_to_sqlite(sqlite_db_path, course_infos, participants)
    write_results_to_sqlite(sqlite_db_path, results)

    # Clean up
    mysql_conn.close()
    print("Data synchronization completed successfully.")


if __name__ == "__main__":
    sync_data('pturf2025')