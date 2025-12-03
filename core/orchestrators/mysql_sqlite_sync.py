from core.connectors.mysql_historical_fetcher import fetch_race_data
from core.transformers.historical_race_transformer import transform_race_data, transform_results
from core.connectors.sqlite_historical_writer import write_races_to_sqlite, write_results_to_sqlite
from utils.env_setup import AppConfig

from core.connectors.mysql_connector import connect_to_mysql


def sync_data(mysql_dbname=None, quinte=False):
    """
    Main function to synchronize data from MySQL to SQLite.

    Args:
        mysql_dbname: Optional name of MySQL database
        quinte: If True, only sync quinte races to historical_quinte table
    """
    config = AppConfig()
    # Connect to data sources
    mysql_conn = connect_to_mysql(mysql_dbname)
    sqlite_db_path = config.get_active_db_path()

    # Determine target tables
    races_table = 'historical_quinte' if quinte else 'historical_races'
    results_table = 'quinte_results' if quinte else 'race_results'

    # Fetch data
    df_raw_data = fetch_race_data(mysql_conn, quinte=quinte, target_table=races_table)

    # Transform race data
    course_infos, participants = transform_race_data(df_raw_data)

    # Transform results data
    results = transform_results(df_raw_data)

    # Write data to SQLite
    write_races_to_sqlite(sqlite_db_path, course_infos, participants, target_table=races_table)
    write_results_to_sqlite(sqlite_db_path, results, target_table=results_table)

    # Clean up
    mysql_conn.close()
    sync_type = "Quinte races" if quinte else "All races"
    print(f"Data synchronization completed successfully. ({sync_type} -> {races_table} & {results_table})")


if __name__ == "__main__":
    sync_data('pturf2025')