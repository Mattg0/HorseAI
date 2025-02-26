from utils.env_setup import load_config, get_sqlite_dbpath, get_mysql_config
config = load_config()

db_path = get_sqlite_dbpath()
mysql = get_mysql_config('pturf2025')

print(f"Active database path: {mysql.dbname}")