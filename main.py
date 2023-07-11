import sys
import database.database_main as database

DEBUG = True

def __main__():
    if "build_database" in sys.argv:
        database.rebuild_database()




