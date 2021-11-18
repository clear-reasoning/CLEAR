import mysql.connector
from mysql.connector import Error
from mysql.connector import errorcode

# Specify the database
DB_NAME = "circles"


def connect(database=DB_NAME, user='root', host='localhost'):
    try:
        cnx = mysql.connector.connect(
            user=user,
            password='404PineApple',
            database=database,
            host=host,
            allow_local_infile=True
        )
        if cnx.is_connected():
            print("Connected to {} database".format(database))
    except Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Access denied: Incorrect user name or password.")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
    return cnx
