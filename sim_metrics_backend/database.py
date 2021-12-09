import mysql.connector
from mysql.connector import Error
from mysql.connector import errorcode

# Specify the database
DB_NAME = "circles"


def connect(database=DB_NAME, user='circles.user', host='circles.banatao.berkeley.edu'):
    """Create a connection to a database.

    Parameters
    ----------
    database: str
        The name of the database.

    Returns
    -------
    cnx: MySQLConnection
        A connection object to mysql.
    """
    cnx = None
    try:
        cnx = mysql.connector.connect(
            user=user,
            database=database,
            host=host,
            allow_local_infile=True,
            ssl_ca='./ssl-cert/ca.pem',
            ssl_cert='./ssl-cert/client-cert.pem',
            ssl_key='./ssl-cert/client-key.pem'
        )
        if cnx.is_connected():
            print("Connected to {} database".format(database))
    except Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Access denied: Incorrect user name or password.")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            raise err
    return cnx


def get_network(cnx, source_id):
    """Get network type for a specific simulation.

    Parameters
    ----------
    cnx: MySQLConnection
        A connection object to mysql.
    source_id: str
        The source id of the simulation.

    Returns
    -------
    network: str
        The network type of the simulation.
    """
    cursor = cnx.cursor()
    network = None
    try:
        cursor.execute("SELECT network FROM metadata_table WHERE source_id = \'{}\';".format(source_id))
    except mysql.connector.Error as err:
        print(err)
    else:
        rows = cursor.fetchall()
        network = rows[0][0]
    finally:
        cursor.close()
    return network

# Examples
if __name__ == '__main__':

    cnx = connect(DB_NAME)

    print(get_network(cnx, 'flow_ffc9edc5f72b428a986cea83fe90fd0b'))

    cnx.close()
