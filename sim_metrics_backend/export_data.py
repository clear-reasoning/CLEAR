import mysql.connector
from mysql.connector import errorcode
from database import DB_NAME, connect

# export data as csv file statement
DATA_EXPORT_STATEMENT = \
    "SELECT * " \
    "FROM {table} " \
    "WHERE 1=1 " \
    "AND source_id=\'{source_id}\' " \
    "INTO OUTFILE \'{output_path}\' "  \
    "FIELDS TERMINATED BY \',\' " \
    "LINES TERMINATED BY \'\\n\' "

DATA_EXPORT_STATEMENT_NO_SOURCE_ID = \
    "SELECT * " \
    "FROM {table} " \
    "WHERE 1=1 " \
    "INTO OUTFILE \'{output_path}\' "  \
    "FIELDS TERMINATED BY \',\' " \
    "LINES TERMINATED BY \'\\n\' "


def get_data(cnx, table, output_path, source_id=None):
    """Export data from specified table and source_id, Make sure output_path file does not already exist."""
    try:
        cursor = cnx.cursor()
        if source_id:
            cursor.execute(DATA_EXPORT_STATEMENT.format(table=table,
                                                        output_path=output_path,
                                                        source_id=source_id,
                                                        ))
        else:
            cursor.execute(DATA_EXPORT_STATEMENT_NO_SOURCE_ID.format(table=table,
                                                                     output_path=output_path,
                                                                     ))
    except mysql.connector.Error as err:
        print(err.msg)
    else:
        print("export data into {} successfully.".format(output_path))
    finally:
        cursor.close()


if __name__ == '__main__':

    cnx = connect(DB_NAME)

    # get_data(cnx, 'fact_vehicle_trace',
    #                'flow_ffc9edc5f72b428a986cea83fe90fd0b',
    #                '/var/lib/mysql-files/test.csv')

    get_data(cnx, 'fact_vehicle_trace',
             '/var/lib/mysql-files/test.csv',
             'flow_ffc9edc5f72b428a986cea83fe90fd0b')

    get_data(cnx, 'fact_safety_metrics_binned',
             '/var/lib/mysql-files/test2.csv')

    cnx.close()
