from query import QueryStrings
from query import prerequisites
import mysql.connector
from mysql.connector import errorcode
from database import DB_NAME, connect

if __name__ == '__main__':

    cnx = connect(DB_NAME)

    for query_name in prerequisites.keys():
        cursor = cnx.cursor(buffered=True)
        query_statement = QueryStrings[query_name].value.format(
            partition='flow_ffc9edc5f72b428a986cea83fe90fd0b',
            inflow_filter='source_id IS NOT NULL',
            outflow_filter='source_id IS NOT NULL',
            start_filter=0,
            max_decel=-1.0,
            leader_max_decel=-2.0)
        try:
            cursor.execute(query_statement)
        except mysql.connector.Error as err:
            print("When executing query {}:".format(query_name))
            print(err.msg)
        else:
            print("Query {} executes successfully.".format(query_name))
        finally:
            cursor.close()
    cnx.close()
