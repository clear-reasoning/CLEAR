
import logging

import luigi

from luigi.contrib import rdbms

logger = logging.getLogger('luigi-interface')

from sim_metrics_backend.query import QueryStrings

from luigi.contrib.mysqldb import MySqlTarget

import datetime

try:
    import mysql.connector
    from mysql.connector import errorcode, Error
except ImportError:
    logger.warning("Loading MySQL module without the python package mysql-connector-python. \
       This will crash at runtime if MySQL functionality is used.")

user='root'
password='404PineApple'
# host='169.229.222.240'
host='localhost'
database = 'circles'


class FACT_VEHICLE_TRACE(luigi.Task):
    target_table = 'fact_energy_trace'
    runtime = datetime.datetime.now()

    def requires(self):
        return []

    def run(self):
        connection = self.output().connect()
        self.output().touch(connection)
        connection.commit()
        connection.close()

    def output(self):
        return MySqlTarget(host=host, database=database, user=user, password=password, table=self.target_table,
                           update_id=str(self.runtime))



class MIDSIZE_SUV_FIT_DENOISED_ACCEL(luigi.Task):
    target_table = 'fact_energy_trace'
    runtime = datetime.datetime.now()
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    partition_name = date
    query = QueryStrings.MIDSIZE_SEDAN_FIT_DENOISED_ACCEL.value
    # user='root'
    # password='404PineApple'
    # host='localhost'
    # database = 'circles'


    def output(self):
        return MySqlTarget(host=host, database=database, user=user, password=password, table=self.target_table,
                           update_id=str(self.runtime))
    def requires(self):
        return [FACT_VEHICLE_TRACE()]
    def run(self):
        connection = self.output().connect()
        for attempt in range(2):
            try:
                cursor = connection.cursor()
                print("caling init copy...")
                cursor.execute(self.query.format(date = self.date, partition = self.partition_name))
            except Error as err:
                if err.errno == errorcode.ER_NO_SUCH_TABLE and attempt == 0:
                    # if first attempt fails with "relation not found", try creating table
                    # logger.info("Creating table %s", self.table)
                    connection.reconnect()
                else:
                    raise
        self.output().touch(connection)
        connection.commit()
        connection.close()


if __name__ == '__main__':
    luigi_run_result = luigi.build([MIDSIZE_SUV_FIT_DENOISED_ACCEL()], detailed_summary=True, no_lock=False, local_scheduler=True)
    print(luigi_run_result.summary_text)




# aa = MIDSIZE_SUV_FIT_DENOISED_ACCEL()
