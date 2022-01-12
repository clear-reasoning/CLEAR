import datetime
from luigi.contrib.mysqldb import MySqlTarget
from sim_metrics_backend.query import QueryStrings
import sys
from functools import partial
import logging

import luigi

from luigi.contrib import rdbms

logger = logging.getLogger('luigi-interface')

sys.path.append('..')


try:
    import mysql.connector
    from mysql.connector import errorcode, Error
except ImportError:
    logger.warning("Loading MySQL module without the python package mysql-connector-python. \
       This will crash at runtime if MySQL functionality is used.")

user = 'circles.user'
password = '404PineApple'
# host='169.229.222.240'
host = 'circles.banatao.berkeley.edu'
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


class TACOMA_FIT_DENOISED_ACCEL(luigi.Task):

    target_table = 'fact_vehicle_trace'
    runtime = datetime.datetime.now()
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    partition_name = luigi.ListParameter()
    query = QueryStrings.TACOMA_FIT_DENOISED_ACCEL.value

    def output(self):
        return MySqlTarget(host=host, database=database, user=user, password=password,
                           table=self.target_table, update_id=str(self.runtime))

    def requires(self):
        return [FACT_VEHICLE_TRACE()]

    def run(self):
        connection = self.output().connect()
        for cur_partition in self.partition_name:
            try:
                cursor = connection.cursor(buffered=True)
                print("executing query TACOMA_FIT_DENOISED_ACCEL...")
                cursor.execute(self.query.format(partition=cur_partition))
            except Error as err:
                if err.errno == errorcode.ER_NO_SUCH_TABLE:
                    # if first attempt fails with "relation not found", try creating table
                    # logger.info("Creating table %s", self.table)
                    connection.reconnect()
                else:
                    raise

        self.output().touch(connection)
        connection.commit()
        connection.close()


class PRIUS_FIT_DENOISED_ACCEL(luigi.Task):

    target_table = 'fact_vehicle_trace'
    runtime = datetime.datetime.now()
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    partition_name = luigi.ListParameter()
    query = QueryStrings.PRIUS_FIT_DENOISED_ACCEL.value

    def output(self):
        return MySqlTarget(host=host, database=database, user=user, password=password,
                           table=self.target_table, update_id=str(self.runtime))

    def requires(self):
        return [FACT_VEHICLE_TRACE()]

    def run(self):
        connection = self.output().connect()
        for cur_partition in self.partition_name:
            try:
                cursor = connection.cursor(buffered=True)
                print("executing query PRIUS_FIT_DENOISED_ACCEL...")
                cur_query = self.query.format(partition=cur_partition)
                cursor.execute(cur_query)
            except Error as err:
                if err.errno == errorcode.ER_NO_SUCH_TABLE:
                    # if first attempt fails with "relation not found", try creating table
                    # logger.info("Creating table %s", self.table)
                    connection.reconnect()
                else:
                    raise

        self.output().touch(connection)
        connection.commit()
        connection.close()


class MIDSIZE_SUV_FIT_DENOISED_ACCEL(luigi.Task):

    target_table = 'fact_energy_trace'
    runtime = datetime.datetime.now()
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    partition_name = luigi.ListParameter()
    query = QueryStrings.MIDSIZE_SUV_FIT_DENOISED_ACCEL.value

    def output(self):
        return MySqlTarget(host=host, database=database, user=user, password=password, table=self.target_table,
                           update_id=str(self.runtime))

    def requires(self):
        return [FACT_VEHICLE_TRACE()]

    def run(self):
        connection = self.output().connect()
        for cur_partition in self.partition_name:
            try:
                cursor = connection.cursor(buffered=True)
                print("executing query MIDSIZE_SUV_FIT_DENOISED_ACCEL...")
                cursor.execute(self.query.format(partition=cur_partition))
            except Error as err:
                if err.errno == errorcode.ER_NO_SUCH_TABLE:
                    # if first attempt fails with "relation not found", try creating table
                    # logger.info("Creating table %s", self.table)
                    connection.reconnect()
                else:
                    raise
        self.output().touch(connection)
        connection.commit()
        connection.close()


class COMPACT_SEDAN_FIT_DENOISED_ACCEL(luigi.Task):
    target_table = 'fact_energy_trace'
    runtime = datetime.datetime.now()
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    partition_name = luigi.ListParameter()
    query = QueryStrings.COMPACT_SEDAN_FIT_DENOISED_ACCEL.value

    def output(self):
        return MySqlTarget(host=host, database=database, user=user, password=password, table=self.target_table,
                           update_id=str(self.runtime))

    def requires(self):
        return [FACT_VEHICLE_TRACE()]

    def run(self):
        connection = self.output().connect()
        for cur_partition in self.partition_name:
            try:
                cursor = connection.cursor(buffered=True)
                print("executing query COMPACT_SEDAN_FIT_DENOISED_ACCEL...")
                cursor.execute(self.query.format(partition=cur_partition))
            except Error as err:
                if err.errno == errorcode.ER_NO_SUCH_TABLE:
                    # if first attempt fails with "relation not found", try creating table
                    # logger.info("Creating table %s", self.table)
                    connection.reconnect()
                else:
                    raise
        self.output().touch(connection)
        connection.commit()
        connection.close()


class MIDSIZE_SEDAN_FIT_DENOISED_ACCEL(luigi.Task):
    target_table = 'fact_energy_trace'
    runtime = datetime.datetime.now()
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    partition_name = luigi.ListParameter()
    query = QueryStrings.MIDSIZE_SEDAN_FIT_DENOISED_ACCEL.value

    def output(self):
        return MySqlTarget(host=host, database=database, user=user, password=password, table=self.target_table,
                           update_id=str(self.runtime))

    def requires(self):
        return [FACT_VEHICLE_TRACE()]

    def run(self):
        connection = self.output().connect()
        for cur_partition in self.partition_name:
            try:
                cursor = connection.cursor(buffered=True)
                print("executing query MIDSIZE_SEDAN_FIT_DENOISED_ACCEL...")
                cursor.execute(self.query.format(partition=cur_partition))
            except Error as err:
                if err.errno == errorcode.ER_NO_SUCH_TABLE:
                    # if first attempt fails with "relation not found", try creating table
                    # logger.info("Creating table %s", self.table)
                    connection.reconnect()
                else:
                    raise
        self.output().touch(connection)
        connection.commit()
        connection.close()


class RAV4_2019_FIT_DENOISED_ACCEL(luigi.Task):
    target_table = 'fact_energy_trace'
    runtime = datetime.datetime.now()
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    partition_name = luigi.ListParameter()
    query = QueryStrings.RAV4_2019_FIT_DENOISED_ACCEL.value

    def output(self):
        return MySqlTarget(host=host, database=database, user=user, password=password, table=self.target_table,
                           update_id=str(self.runtime))

    def requires(self):
        return [FACT_VEHICLE_TRACE()]

    def run(self):
        connection = self.output().connect()
        for cur_partition in self.partition_name:
            try:
                cursor = connection.cursor(buffered=True)
                print("executing query RAV4_2019_FIT_DENOISED_ACCEL...")
                cursor.execute(self.query.format(partition=cur_partition))
            except Error as err:
                if err.errno == errorcode.ER_NO_SUCH_TABLE:
                    # if first attempt fails with "relation not found", try creating table
                    # logger.info("Creating table %s", self.table)
                    connection.reconnect()
                else:
                    raise
        self.output().touch(connection)
        connection.commit()
        connection.close()


class LIGHT_DUTY_PICKUP_FIT_DENOISED_ACCEL(luigi.Task):
    target_table = 'fact_energy_trace'
    runtime = datetime.datetime.now()
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    partition_name = luigi.ListParameter()
    query = QueryStrings.LIGHT_DUTY_PICKUP_FIT_DENOISED_ACCEL.value

    def output(self):
        return MySqlTarget(host=host, database=database, user=user, password=password, table=self.target_table,
                           update_id=str(self.runtime))

    def requires(self):
        return [FACT_VEHICLE_TRACE()]

    def run(self):
        connection = self.output().connect()
        for cur_partition in self.partition_name:
            try:
                cursor = connection.cursor(buffered=True)
                print("executing query LIGHT_DUTY_PICKUP_FIT_DENOISED_ACCEL...")
                cursor.execute(self.query.format(partition=cur_partition))
            except Error as err:
                if err.errno == errorcode.ER_NO_SUCH_TABLE:
                    # if first attempt fails with "relation not found", try creating table
                    # logger.info("Creating table %s", self.table)
                    connection.reconnect()
                else:
                    raise
        self.output().touch(connection)
        connection.commit()
        connection.close()


class CLASS3_PND_TRUCK_FIT_DENOISED_ACCEL(luigi.Task):
    target_table = 'fact_energy_trace'
    runtime = datetime.datetime.now()
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    partition_name = luigi.ListParameter()
    query = QueryStrings.CLASS3_PND_TRUCK_FIT_DENOISED_ACCEL.value

    def output(self):
        return MySqlTarget(host=host, database=database, user=user, password=password, table=self.target_table,
                           update_id=str(self.runtime))

    def requires(self):
        return [FACT_VEHICLE_TRACE()]

    def run(self):
        connection = self.output().connect()
        for cur_partition in self.partition_name:
            try:
                cursor = connection.cursor(buffered=True)
                print("executing query CLASS3_PND_TRUCK_FIT_DENOISED_ACCEL...")
                cursor.execute(self.query.format(partition=cur_partition))
            except Error as err:
                if err.errno == errorcode.ER_NO_SUCH_TABLE:
                    # if first attempt fails with "relation not found", try creating table
                    # logger.info("Creating table %s", self.table)
                    connection.reconnect()
                else:
                    raise
        self.output().touch(connection)
        connection.commit()
        connection.close()


class CLASS8_TRACTOR_TRAILER_FIT_DENOISED_ACCEL(luigi.Task):
    target_table = 'fact_energy_trace'
    runtime = datetime.datetime.now()
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    partition_name = luigi.ListParameter()
    query = QueryStrings.CLASS8_TRACTOR_TRAILER_FIT_DENOISED_ACCEL.value

    def output(self):
        return MySqlTarget(host=host, database=database, user=user, password=password, table=self.target_table,
                           update_id=str(self.runtime))

    def requires(self):
        return [FACT_VEHICLE_TRACE()]

    def run(self):
        connection = self.output().connect()
        for cur_partition in self.partition_name:
            try:
                cursor = connection.cursor(buffered=True)
                print("executing query CLASS8_TRACTOR_TRAILER_FIT_DENOISED_ACCEL...")
                cursor.execute(self.query.format(partition=cur_partition))
            except Error as err:
                if err.errno == errorcode.ER_NO_SUCH_TABLE:
                    # if first attempt fails with "relation not found", try creating table
                    # logger.info("Creating table %s", self.table)
                    connection.reconnect()
                else:
                    raise
        self.output().touch(connection)
        connection.commit()
        connection.close()


class FACT_INFEASIBLE_FLAGS(luigi.Task):
    target_table = 'fact_infeasible_flags'
    runtime = datetime.datetime.now()
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    partition_name = luigi.ListParameter()
    start_filter = luigi.FloatParameter()
    inflow_filter = luigi.Parameter()
    outflow_filter = luigi.Parameter()
    query = QueryStrings.FACT_INFEASIBLE_FLAGS.value

    def output(self):
        return MySqlTarget(host=host, database=database, user=user, password=password, table=self.target_table,
                           update_id=str(self.runtime))

    def requires(self):
        return [TACOMA_FIT_DENOISED_ACCEL(partition_name=self.partition_name),
                PRIUS_FIT_DENOISED_ACCEL(partition_name=self.partition_name),
                COMPACT_SEDAN_FIT_DENOISED_ACCEL(partition_name=self.partition_name),
                MIDSIZE_SEDAN_FIT_DENOISED_ACCEL(partition_name=self.partition_name),
                RAV4_2019_FIT_DENOISED_ACCEL(partition_name=self.partition_name),
                MIDSIZE_SUV_FIT_DENOISED_ACCEL(partition_name=self.partition_name),
                LIGHT_DUTY_PICKUP_FIT_DENOISED_ACCEL(partition_name=self.partition_name),
                CLASS3_PND_TRUCK_FIT_DENOISED_ACCEL(partition_name=self.partition_name),
                CLASS8_TRACTOR_TRAILER_FIT_DENOISED_ACCEL(partition_name=self.partition_name)]

    def run(self):
        connection = self.output().connect()
        for cur_partition in self.partition_name:
            try:
                cursor = connection.cursor(buffered=True)
                cur_query = self.query.format(partition=cur_partition, start_filter=self.start_filter,
                                              inflow_filter=self.inflow_filter, outflow_filter=self.outflow_filter)
                print("executing query FACT_INFEASIBLE_FLAGS...")
                cursor.execute(cur_query)
            except Error as err:
                if err.errno == errorcode.ER_NO_SUCH_TABLE:
                    # if first attempt fails with "relation not found", try creating table
                    # logger.info("Creating table %s", self.table)
                    connection.reconnect()
                else:
                    raise
        self.output().touch(connection)
        connection.commit()
        connection.close()


class FACT_VEHICLE_FUEL_EFFICIENCY_AGG(luigi.Task):
    target_table = 'fact_vehicle_fuel_efficiency_agg'
    runtime = datetime.datetime.now()
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    partition_name = luigi.ListParameter()
    start_filter = luigi.FloatParameter()
    inflow_filter = luigi.Parameter()
    outflow_filter = luigi.Parameter()
    query = QueryStrings.FACT_VEHICLE_FUEL_EFFICIENCY_AGG.value

    def output(self):
        return MySqlTarget(host=host, database=database, user=user, password=password, table=self.target_table,
                           update_id=str(self.runtime))

    def requires(self):
        return [FACT_VEHICLE_TRACE(),
                TACOMA_FIT_DENOISED_ACCEL(partition_name=self.partition_name),
                PRIUS_FIT_DENOISED_ACCEL(partition_name=self.partition_name),
                COMPACT_SEDAN_FIT_DENOISED_ACCEL(partition_name=self.partition_name),
                MIDSIZE_SEDAN_FIT_DENOISED_ACCEL(partition_name=self.partition_name),
                RAV4_2019_FIT_DENOISED_ACCEL(partition_name=self.partition_name),
                MIDSIZE_SUV_FIT_DENOISED_ACCEL(partition_name=self.partition_name),
                LIGHT_DUTY_PICKUP_FIT_DENOISED_ACCEL(partition_name=self.partition_name),
                CLASS3_PND_TRUCK_FIT_DENOISED_ACCEL(partition_name=self.partition_name),
                CLASS8_TRACTOR_TRAILER_FIT_DENOISED_ACCEL(partition_name=self.partition_name)]

    def run(self):
        connection = self.output().connect()
        for cur_partition in self.partition_name:
            try:
                cursor = connection.cursor(buffered=True)
                cur_query = self.query.format(partition=cur_partition, start_filter=self.start_filter,
                                              inflow_filter=self.inflow_filter, outflow_filter=self.outflow_filter)
                print("executing query FACT_VEHICLE_FUEL_EFFICIENCY_AGG...")
                cursor.execute(cur_query)
            except Error as err:
                if err.errno == errorcode.ER_NO_SUCH_TABLE:
                    # if first attempt fails with "relation not found", try creating table
                    # logger.info("Creating table %s", self.table)
                    connection.reconnect()
                else:
                    raise
        self.output().touch(connection)
        connection.commit()
        connection.close()


class FACT_SAFETY_METRICS_3D(luigi.Task):
    target_table = 'fact_safety_metrics'
    runtime = datetime.datetime.now()
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    partition_name = luigi.ListParameter()
    start_filter = luigi.FloatParameter()
    max_decel = luigi.FloatParameter()
    leader_max_decel = luigi.FloatParameter()
    inflow_filter = luigi.Parameter()
    outflow_filter = luigi.Parameter()
    query = QueryStrings.FACT_SAFETY_METRICS_3D.value

    def output(self):
        return MySqlTarget(host=host, database=database, user=user, password=password, table=self.target_table,
                           update_id=str(self.runtime))

    def requires(self):
        return [FACT_VEHICLE_TRACE()]

    def run(self):
        connection = self.output().connect()
        for cur_partition in self.partition_name:
            try:
                cursor = connection.cursor(buffered=True)
                cur_query = self.query.format(partition=cur_partition,
                                              start_filter=self.start_filter,
                                              max_decel=self.max_decel,
                                              leader_max_decel=self.leader_max_decel,
                                              inflow_filter=self.inflow_filter,
                                              outflow_filter=self.outflow_filter
                                              )
                print("executing query FACT_SAFETY_METRICS_3D...")
                cursor.execute(cur_query)
            except Error as err:
                if err.errno == errorcode.ER_NO_SUCH_TABLE:
                    # if first attempt fails with "relation not found", try creating table
                    # logger.info("Creating table %s", self.table)
                    connection.reconnect()
                else:
                    raise
        self.output().touch(connection)
        connection.commit()
        connection.close()


class FACT_NETWORK_THROUGHPUT_AGG(luigi.Task):
    target_table = 'fact_network_throughput_agg'
    runtime = datetime.datetime.now()
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    partition_name = luigi.ListParameter()
    start_filter = luigi.FloatParameter()
    inflow_filter = luigi.Parameter()
    query = QueryStrings.FACT_NETWORK_THROUGHPUT_AGG.value

    def output(self):
        return MySqlTarget(host=host, database=database, user=user, password=password, table=self.target_table,
                           update_id=str(self.runtime))

    def requires(self):
        return [FACT_VEHICLE_TRACE()]

    def run(self):
        connection = self.output().connect()
        for cur_partition in self.partition_name:
            try:
                cursor = connection.cursor(buffered=True)
                cur_query = self.query.format(partition=cur_partition,
                                              start_filter=self.start_filter,
                                              inflow_filter=self.inflow_filter
                                              )
                print("executing query FACT_NETWORK_THROUGHPUT_AGG...")
                cursor.execute(cur_query)
            except Error as err:
                if err.errno == errorcode.ER_NO_SUCH_TABLE:
                    # if first attempt fails with "relation not found", try creating table
                    # logger.info("Creating table %s", self.table)
                    connection.reconnect()
                else:
                    raise
        self.output().touch(connection)
        connection.commit()
        connection.close()


class FACT_NETWORK_SPEED(luigi.Task):
    target_table = 'fact_network_speed'
    runtime = datetime.datetime.now()
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    partition_name = luigi.ListParameter()
    start_filter = luigi.FloatParameter()
    inflow_filter = luigi.Parameter()
    outflow_filter = luigi.Parameter()
    query = QueryStrings.FACT_NETWORK_SPEED.value

    def output(self):
        return MySqlTarget(host=host, database=database, user=user, password=password, table=self.target_table,
                           update_id=str(self.runtime))

    def requires(self):
        return [FACT_VEHICLE_TRACE()]

    def run(self):
        connection = self.output().connect()
        for cur_partition in self.partition_name:
            try:
                cursor = connection.cursor(buffered=True)
                cur_query = self.query.format(partition=cur_partition,
                                              start_filter=self.start_filter,
                                              inflow_filter=self.inflow_filter,
                                              outflow_filter=self.outflow_filter
                                              )
                print("executing query FACT_NETWORK_SPEED...")
                cursor.execute(cur_query)
            except Error as err:
                if err.errno == errorcode.ER_NO_SUCH_TABLE:
                    # if first attempt fails with "relation not found", try creating table
                    # logger.info("Creating table %s", self.table)
                    connection.reconnect()
                else:
                    raise
        self.output().touch(connection)
        connection.commit()
        connection.close()


class FACT_VEHICLE_METRICS(luigi.Task):
    target_table = 'fact_vehicle_metrics'
    runtime = datetime.datetime.now()
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    partition_name = luigi.ListParameter()
    start_filter = luigi.FloatParameter()
    inflow_filter = luigi.Parameter()
    outflow_filter = luigi.Parameter()
    query = QueryStrings.FACT_VEHICLE_METRICS.value

    def output(self):
        return MySqlTarget(host=host, database=database, user=user, password=password, table=self.target_table,
                           update_id=str(self.runtime))

    def requires(self):
        return [FACT_VEHICLE_TRACE()]

    def run(self):
        connection = self.output().connect()
        for cur_partition in self.partition_name:
            try:
                cursor = connection.cursor(buffered=True)
                cur_query = self.query.format(partition=cur_partition,
                                              start_filter=self.start_filter,
                                              inflow_filter=self.inflow_filter,
                                              outflow_filter=self.outflow_filter
                                              )
                print("executing query FACT_VEHICLE_METRICS...")
                cursor.execute(cur_query)
            except Error as err:
                if err.errno == errorcode.ER_NO_SUCH_TABLE:
                    # if first attempt fails with "relation not found", try creating table
                    # logger.info("Creating table %s", self.table)
                    connection.reconnect()
                else:
                    raise
        self.output().touch(connection)
        connection.commit()
        connection.close()


class FACT_NETWORK_FUEL_EFFICIENCY_AGG(luigi.Task):
    target_table = 'fact_network_fuel_efficiency_agg'
    runtime = datetime.datetime.now()
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    partition_name = luigi.ListParameter()
    start_filter = luigi.FloatParameter()
    inflow_filter = luigi.Parameter()
    outflow_filter = luigi.Parameter()
    query = QueryStrings.FACT_NETWORK_FUEL_EFFICIENCY_AGG.value

    def output(self):
        return MySqlTarget(host=host, database=database, user=user, password=password, table=self.target_table,
                           update_id=str(self.runtime))

    def requires(self):
        return [FACT_VEHICLE_FUEL_EFFICIENCY_AGG(partition_name=self.partition_name, start_filter=self.start_filter,
                                                 inflow_filter=self.inflow_filter, outflow_filter=self.outflow_filter)]

    def run(self):
        connection = self.output().connect()
        for cur_partition in self.partition_name:
            try:
                cursor = connection.cursor(buffered=True)
                cur_query = self.query.format(partition=cur_partition)
                print("executing query FACT_NETWORK_FUEL_EFFICIENCY_AGG...")
                cursor.execute(cur_query)
            except Error as err:
                if err.errno == errorcode.ER_NO_SUCH_TABLE:
                    # if first attempt fails with "relation not found", try creating table
                    # logger.info("Creating table %s", self.table)
                    connection.reconnect()
                else:
                    raise
        self.output().touch(connection)
        connection.commit()
        connection.close()


class FACT_SAFETY_METRICS_AGG(luigi.Task):
    target_table = 'fact_safety_metrics_agg'
    runtime = datetime.datetime.now()
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    partition_name = luigi.ListParameter()
    start_filter = luigi.FloatParameter()
    max_decel = luigi.FloatParameter()
    leader_max_decel = luigi.FloatParameter()
    inflow_filter = luigi.Parameter()
    outflow_filter = luigi.Parameter()
    query = QueryStrings.FACT_SAFETY_METRICS_AGG.value

    def output(self):
        return MySqlTarget(host=host, database=database, user=user, password=password, table=self.target_table,
                           update_id=str(self.runtime))

    def requires(self):
        return [FACT_SAFETY_METRICS_3D(partition_name=self.partition_name, start_filter=self.start_filter,
                                       max_decel=self.max_decel, leader_max_decel=self.leader_max_decel,
                                       inflow_filter=self.inflow_filter, outflow_filter=self.outflow_filter)]

    def run(self):
        connection = self.output().connect()
        for cur_partition in self.partition_name:
            try:
                cursor = connection.cursor(buffered=True)
                cur_query = self.query.format(partition=cur_partition)
                print("executing query FACT_SAFETY_METRICS_AGG...")
                cursor.execute(cur_query)
            except Error as err:
                if err.errno == errorcode.ER_NO_SUCH_TABLE:
                    # if first attempt fails with "relation not found", try creating table
                    # logger.info("Creating table %s", self.table)
                    connection.reconnect()
                else:
                    raise
        self.output().touch(connection)
        connection.commit()
        connection.close()


class LEADERBOARD_CHART(luigi.Task):
    target_table = 'leaderboard_chart'
    runtime = datetime.datetime.now()
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    partition_name = luigi.ListParameter()
    start_filter = luigi.FloatParameter()
    max_decel = luigi.FloatParameter()
    leader_max_decel = luigi.FloatParameter()
    inflow_filter = luigi.Parameter()
    outflow_filter = luigi.Parameter()
    query = QueryStrings.LEADERBOARD_CHART.value

    def output(self):
        return MySqlTarget(host=host, database=database, user=user, password=password, table=self.target_table,
                           update_id=str(self.runtime))

    def requires(self):
        return [
            FACT_NETWORK_THROUGHPUT_AGG(
                partition_name=self.partition_name,
                start_filter=self.start_filter,
                inflow_filter=self.inflow_filter),
            FACT_NETWORK_SPEED(
                partition_name=self.partition_name,
                start_filter=self.start_filter,
                inflow_filter=self.inflow_filter,
                outflow_filter=self.outflow_filter),
            FACT_VEHICLE_METRICS(
                partition_name=self.partition_name,
                start_filter=self.start_filter,
                inflow_filter=self.inflow_filter,
                outflow_filter=self.outflow_filter),
            FACT_NETWORK_FUEL_EFFICIENCY_AGG(
                partition_name=self.partition_name,
                start_filter=self.start_filter,
                inflow_filter=self.inflow_filter,
                outflow_filter=self.outflow_filter),
            FACT_SAFETY_METRICS_AGG(
                partition_name=self.partition_name,
                start_filter=self.start_filter,
                max_decel=self.max_decel,
                leader_max_decel=self.leader_max_decel,
                inflow_filter=self.inflow_filter,
                outflow_filter=self.outflow_filter),
            FACT_INFEASIBLE_FLAGS(
                partition_name=self.partition_name,
                start_filter=self.start_filter,
                inflow_filter=self.inflow_filter,
                outflow_filter=self.outflow_filter),
        ]

    def run(self):
        connection = self.output().connect()
        for cur_partition in self.partition_name:
            try:
                cursor = connection.cursor(buffered=True)
                cur_query = self.query.format(partition=cur_partition)
                print("executing query LEADERBOARD_CHART...")
                cursor.execute(cur_query)
            except Error as err:
                if err.errno == errorcode.ER_NO_SUCH_TABLE:
                    # if first attempt fails with "relation not found", try creating table
                    # logger.info("Creating table %s", self.table)
                    connection.reconnect()
                else:
                    raise
        self.output().touch(connection)
        connection.commit()
        connection.close()


class LEADERBOARD_CHART_AGG(luigi.Task):
    target_table = 'leaderboard_chart_agg'
    runtime = datetime.datetime.now()
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    partition_name = luigi.ListParameter()
    start_filter = luigi.FloatParameter()
    max_decel = luigi.FloatParameter()
    leader_max_decel = luigi.FloatParameter()
    inflow_filter = luigi.Parameter()
    outflow_filter = luigi.Parameter()
    query = QueryStrings.LEADERBOARD_CHART_AGG.value

    def output(self):
        return MySqlTarget(host=host, database=database, user=user, password=password, table=self.target_table,
                           update_id=str(self.runtime))

    def requires(self):
        return [
            LEADERBOARD_CHART(
                partition_name=self.partition_name,
                start_filter=self.start_filter,
                max_decel=self.max_decel,
                leader_max_decel=self.leader_max_decel,
                inflow_filter=self.inflow_filter,
                outflow_filter=self.outflow_filter)]

    def run(self):
        connection = self.output().connect()
        for cur_partition in self.partition_name:
            try:
                cursor = connection.cursor(buffered=True)
                print("executing query LEADERBOARD_CHART_AGG...")
                cursor.execute(self.query)
            except Error as err:
                if err.errno == errorcode.ER_NO_SUCH_TABLE:
                    # if first attempt fails with "relation not found", try creating table
                    # logger.info("Creating table %s", self.table)
                    connection.reconnect()
                else:
                    raise
        self.output().touch(connection)
        connection.commit()
        connection.close()


'''
End point of the Graph
'''


class FACT_AV_TRACE(luigi.Task):

    target_table = 'fact_vehicle_trace'
    runtime = datetime.datetime.now()
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    partition_name = luigi.ListParameter()
    start_filter = luigi.FloatParameter()
    inflow_filter = luigi.Parameter()
    outflow_filter = luigi.Parameter()
    query = QueryStrings.FACT_AV_TRACE.value

    def output(self):
        return MySqlTarget(host=host, database=database, user=user, password=password,
                           table=self.target_table, update_id=str(self.runtime))

    def requires(self):
        return [FACT_VEHICLE_TRACE()]

    def run(self):
        connection = self.output().connect()
        for cur_partition in self.partition_name:
            try:
                cursor = connection.cursor(buffered=True)
                print("executing query FACT_AV_TRACE...")
                cur_query = self.query.format(partition=cur_partition, start_filter=self.start_filter,
                                              inflow_filter=self.inflow_filter, outflow_filter=self.outflow_filter)
                # print(cur_query)
                cursor.execute(cur_query)
            except Error as err:
                if err.errno == errorcode.ER_NO_SUCH_TABLE:
                    # if first attempt fails with "relation not found", try creating table
                    # logger.info("Creating table %s", self.table)
                    connection.reconnect()
                else:
                    raise

        self.output().touch(connection)
        connection.commit()
        connection.close()


class FACT_SAFETY_METRICS_2D(luigi.Task):

    target_table = 'fact_vehicle_trace'
    runtime = datetime.datetime.now()
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    partition_name = luigi.ListParameter()
    start_filter = luigi.FloatParameter()
    inflow_filter = luigi.Parameter()
    outflow_filter = luigi.Parameter()
    query = QueryStrings.FACT_SAFETY_METRICS_2D.value

    def output(self):
        return MySqlTarget(host=host, database=database, user=user, password=password,
                           table=self.target_table, update_id=str(self.runtime))

    def requires(self):
        return [FACT_VEHICLE_TRACE()]

    def run(self):
        connection = self.output().connect()
        for cur_partition in self.partition_name:
            try:
                cursor = connection.cursor(buffered=True)
                print("executing query FACT_SAFETY_METRICS_2D...")
                cur_query = self.query.format(partition=cur_partition, start_filter=self.start_filter,
                                              inflow_filter=self.inflow_filter, outflow_filter=self.outflow_filter)
                print(cur_query)
                cursor.execute(cur_query)
            except Error as err:
                if err.errno == errorcode.ER_NO_SUCH_TABLE:
                    # if first attempt fails with "relation not found", try creating table
                    # logger.info("Creating table %s", self.table)
                    connection.reconnect()
                else:
                    raise

        self.output().touch(connection)
        connection.commit()
        connection.close()


class FACT_NETWORK_INFLOWS_OUTFLOWS(luigi.Task):
    target_table = 'fact_network_inflows_outflows'
    runtime = datetime.datetime.now()
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    partition_name = luigi.ListParameter()
    start_filter = luigi.FloatParameter()
    inflow_filter = luigi.Parameter()
    outflow_filter = luigi.Parameter()
    query = QueryStrings.FACT_NETWORK_INFLOWS_OUTFLOWS.value

    def output(self):
        return MySqlTarget(host=host, database=database, user=user, password=password,
                           table=self.target_table, update_id=str(self.runtime))

    def requires(self):
        return [FACT_VEHICLE_TRACE()]

    def run(self):
        connection = self.output().connect()
        for cur_partition in self.partition_name:
            try:
                cursor = connection.cursor(buffered=True)
                print("executing query FACT_NETWORK_INFLOWS_OUTFLOWS...")
                cur_query = self.query.format(partition=cur_partition, start_filter=self.start_filter,
                                              inflow_filter=self.inflow_filter, outflow_filter=self.outflow_filter)
                # print(cur_query)
                cursor.execute(cur_query)
            except Error as err:
                if err.errno == errorcode.ER_NO_SUCH_TABLE:
                    # if first attempt fails with "relation not found", try creating table
                    # logger.info("Creating table %s", self.table)
                    connection.reconnect()
                else:
                    raise

        self.output().touch(connection)
        connection.commit()
        connection.close()


class FACT_SPACE_GAPS_BINNED(luigi.Task):
    target_table = 'fact_space_gaps_binned'
    runtime = datetime.datetime.now()
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    partition_name = luigi.ListParameter()
    query = QueryStrings.FACT_SPACE_GAPS_BINNED.value

    def output(self):
        return MySqlTarget(host=host, database=database, user=user, password=password,
                           table=self.target_table, update_id=str(self.runtime))

    def requires(self):
        return [FACT_VEHICLE_TRACE()]

    def run(self):
        connection = self.output().connect()
        for cur_partition in self.partition_name:
            try:
                cursor = connection.cursor(buffered=True)
                print("executing query FACT_SPACE_GAPS_BINNED...")
                cur_query = self.query.format(partition=cur_partition)
                # print(cur_query)
                cursor.execute(cur_query)
            except Error as err:
                if err.errno == errorcode.ER_NO_SUCH_TABLE:
                    # if first attempt fails with "relation not found", try creating table
                    # logger.info("Creating table %s", self.table)
                    connection.reconnect()
                else:
                    raise

        self.output().touch(connection)
        connection.commit()
        connection.close()


class FACT_TIME_GAPS_BINNED(luigi.Task):
    target_table = 'fact_time_gaps_binned'
    runtime = datetime.datetime.now()
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    partition_name = luigi.ListParameter()
    query = QueryStrings.FACT_TIME_GAPS_BINNED.value

    def output(self):
        return MySqlTarget(host=host, database=database, user=user, password=password,
                           table=self.target_table, update_id=str(self.runtime))

    def requires(self):
        return [FACT_VEHICLE_TRACE()]

    def run(self):
        connection = self.output().connect()
        for cur_partition in self.partition_name:
            try:
                cursor = connection.cursor(buffered=True)
                print("executing query FACT_TIME_GAPS_BINNED...")
                cur_query = self.query.format(partition=cur_partition, date=self.date)
                cursor.execute(cur_query)
            except Error as err:
                if err.errno == errorcode.ER_NO_SUCH_TABLE:
                    # if first attempt fails with "relation not found", try creating table
                    # logger.info("Creating table %s", self.table)
                    connection.reconnect()
                else:
                    raise

        self.output().touch(connection)
        connection.commit()
        connection.close()


class FACT_VEHICLE_COUNTS_BY_TIME(luigi.Task):
    target_table = 'fact_vehicle_counts_by_time'
    runtime = datetime.datetime.now()
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    partition_name = luigi.ListParameter()
    start_filter = luigi.FloatParameter()
    inflow_filter = luigi.Parameter()
    outflow_filter = luigi.Parameter()
    query = QueryStrings.FACT_VEHICLE_COUNTS_BY_TIME.value

    def output(self):
        return MySqlTarget(host=host, database=database, user=user, password=password,
                           table=self.target_table, update_id=str(self.runtime))

    def requires(self):
        return [FACT_VEHICLE_TRACE()]

    def run(self):
        connection = self.output().connect()
        for cur_partition in self.partition_name:
            try:
                cursor = connection.cursor(buffered=True)
                print("executing query FACT_VEHICLE_COUNTS_BY_TIME...")
                cur_query = self.query.format(partition=cur_partition, start_filter=self.start_filter,
                                              inflow_filter=self.inflow_filter, outflow_filter=self.outflow_filter)
                # print(cur_query)
                cursor.execute(cur_query)
            except Error as err:
                if err.errno == errorcode.ER_NO_SUCH_TABLE:
                    # if first attempt fails with "relation not found", try creating table
                    # logger.info("Creating table %s", self.table)
                    connection.reconnect()
                else:
                    raise

        self.output().touch(connection)
        connection.commit()
        connection.close()


class FACT_FOLLOWERSTOPPER_ENVELOPE(luigi.Task):
    target_table = 'fact_followerstopper_envelope'
    runtime = datetime.datetime.now()
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    partition_name = luigi.ListParameter()
    start_filter = luigi.FloatParameter()
    inflow_filter = luigi.Parameter()
    outflow_filter = luigi.Parameter()
    query = QueryStrings.FACT_FOLLOWERSTOPPER_ENVELOPE.value

    def output(self):
        return MySqlTarget(host=host, database=database, user=user, password=password,
                           table=self.target_table, update_id=str(self.runtime))

    def requires(self):
        return [FACT_VEHICLE_TRACE()]

    def run(self):
        connection = self.output().connect()
        for cur_partition in self.partition_name:
            try:
                cursor = connection.cursor(buffered=True)
                print("executing query FACT_FOLLOWERSTOPPER_ENVELOPE...")
                cur_query = self.query.format(partition=cur_partition, start_filter=self.start_filter,
                                              inflow_filter=self.inflow_filter, outflow_filter=self.outflow_filter)
                print(cur_query)
                cursor.execute(cur_query)
            except Error as err:
                if err.errno == errorcode.ER_NO_SUCH_TABLE:
                    # if first attempt fails with "relation not found", try creating table
                    # logger.info("Creating table %s", self.table)
                    connection.reconnect()
                else:
                    raise

        self.output().touch(connection)
        connection.commit()
        connection.close()


class FACT_NETWORK_METRICS_BY_DISTANCE_AGG(luigi.Task):
    target_table = 'fact_network_metrics_by_distance_agg'
    runtime = datetime.datetime.now()
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    partition_name = luigi.ListParameter()
    start_filter = luigi.FloatParameter()
    inflow_filter = luigi.Parameter()
    outflow_filter = luigi.Parameter()
    query = QueryStrings.FACT_NETWORK_METRICS_BY_DISTANCE_AGG.value

    def output(self):
        return MySqlTarget(host=host, database=database, user=user, password=password, table=self.target_table,
                           update_id=str(self.runtime))

    def requires(self):
        return [FACT_VEHICLE_TRACE(),
                TACOMA_FIT_DENOISED_ACCEL(partition_name=self.partition_name),
                PRIUS_FIT_DENOISED_ACCEL(partition_name=self.partition_name),
                COMPACT_SEDAN_FIT_DENOISED_ACCEL(partition_name=self.partition_name),
                MIDSIZE_SEDAN_FIT_DENOISED_ACCEL(partition_name=self.partition_name),
                RAV4_2019_FIT_DENOISED_ACCEL(partition_name=self.partition_name),
                MIDSIZE_SUV_FIT_DENOISED_ACCEL(partition_name=self.partition_name),
                LIGHT_DUTY_PICKUP_FIT_DENOISED_ACCEL(partition_name=self.partition_name),
                CLASS3_PND_TRUCK_FIT_DENOISED_ACCEL(partition_name=self.partition_name),
                CLASS8_TRACTOR_TRAILER_FIT_DENOISED_ACCEL(partition_name=self.partition_name)
                ]

    def run(self):
        connection = self.output().connect()
        for cur_partition in self.partition_name:
            try:
                cursor = connection.cursor(buffered=True)
                cur_query = self.query.format(partition=cur_partition, start_filter=self.start_filter,
                                              inflow_filter=self.inflow_filter, outflow_filter=self.outflow_filter)
                print(cur_query)
                print("executing query FACT_NETWORK_METRICS_BY_DISTANCE_AGG...")
                cursor.execute(cur_query)
            except Error as err:
                if err.errno == errorcode.ER_NO_SUCH_TABLE:
                    # if first attempt fails with "relation not found", try creating table
                    # logger.info("Creating table %s", self.table)
                    connection.reconnect()
                else:
                    raise
        self.output().touch(connection)
        connection.commit()
        connection.close()


class FACT_NETWORK_METRICS_BY_TIME_AGG(luigi.Task):
    target_table = 'fact_network_metrics_by_time_agg'
    runtime = datetime.datetime.now()
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    partition_name = luigi.ListParameter()
    start_filter = luigi.FloatParameter()
    inflow_filter = luigi.Parameter()
    outflow_filter = luigi.Parameter()
    query = QueryStrings.FACT_NETWORK_METRICS_BY_TIME_AGG.value

    def output(self):
        return MySqlTarget(host=host, database=database, user=user, password=password, table=self.target_table,
                           update_id=str(self.runtime))

    def requires(self):
        return [FACT_VEHICLE_TRACE(),
                TACOMA_FIT_DENOISED_ACCEL(partition_name=self.partition_name),
                PRIUS_FIT_DENOISED_ACCEL(partition_name=self.partition_name),
                COMPACT_SEDAN_FIT_DENOISED_ACCEL(partition_name=self.partition_name),
                MIDSIZE_SEDAN_FIT_DENOISED_ACCEL(partition_name=self.partition_name),
                RAV4_2019_FIT_DENOISED_ACCEL(partition_name=self.partition_name),
                MIDSIZE_SUV_FIT_DENOISED_ACCEL(partition_name=self.partition_name),
                LIGHT_DUTY_PICKUP_FIT_DENOISED_ACCEL(partition_name=self.partition_name),
                CLASS3_PND_TRUCK_FIT_DENOISED_ACCEL(partition_name=self.partition_name),
                CLASS8_TRACTOR_TRAILER_FIT_DENOISED_ACCEL(partition_name=self.partition_name)
                ]

    def run(self):
        connection = self.output().connect()
        for cur_partition in self.partition_name:
            try:
                cursor = connection.cursor(buffered=True)
                cur_query = self.query.format(partition=cur_partition, start_filter=self.start_filter,
                                              inflow_filter=self.inflow_filter, outflow_filter=self.outflow_filter)
                print("executing query FACT_NETWORK_METRICS_BY_TIME_AGG...")
                cursor.execute(cur_query)
            except Error as err:
                if err.errno == errorcode.ER_NO_SUCH_TABLE:
                    # if first attempt fails with "relation not found", try creating table
                    # logger.info("Creating table %s", self.table)
                    connection.reconnect()
                else:
                    raise
        self.output().touch(connection)
        connection.commit()
        connection.close()


class FACT_VEHICLE_FUEL_EFFICIENCY_BINNED(luigi.Task):
    target_table = 'fact_vehicle_fuel_efficiency_binned'
    runtime = datetime.datetime.now()
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    partition_name = luigi.ListParameter()
    start_filter = luigi.FloatParameter()
    inflow_filter = luigi.Parameter()
    outflow_filter = luigi.Parameter()
    query = QueryStrings.FACT_VEHICLE_FUEL_EFFICIENCY_BINNED.value

    def output(self):
        return MySqlTarget(host=host, database=database, user=user, password=password, table=self.target_table,
                           update_id=str(self.runtime))

    def requires(self):
        return [
            FACT_VEHICLE_FUEL_EFFICIENCY_AGG(partition_name=self.partition_name, start_filter=self.start_filter,
                                             inflow_filter=self.inflow_filter, outflow_filter=self.outflow_filter)
        ]

    def run(self):
        connection = self.output().connect()
        for cur_partition in self.partition_name:
            try:
                cursor = connection.cursor(buffered=True)
                cur_query = self.query.format(partition=cur_partition, start_filter=self.start_filter,
                                              inflow_filter=self.inflow_filter, outflow_filter=self.outflow_filter)
                print("executing query FACT_VEHICLE_FUEL_EFFICIENCY_BINNED...")
                cursor.execute(self.query)
            except Error as err:
                if err.errno == errorcode.ER_NO_SUCH_TABLE:
                    # if first attempt fails with "relation not found", try creating table
                    # logger.info("Creating table %s", self.table)
                    connection.reconnect()
                else:
                    raise
        self.output().touch(connection)
        connection.commit()
        connection.close()


class FACT_SAFETY_METRICS_BINNED(luigi.Task):
    target_table = 'fact_safety_metrics_binned'
    runtime = datetime.datetime.now()
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    partition_name = luigi.ListParameter()
    start_filter = luigi.FloatParameter()
    max_decel = luigi.FloatParameter()
    leader_max_decel = luigi.FloatParameter()
    inflow_filter = luigi.Parameter()
    outflow_filter = luigi.Parameter()
    query = QueryStrings.FACT_SAFETY_METRICS_BINNED.value

    def output(self):
        return MySqlTarget(host=host, database=database, user=user, password=password, table=self.target_table,
                           update_id=str(self.runtime))

    def requires(self):
        return [
            FACT_SAFETY_METRICS_3D(partition_name=self.partition_name, start_filter=self.start_filter,
                                   max_decel=self.max_decel,
                                   leader_max_decel=self.leader_max_decel,
                                   inflow_filter=self.inflow_filter, outflow_filter=self.outflow_filter)
        ]

    def run(self):
        connection = self.output().connect()
        for cur_partition in self.partition_name:
            try:
                cursor = connection.cursor(buffered=True)
                cur_query = self.query.format(partition=cur_partition)
                print("executing query FACT_SAFETY_METRICS_BINNED...")
                cursor.execute(self.query)
            except Error as err:
                if err.errno == errorcode.ER_NO_SUCH_TABLE:
                    # if first attempt fails with "relation not found", try creating table
                    # logger.info("Creating table %s", self.table)
                    connection.reconnect()
                else:
                    raise
        self.output().touch(connection)
        connection.commit()
        connection.close()


class FACT_TOP_SCORES(luigi.Task):
    target_table = 'fact_top_scores'
    runtime = datetime.datetime.now()
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    partition_name = luigi.ListParameter()
    start_filter = luigi.FloatParameter()
    max_decel = luigi.FloatParameter()
    leader_max_decel = luigi.FloatParameter()
    inflow_filter = luigi.Parameter()
    outflow_filter = luigi.Parameter()
    query = QueryStrings.FACT_TOP_SCORES.value

    def output(self):
        return MySqlTarget(host=host, database=database, user=user, password=password, table=self.target_table,
                           update_id=str(self.runtime))

    def requires(self):
        return [
            LEADERBOARD_CHART_AGG(partition_name=self.partition_name, start_filter=self.start_filter,
                                  max_decel=self.max_decel,
                                  leader_max_decel=self.leader_max_decel,
                                  inflow_filter=self.inflow_filter, outflow_filter=self.outflow_filter)
        ]

    def run(self):
        connection = self.output().connect()
        try:
            cursor = connection.cursor(buffered=True)
            print("executing query FACT_TOP_SCORES...")
            cursor.execute(self.query)
        except Error as err:
            if err.errno == errorcode.ER_NO_SUCH_TABLE:
                # if first attempt fails with "relation not found", try creating table
                # logger.info("Creating table %s", self.table)
                connection.reconnect()
            else:
                raise
        self.output().touch(connection)
        connection.commit()
        connection.close()
