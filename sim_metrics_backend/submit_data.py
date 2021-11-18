from configparser import SectionProxy
from sim_metrics_backend.database import connect
import pandas as pd
import mysql.connector  # pip install mysql-connector-python
import time
import os

FLOW_DATA_TABLE_NAME = "fact_vehicle_trace"
METADATA_TABLE_NAME = "metadata_table"
BATCH_SIZE = 200
NAN_VALUES = {"headway": 252.0, "leader_id": "",
              "follower_id": "", "leader_rel_speed": 0}

FLOW_DATA_SQL = 'INSERT INTO `fact_vehicle_trace` ('\
    '`time_step`,`id`,`x`,`y`,`speed`,`headway`,'\
    '`leader_id`,`follower_id`,`leader_rel_speed`,'\
    '`target_accel_with_noise_with_failsafe`,'\
    '`target_accel_no_noise_no_failsafe`,'\
    '`target_accel_with_noise_no_failsafe`,'\
    '`target_accel_no_noise_with_failsafe`,'\
    '`realized_accel`,`road_grade`,`edge_id`,'\
    '`lane_id`,`distance`,`relative_position`,'\
    '`source_id`,`run_id`,`submission_date`) '\
    'VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'

METADATA_SQL = 'INSERT INTO `metadata_table` ('\
    '`source_id`,`submission_time`,`network`,'\
    '`is_baseline`,`submitter_name`,`strategy`,'\
    '`version`,`on_ramp`,`penetration_rate`,'\
    '`road_grade`,`is_benchmark`)'\
    'VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'

SERVER_USER = 'circles'
SERVER_ADDR = 'circles.banatao.berkeley.edu'
SERVER_PATH = '~/sdb/{}/'


def submit(data, isMeta, cnx):
    if isMeta:
        print("submitting metadata")
    else:
        print("submitting flow data")
    cursor = cnx.cursor()
    start = 0
    while start < data.shape[0]:
        try:
            batchData = data[start:start+BATCH_SIZE].fillna(NAN_VALUES)
            if isMeta == False:
                # add submission date
                submissionDate = time.strftime("%Y-%m-%d", time.localtime())
                batchData.insert(
                    batchData.shape[1], "submission_data", submissionDate)
            # convert to tuples in order to use executemany
            batchDataInTuples = [tuple(row) for row in batchData.values]
            sql = METADATA_SQL if isMeta == True else FLOW_DATA_SQL  # choose SQL
            cursor.executemany(sql, batchDataInTuples)
            start += BATCH_SIZE
        except mysql.connector.Error as err:
            print(cursor.statement)
            print("error:", err.msg)
            break
    else:
        cnx.commit()
        print(data.shape[0], "records are created")
    cursor.close()


def submitData(filePath, isMeta):
    cnx = connect(user='circles.user', host='circles.banatao.berkeley.edu')  # get database connection

    try:
        csvData = pd.read_csv(filePath)
    except:
        print("filePath: '{}' does not exist".format(filePath))
    # there are two types of files
    submit(csvData, isMeta, cnx)

    cnx.close()


def uploadPng(filePath, image_type):
    # need to run ssh-keygen or put in password every time.
    os.system("scp %s %s@%s:%s" %
              (filePath, SERVER_USER, SERVER_ADDR, SERVER_PATH.format(image_type)))


if __name__ == '__main__':
    filePath = "/Users/dongwang/Desktop/capstone/pic.png"
    now = time.time()
    uploadPng(filePath)
    print(time.time()-now)
