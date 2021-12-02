
from sim_metrics_backend.data_pipeline.luigiSqlTask import *

import luigi

import pandas as pd

import sys
sys.path.append('..')
from sim_metrics_backend.database import get_network
from sim_metrics_backend.query import network_filters

max_decel = -1.0
leader_max_decel = -2.0

def run_queries_on_new_data(cnx, newdata):

    source_ids = newdata['source_id'].unique()
    run_luigi(cnx, source_ids)


def run_luigi(cnt, source_ids):

    network = get_network(cnt, source_ids[0])

    start_filter = network_filters[network]['warmup_steps']
    inflow_filter = network_filters[network]['inflow_filter']
    outflow_filter = network_filters[network]['outflow_filter']

    luigiTaskList = []
    # luigiTaskList.append(luigiSqlTask.FACT_AV_TRACE(partition_name=source_ids, start_filter=start_filter, inflow_filter=inflow_filter, outflow_filter=outflow_filter))
    # luigiTaskList.append(luigiSqlTask.TACOMA_FIT_DENOISED_ACCEL(partition_name=source_ids))
    # luigiTaskList.append(luigiSqlTask.PRIUS_FIT_DENOISED_ACCEL(partition_name=source_ids))
    # luigiTaskList.append(luigiSqlTask.MIDSIZE_SUV_FIT_DENOISED_ACCEL(partition_name=source_ids))
    # luigiTaskList.append(luigiSqlTask.COMPACT_SEDAN_FIT_DENOISED_ACCEL(partition_name=source_ids))
    # luigiTaskList.append(luigiSqlTask.MIDSIZE_SEDAN_FIT_DENOISED_ACCEL(partition_name = source_ids))
    # luigiTaskList.append(luigiSqlTask.RAV4_2019_FIT_DENOISED_ACCEL(partition_name = source_ids))
    # luigiTaskList.append(luigiSqlTask.LIGHT_DUTY_PICKUP_FIT_DENOISED_ACCEL(partition_name = source_ids))
    # luigiTaskList.append(luigiSqlTask.CLASS3_PND_TRUCK_FIT_DENOISED_ACCEL(partition_name = source_ids))
    # luigiTaskList.append(luigiSqlTask.CLASS8_TRACTOR_TRAILER_FIT_DENOISED_ACCEL(partition_name = source_ids))
    # luigiTaskList.append(luigiSqlTask.FACT_SAFETY_METRICS_3D(partition_name=source_ids, start_filter = start_filter, max_decel=max_decel, leader_max_decel=leader_max_decel, inflow_filter=inflow_filter, outflow_filter=outflow_filter))
    # luigiTaskList.append(luigiSqlTask.FACT_NETWORK_THROUGHPUT_AGG(partition_name = source_ids, start_filter=start_filter, inflow_filter=inflow_filter))
    # luigiTaskList.append(luigiSqlTask.FACT_NETWORK_SPEED(partition_name = source_ids, start_filter=start_filter, inflow_filter=inflow_filter, outflow_filter = outflow_filter))
    # luigiTaskList.append(luigiSqlTask.FACT_VEHICLE_METRICS(partition_name = source_ids, start_filter=start_filter, inflow_filter=inflow_filter, outflow_filter = outflow_filter))

    # luigiTaskList.append(luigiSqlTask.FACT_VEHICLE_FUEL_EFFICIENCY_AGG(partition_name=source_ids, start_filter=start_filter, inflow_filter=inflow_filter, outflow_filter=outflow_filter))

    # luigiTaskList.append(luigiSqlTask.FACT_INFEASIBLE_FLAGS(partition_name=source_ids, start_filter=start_filter, inflow_filter=inflow_filter, outflow_filter=outflow_filter))
    # luigiTaskList.append(luigiSqlTask.FACT_NETWORK_FUEL_EFFICIENCY_AGG(partition_name=source_ids, start_filter=start_filter, inflow_filter=inflow_filter, outflow_filter=outflow_filter))
    # luigiTaskList.append(luigiSqlTask.FACT_SAFETY_METRICS_AGG(partition_name=source_ids, start_filter = start_filter, max_decel=max_decel, leader_max_decel=leader_max_decel, inflow_filter=inflow_filter, outflow_filter=outflow_filter))

    # luigiTaskList.append(luigiSqlTask.LEADERBOARD_CHART(partition_name=source_ids, start_filter = start_filter, max_decel=max_decel, leader_max_decel=leader_max_decel, inflow_filter=inflow_filter, outflow_filter=outflow_filter))
    # luigiTaskList.append(luigiSqlTask.LEADERBOARD_CHART_AGG(partition_name=source_ids, start_filter = start_filter, max_decel=max_decel, leader_max_decel=leader_max_decel, inflow_filter=inflow_filter, outflow_filter=outflow_filter))

    '''
    Only add end point
    '''
    #problematic queries not response?
    source_ids = list(source_ids)
    luigiTaskList.append(FACT_SAFETY_METRICS_2D(partition_name=source_ids, start_filter=start_filter, inflow_filter=inflow_filter, outflow_filter=outflow_filter))
    luigiTaskList.append(FACT_SPACE_GAPS_BINNED(partition_name=source_ids))
    luigiTaskList.append(FACT_TIME_GAPS_BINNED(partition_name=source_ids))

    luigiTaskList.append(FACT_NETWORK_INFLOWS_OUTFLOWS(partition_name=source_ids, start_filter=start_filter, inflow_filter=inflow_filter, outflow_filter=outflow_filter))
    luigiTaskList.append(FACT_AV_TRACE(partition_name=source_ids, start_filter=start_filter, inflow_filter=inflow_filter, outflow_filter=outflow_filter))

    luigiTaskList.append(FACT_VEHICLE_COUNTS_BY_TIME(partition_name=source_ids, start_filter=start_filter, inflow_filter=inflow_filter, outflow_filter=outflow_filter))
    luigiTaskList.append(FACT_FOLLOWERSTOPPER_ENVELOPE(partition_name=source_ids, start_filter=start_filter, inflow_filter=inflow_filter, outflow_filter=outflow_filter))
    luigiTaskList.append(FACT_NETWORK_METRICS_BY_DISTANCE_AGG(partition_name=source_ids, start_filter=start_filter, inflow_filter=inflow_filter, outflow_filter=outflow_filter))
    luigiTaskList.append(FACT_NETWORK_METRICS_BY_TIME_AGG(partition_name=source_ids, start_filter=start_filter, inflow_filter=inflow_filter, outflow_filter=outflow_filter))
    luigiTaskList.append(FACT_VEHICLE_FUEL_EFFICIENCY_BINNED(partition_name=source_ids, start_filter=start_filter, inflow_filter=inflow_filter, outflow_filter=outflow_filter))
    luigiTaskList.append(FACT_SAFETY_METRICS_BINNED(partition_name=source_ids, start_filter=start_filter, max_decel = max_decel, leader_max_decel = leader_max_decel, inflow_filter=inflow_filter, outflow_filter=outflow_filter))
    luigiTaskList.append(FACT_TOP_SCORES(partition_name=source_ids, start_filter=start_filter, max_decel = max_decel, leader_max_decel = leader_max_decel, inflow_filter=inflow_filter, outflow_filter=outflow_filter))



    luigi.build(luigiTaskList, detailed_summary=True, local_scheduler=True)


if __name__ == '__main__':

    cur_partition_name = ["flow_b0a18b6def834aefa1fd43a99eeee863"]
    run_luigi(cur_partition_name)



