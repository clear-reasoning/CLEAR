import mysql.connector
from mysql.connector import errorcode
from database import DB_NAME, connect

# load data from csv file statements
CLIENT_LOAD_STATEMENT = \
    "LOAD DATA LOCAL INFILE  \'{data_path}\' " \
    "INTO TABLE {table} " \
    "FIELDS TERMINATED BY \',\' " \
    "ENCLOSED BY \'\"\' " \
    "LINES TERMINATED BY \'\\n\' " \
    "IGNORE 1 ROWS;"

CLIENT_LOAD_STATEMENT_NO_ENCLOSING = \
    "LOAD DATA LOCAL INFILE  \'{data_path}\' " \
    "INTO TABLE {table} " \
    "FIELDS TERMINATED BY \',\' " \
    "LINES TERMINATED BY \'\\n\' " \
    "IGNORE 1 ROWS;"

def load_data(cnx, table, data_path, enclosing=True):
    try:
        cursor = cnx.cursor()
        if enclosing:
            cursor.execute(CLIENT_LOAD_STATEMENT.format(
                table=table, data_path=data_path))
        else:
            cursor.execute(CLIENT_LOAD_STATEMENT_NO_ENCLOSING.format(
                table=table, data_path=data_path))
    except mysql.connector.Error as err:
        print(err.msg)
    else:
        cnx.commit()
        print("load data into {} successfully.".format(table))
    finally:
        cursor.close()

if __name__ == '__main__':

    cnx = connect(DB_NAME)

    # load_data(cnx, 'fact_vehicle_trace',
    #                '/home/circles/Downloads/flow_ffc9edc5f72b428a986cea83fe90fd0b.csv',
    #                enclosing=False)
    # load_data(cnx, 'metadata_table',
    #                '/home/circles/Downloads/flow_ffc9edc5f72b428a986cea83fe90fd0b_METADATA.csv',
    #                enclosing=False)
    # load_data(cnx, 'fact_energy_trace',
    #                '/home/circles/Downloads/fact_energy_trace.csv')
    # load_data(cnx, 'fact_infeasible_flags',
    #                '/home/circles/Downloads/fact_infeasible_flags.csv')
    # load_data(cnx, 'fact_vehicle_counts_by_time',
    #                '/home/circles/Downloads/fact_vehicle_counts_by_time.csv')
    # load_data(cnx, 'fact_safety_metrics',
    #                '/home/circles/Downloads/fact_safety_metrics.csv')
    # load_data(cnx, 'fact_safety_metrics_agg',
    #                '/home/circles/Downloads/fact_safety_metrics_agg.csv')
    # load_data(cnx, 'fact_safety_metrics_binned',
    #                '/home/circles/Downloads/fact_safety_metrics_binned.csv')
    # load_data(cnx, 'fact_network_throughput_agg',
    #                '/home/circles/Downloads/fact_network_throughput_agg.csv')
    # load_data(cnx, 'fact_network_inflows_outflows',
    #                '/home/circles/Downloads/fact_network_inflows_outflows.csv')
    # load_data(cnx, 'fact_network_speed',
    #                '/home/circles/Downloads/fact_network_speed.csv')
    # load_data(cnx, 'fact_vehicle_metrics',
    #                '/home/circles/Downloads/fact_vehicle_metrics.csv')
    # load_data(cnx, 'fact_followerstopper_envelope',
    #                '/home/circles/Downloads/fact_followerstopper_envelope.csv')
    # load_data(cnx, 'fact_vehicle_fuel_efficiency_agg',
    #                '/home/circles/Downloads/fact_vehicle_fuel_efficiency_agg.csv')
    # load_data(cnx, 'fact_vehicle_fuel_efficiency_binned',
    #                '/home/circles/Downloads/fact_vehicle_fuel_efficiency_binned.csv')
    # load_data(cnx, 'fact_network_metrics_by_distance_agg',
    #                '/home/circles/Downloads/fact_network_metrics_by_distance_agg.csv')
    # load_data(cnx, 'fact_network_metrics_by_time_agg',
    #                '/home/circles/Downloads/fact_network_metrics_by_time_agg.csv')
    # load_data(cnx, 'fact_network_fuel_efficiency_agg',
    #                '/home/circles/Downloads/fact_network_fuel_efficiency_agg.csv')
    # load_data(cnx, 'leaderboard_chart',
    #                '/home/circles/Downloads/leaderboard_chart.csv')
    # load_data(cnx, 'leaderboard_chart_agg',
    #                '/home/circles/Downloads/leaderboard_chart_agg.csv')
    # load_data(cnx, 'fact_top_scores',
    #                '/home/circles/Downloads/fact_top_scores.csv')
    # load_data(cnx, 'fact_vehicle_trace',
    #                '/home/circles/Downloads/baselines/flow_b0a18b6def834aefa1fd43a99eeee863.csv',
    #                enclosing=False)
    # load_data(cnx, 'metadata_table',
    #                '/home/circles/Downloads/baselines/flow_b0a18b6def834aefa1fd43a99eeee863_METADATA.csv',
    #                enclosing=False)
    # load_data(cnx, 'fact_energy_trace',
    #                '/home/circles/Downloads/baselines/fact_energy_trace.csv')
    # load_data(cnx, 'fact_infeasible_flags',
    #                '/home/circles/Downloads/baselines/fact_infeasible_flags.csv')
    # load_data(cnx, 'fact_vehicle_counts_by_time',
    #                '/home/circles/Downloads/baselines/fact_vehicle_counts_by_time.csv')
    # load_data(cnx, 'fact_safety_metrics',
    #                '/home/circles/Downloads/baselines/fact_safety_metrics.csv')
    # load_data(cnx, 'fact_safety_metrics_agg',
    #                '/home/circles/Downloads/baselines/fact_safety_metrics_agg.csv')
    # load_data(cnx, 'fact_safety_metrics_binned',
    #                '/home/circles/Downloads/baselines/fact_safety_metrics_binned.csv')
    # load_data(cnx, 'fact_network_throughput_agg',
    #                '/home/circles/Downloads/baselines/fact_network_throughput_agg.csv')
    # load_data(cnx, 'fact_network_inflows_outflows',
    #                '/home/circles/Downloads/baselines/fact_network_inflows_outflows.csv')
    # load_data(cnx, 'fact_network_speed',
    #                '/home/circles/Downloads/baselines/fact_network_speed.csv')
    # load_data(cnx, 'fact_vehicle_metrics',
    #                '/home/circles/Downloads/baselines/fact_vehicle_metrics.csv')
    # load_data(cnx, 'fact_followerstopper_envelope',
    #                '/home/circles/Downloads/baselines/fact_followerstopper_envelope.csv')
    # load_data(cnx, 'fact_vehicle_fuel_efficiency_agg',
    #                '/home/circles/Downloads/baselines/fact_vehicle_fuel_efficiency_agg.csv')
    # load_data(cnx, 'fact_vehicle_fuel_efficiency_binned',
    #                '/home/circles/Downloads/baselines/fact_vehicle_fuel_efficiency_binned.csv')
    # load_data(cnx, 'fact_network_metrics_by_distance_agg',
    #                '/home/circles/Downloads/baselines/fact_network_metrics_by_distance_agg.csv')
    # load_data(cnx, 'fact_network_metrics_by_time_agg',
    #                '/home/circles/Downloads/baselines/fact_network_metrics_by_time_agg.csv')
    # load_data(cnx, 'fact_network_fuel_efficiency_agg',
    #                '/home/circles/Downloads/baselines/fact_network_fuel_efficiency_agg.csv')
    # load_data(cnx, 'leaderboard_chart',
    #                '/home/circles/Downloads/baselines/leaderboard_chart.csv')
    # load_data(cnx, 'fact_vehicle_distributions',
    #                '/home/circles/Downloads/distributions.csv',
    #                enclosing=False)
    load_data(cnx, 'fact_safety_matrix',
                   '/home/circles/Downloads/fact_safety_matrix.csv',
                   enclosing=False)



    cnx.close()