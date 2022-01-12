import mysql.connector
from mysql.connector import errorcode
from database import DB_NAME, connect

# Specify the name and schema of all the tables
TABLES = {}
TABLES["fact_vehicle_trace"] = (
    "CREATE TABLE IF NOT EXISTS `fact_vehicle_trace` ("
    "`time_step` float,"
    "`id` varchar(50),"
    "`x` float,"
    "`y` float,"
    "`speed` float,"
    "`headway` float,"
    "`leader_id` varchar(50),"
    "`follower_id` varchar(50),"
    "`leader_rel_speed` float,"
    "`target_accel_with_noise_with_failsafe` float,"
    "`target_accel_no_noise_no_failsafe` float,"
    "`target_accel_with_noise_no_failsafe` float,"
    "`target_accel_no_noise_with_failsafe` float,"
    "`realized_accel` float,"
    "`road_grade` float,"
    "`edge_id` varchar(50),"
    "`lane_id` varchar(50),"
    "`distance` float,"
    "`relative_position` float,"
    "`source_id` char(37),"
    "`run_id` char(6),"
    "`submission_date` date,"
    "PRIMARY KEY (`source_id`, `run_id`, `time_step`, `id`)"
    ") ENGINE=InnoDB")
TABLES["fact_energy_trace"] = (
    "CREATE TABLE IF NOT EXISTS `fact_energy_trace` ("
    "`id` varchar(50),"
    "`time_step` float,"
    "`speed` float,"
    "`acceleration` float,"
    "`road_grade` float,"
    "`edge_id` varchar(50),"
    "`power` float,"
    "`fuel_rate_mass` float,"
    "`fuel_rate_vol` float,"
    "`energy_model_id` varchar(50),"
    "`source_id` char(37),"
    "`infeasible_flag` float,"
    "PRIMARY KEY (`source_id`, `energy_model_id`, `time_step`, `id`)"
    ") ENGINE=InnoDB")
TABLES["fact_infeasible_flags"] = (
    "CREATE TABLE IF NOT EXISTS `fact_infeasible_flags` ("
    "`source_id` char(37),"
    "`distribution_model_id` varchar(100),"
    "`percent_infeasible` float,"
    "PRIMARY KEY (`source_id`, `distribution_model_id`)"
    ") ENGINE=InnoDB")
TABLES["fact_vehicle_counts_by_time"] = (
    "CREATE TABLE IF NOT EXISTS `fact_vehicle_counts_by_time` ("
    "`source_id` char(37),"
    "`time_step` float,"
    "`vehicle_count` int,"
    "PRIMARY KEY (`source_id`, `time_step`)"
    ") ENGINE=InnoDB")
TABLES["fact_safety_metrics"] = (
    "CREATE TABLE IF NOT EXISTS `fact_safety_metrics` ("
    "`id` varchar(50),"
    "`time_step` float,"
    "`safety_value` float,"
    "`safety_model` varchar(50),"
    "`source_id` char(37),"
    "PRIMARY KEY (`source_id`, `time_step`, `id`)"
    ") ENGINE=InnoDB")
TABLES["fact_safety_metrics_agg"] = (
    "CREATE TABLE IF NOT EXISTS `fact_safety_metrics_agg` ("
    "`source_id` char(37),"
    "`safety_rate` float,"
    "`safety_value_max` float,"
    "PRIMARY KEY (`source_id`)"
    ") ENGINE=InnoDB")
TABLES["fact_safety_metrics_binned"] = (
    "CREATE TABLE IF NOT EXISTS `fact_safety_metrics_binned` ("
    "`source_id` char(37),"
    "`safety_value_bin` varchar(30),"
    "`count` int,"
    "PRIMARY KEY (`source_id`, `safety_value_bin`)"
    ") ENGINE=InnoDB")
TABLES["fact_network_throughput_agg"] = (
    "CREATE TABLE IF NOT EXISTS `fact_network_throughput_agg` ("
    "`source_id` char(37),"
    "`throughput_per_hour` float,"
    "PRIMARY KEY (`source_id`)"
    ") ENGINE=InnoDB")
TABLES["fact_network_inflows_outflows"] = (
    "CREATE TABLE IF NOT EXISTS `fact_network_inflows_outflows` ("
    "`time_step` float,"
    "`source_id` char(37),"
    "`inflow_rate` int,"
    "`outflow_rate` int,"
    "PRIMARY KEY (`source_id`, `time_step`)"
    ") ENGINE=InnoDB")
TABLES["fact_network_speed"] = (
    "CREATE TABLE IF NOT EXISTS `fact_network_speed` ("
    "`source_id` char(37),"
    "`avg_instantaneous_speed` float,"
    "`avg_network_speed` float,"
    "`total_vmt` float,"
    "PRIMARY KEY (`source_id`)"
    ") ENGINE=InnoDB")
TABLES["fact_vehicle_metrics"] = (
    "CREATE TABLE IF NOT EXISTS `fact_vehicle_metrics` ("
    "`source_id` char(37),"
    "`lane_changes_per_vehicle` float,"
    "`space_gap_min` float,"
    "`space_gap_max` float,"
    "`space_gap_avg` float,"
    "`space_gap_stddev` float,"
    "`av_space_gap_min` float,"
    "`av_space_gap_max` float,"
    "`av_space_gap_avg` float,"
    "`av_space_gap_stddev` float,"
    "`time_gap_min` float,"
    "`time_gap_max` float,"
    "`time_gap_avg` float,"
    "`time_gap_stddev` float,"
    "`av_time_gap_min` float,"
    "`av_time_gap_max` float,"
    "`av_time_gap_avg` float,"
    "`av_time_gap_stddev` float,"
    "`speed_min` float,"
    "`speed_max` float,"
    "`speed_avg` float,"
    "`speed_stddev` float,"
    "`accel_min` float,"
    "`accel_max` float,"
    "`accel_avg` float,"
    "`accel_stddev` float,"
    "PRIMARY KEY (`source_id`)"
    ") ENGINE=InnoDB")
TABLES["fact_followerstopper_envelope"] = (
    "CREATE TABLE IF NOT EXISTS `fact_followerstopper_envelope` ("
    "`source_id` char(37),"
    "`region_1_proportion` float,"
    "`region_2_proportion` float,"
    "`region_3_proportion` float,"
    "`region_4_proportion` float,"
    "`av_region_1_proportion` float,"
    "`av_region_2_proportion` float,"
    "`av_region_3_proportion` float,"
    "`av_region_4_proportion` float,"
    "PRIMARY KEY (`source_id`)"
    ") ENGINE=InnoDB")
TABLES["fact_vehicle_fuel_efficiency_agg"] = (
    "CREATE TABLE IF NOT EXISTS `fact_vehicle_fuel_efficiency_agg` ("
    "`source_id` char(37),"
    "`id` varchar(50),"
    "`energy_model_id` varchar(50),"
    "`distance_meters` float,"
    "`energy_joules` float,"
    "`fuel_grams` float,"
    "`fuel_gallons` float,"
    "`efficiency_meters_per_joules` float,"
    "`efficiency_miles_per_gallon` float,"
    "`is_locally_measurable` int,"
    "PRIMARY KEY (`source_id`, `energy_model_id`, `id`)"
    ") ENGINE=InnoDB")
TABLES["fact_vehicle_fuel_efficiency_binned"] = (
    "CREATE TABLE IF NOT EXISTS `fact_vehicle_fuel_efficiency_binned` ("
    "`source_id` char(37),"
    "`distribution_model_id` varchar(100),"
    "`fuel_efficiency_bin` varchar(50),"
    "`count` int,"
    "PRIMARY KEY (`source_id`, `distribution_model_id`, `fuel_efficiency_bin`)"
    ") ENGINE=InnoDB")
TABLES["fact_network_metrics_by_distance_agg"] = (
    "CREATE TABLE IF NOT EXISTS `fact_network_metrics_by_distance_agg` ("
    "`source_id` char(37),"
    "`distance_meters_bin` float,"
    "`cumulative_energy_avg` float,"
    "`cumulative_energy_lower_bound` float,"
    "`cumulative_energy_upper_bound` float,"
    "`speed_avg` float,"
    "`speed_upper_bound` float,"
    "`speed_lower_bound` float,"
    "`accel_avg` float,"
    "`accel_upper_bound` float,"
    "`accel_lower_bound` float,"
    "`instantaneous_energy_avg` float,"
    "`instantaneous_energy_upper_bound` float,"
    "`instantaneous_energy_lower_bound` float,"
    "PRIMARY KEY (`source_id`)"
    ") ENGINE=InnoDB")
TABLES["fact_network_metrics_by_time_agg"] = (
    "CREATE TABLE IF NOT EXISTS `fact_network_metrics_by_time_agg` ("
    "`source_id` char(37),"
    "`time_seconds_bin` float,"
    "`cumulative_energy_avg` float,"
    "`cumulative_energy_lower_bound` float,"
    "`cumulative_energy_upper_bound` float,"
    "`speed_avg` float,"
    "`speed_upper_bound` float,"
    "`speed_lower_bound` float,"
    "`accel_avg` float,"
    "`accel_upper_bound` float,"
    "`accel_lower_bound` float,"
    "`instantaneous_energy_avg` float,"
    "`instantaneous_energy_upper_bound` float,"
    "`instantaneous_energy_lower_bound` float,"
    "PRIMARY KEY (`source_id`)"
    ") ENGINE=InnoDB")
TABLES["fact_network_fuel_efficiency_agg"] = (
    "CREATE TABLE IF NOT EXISTS `fact_network_fuel_efficiency_agg` ("
    "`source_id` char(37),"
    "`distribution_model_id` varchar(100),"
    "`efficiency_meters_per_kilojoules` float,"
    "`efficiency_miles_per_gallon` float,"
    "`efficiency_meters_per_kilojoules_local` float,"
    "`efficiency_miles_per_gallon_local` float,"
    "PRIMARY KEY (`source_id`, `distribution_model_id`)"
    ") ENGINE=InnoDB")
TABLES["leaderboard_chart"] = (
    "CREATE TABLE IF NOT EXISTS `leaderboard_chart` ("
    "`source_id` char(37),"
    "`throughput_per_hour` float,"
    "`avg_instantaneous_speed` float,"
    "`avg_network_speed` float,"
    "`total_vmt` float,"
    "`lane_changes_per_vehicle` float,"
    "`space_gap_min` float,"
    "`space_gap_max` float,"
    "`space_gap_avg` float,"
    "`space_gap_stddev` float,"
    "`av_space_gap_min` float,"
    "`av_space_gap_max` float,"
    "`av_space_gap_avg` float,"
    "`av_space_gap_stddev` float,"
    "`time_gap_min` float,"
    "`time_gap_max` float,"
    "`time_gap_avg` float,"
    "`time_gap_stddev` float,"
    "`av_time_gap_min` float,"
    "`av_time_gap_max` float,"
    "`av_time_gap_avg` float,"
    "`av_time_gap_stddev` float,"
    "`speed_min` float,"
    "`speed_max` float,"
    "`speed_avg` float,"
    "`speed_stddev` float,"
    "`accel_min` float,"
    "`accel_max` float,"
    "`accel_avg` float,"
    "`accel_stddev` float,"
    "`safety_rate` float,"
    "`safety_value_max` float,"
    "`prius_percent_infeasible` float,"
    "`tacoma_percent_infeasible` float,"
    "`midsize_sedan_percent_infeasible` float,"
    "`midsize_suv_percent_infeasible` float,"
    "`distribution_v0_percent_infeasible` float,"
    "`rav4_percent_infeasible` float,"
    "`prius_efficiency_meters_per_kilojoules` float,"
    "`tacoma_efficiency_meters_per_kilojoules` float,"
    "`midsize_sedan_efficiency_meters_per_kilojoules` float,"
    "`midsize_suv_efficiency_meters_per_kilojoules` float,"
    "`distribution_v0_efficiency_meters_per_kilojoules` float,"
    "`rav4_efficiency_meters_per_kilojoules` float,"
    "`prius_efficiency_miles_per_gallon` float,"
    "`tacoma_efficiency_miles_per_gallon` float,"
    "`midsize_sedan_efficiency_miles_per_gallon` float,"
    "`midsize_suv_efficiency_miles_per_gallon` float,"
    "`distribution_v0_efficiency_miles_per_gallon` float,"
    "`rav4_efficiency_miles_per_gallon` float,"
    "`prius_efficiency_miles_per_gallon_local` float,"
    "`tacoma_efficiency_miles_per_gallon_local` float,"
    "`midsize_sedan_efficiency_miles_per_gallon_local` float,"
    "`midsize_suv_efficiency_miles_per_gallon_local` float,"
    "`distribution_v0_efficiency_miles_per_gallon_local` float,"
    "`rav4_efficiency_miles_per_gallon_local` float,"
    "PRIMARY KEY (`source_id`)"
    ") ENGINE=InnoDB")
TABLES["leaderboard_chart_agg"] = (
    "CREATE TABLE IF NOT EXISTS `leaderboard_chart_agg` ("
    "`submission_date` date,"
    "`source_id` char(37),"
    "`submitter_name` varchar(50),"
    "`strategy` varchar(50),"
    "`network` varchar(50),"
    "`is_baseline` char(5),"
    "`is_benchmark` char(5),"
    "`tacoma_efficiency_miles_per_gallon` float,"
    "`prius_efficiency_miles_per_gallon` float,"
    "`midsize_sedan_efficiency_miles_per_gallon` float,"
    "`midsize_suv_efficiency_miles_per_gallon` float,"
    "`distribution_v0_efficiency_miles_per_gallon` float,"
    "`rav4_efficiency_miles_per_gallon` float,"
    "`efficiency` varchar(50),"
    "`tacoma_efficiency_miles_per_gallon_local` float,"
    "`prius_efficiency_miles_per_gallon_local` float,"
    "`midsize_sedan_efficiency_miles_per_gallon_local` float,"
    "`midsize_suv_efficiency_miles_per_gallon_local` float,"
    "`distribution_v0_efficiency_miles_per_gallon_local` float,"
    "`rav4_efficiency_miles_per_gallon_local` float,"
    "`efficiency_local` varchar(50),"
    "`percent_infeasible` varchar(50),"
    "`inflow` varchar(50),"
    "`speed` varchar(50),"
    "`vmt` varchar(50),"
    "`lane_changes` varchar(50),"
    "`space_gap` varchar(50),"
    "`av_space_gap` varchar(50),"
    "`time_gap` varchar(50),"
    "`av_time_gap` varchar(50),"
    "`speed_stddev` varchar(50),"
    "`accel_stddev` varchar(50),"
    "`safety_rate` float,"
    "`safety_value_max` float,"
    "PRIMARY KEY (`source_id`)"
    ") ENGINE=InnoDB")
TABLES["fact_top_scores"] = (
    "CREATE TABLE IF NOT EXISTS `fact_top_scores` ("
    "`network` varchar(50),"
    "`submission_date` date,"
    "`tacoma_max_score` float,"
    "`prius_max_score` float,"
    "`midsize_sedan_max_score` float,"
    "`midsize_suv_max_score` float,"
    "`distribution_v0_max_score` float,"
    "`rav4_max_score` float,"
    "PRIMARY KEY (`network`)"
    ") ENGINE=InnoDB")
TABLES["metadata_table"] = (
    "CREATE TABLE IF NOT EXISTS `metadata_table` ("
    "`source_id` char(37),"
    "`submission_date` date,"
    "`network` varchar(50),"
    "`is_baseline` char(5),"
    "`submitter_name` varchar(50),"
    "`strategy` varchar(50),"
    "`version` varchar(10),"
    "`on_ramp` char(5),"
    "`penetration_rate` varchar(10),"
    "`road_grade` char(5),"
    "`is_benchmark` char(5),"
    "`controller` varchar(50),"
    "`transfer_test_name` varchar(50),"
    "`transfer_test_parameter` varchar(50),"
    "PRIMARY KEY (`source_id`)"
    ") ENGINE=InnoDB")
TABLES["fact_av_trace"] = (
    "CREATE TABLE IF NOT EXISTS `fact_av_trace` ("
    "`source_id` char(37),"
    "`time_step` float,"
    "`id` varchar(50),"
    "`is_av` char(5),"
    "`position` float,"
    "`speed` float,"
    "`acceleration` float,"
    "`space_gap` float,"
    "`time_gap` float,"
    "`fs_region` int,"
    "PRIMARY KEY (`source_id`, `time_step`, `id`)"
    ") ENGINE=InnoDB")
TABLES["baseline_table"] = (
    "CREATE TABLE IF NOT EXISTS `baseline_table` ("
    "`network` varchar(50),"
    "`version` varchar(50),"
    "`on_ramp` char(5),"
    "`road_grade` char(5),"
    "`source_id` char(37),"
    "PRIMARY KEY (`network`, `version`, `on_ramp`, `road_grade`)"
    ") ENGINE=InnoDB")
TABLES["fact_safety_matrix"] = (
    "CREATE TABLE IF NOT EXISTS `fact_safety_matrix` ("
    "`headway_lower` float,"
    "`headway_upper` float,"
    "`rel_speed_lower` float,"
    "`rel_speed_upper` float,"
    "`value_lower_left` float,"
    "`value_lower_right` float,"
    "`value_upper_left` float,"
    "`value_upper_right` float,"
    "PRIMARY KEY (`headway_lower`, `headway_upper`,"
    "             `rel_speed_lower`, `rel_speed_upper`,"
    "             `value_lower_left`, `value_lower_right`,"
    "             `value_upper_left`, `value_upper_right`)"
    ") ENGINE=InnoDB")
TABLES["fact_space_gaps_binned"] = (
    "CREATE TABLE IF NOT EXISTS `fact_space_gaps_binned` ("
    "`source_id` char(37),"
    "`safety_value_bin` varchar(30),"
    "`space_gap_count` int,"
    "`av_space_gap_count` int,"
    "PRIMARY KEY (`source_id`)"
    ") ENGINE=InnoDB")
TABLES["fact_time_gaps_binned"] = (
    "CREATE TABLE IF NOT EXISTS `fact_time_gaps_binned` ("
    "`source_id` char(37),"
    "`safety_value_bin` varchar(30),"
    "`time_gap_count` int,"
    "`av_time_gap_count` int,"
    "PRIMARY KEY (`source_id`)"
    ") ENGINE=InnoDB")
TABLES["fact_vehicle_distributions"] = (
    "CREATE TABLE IF NOT EXISTS `fact_vehicle_distributions` ("
    "`rank` int,"
    "`all_tacoma_fit` varchar(50),"
    "`all_prius_ev_fit` varchar(50),"
    "`fifty_fifty_tacoma_prius` varchar(50),"
    "`all_midsize_sedan` varchar(50),"
    "`all_midsize_suv` varchar(50),"
    "`distribution_v0` varchar(50),"
    "`all_rav4` varchar(50),"
    "PRIMARY KEY (`rank`)"
    ") ENGINE=InnoDB")


def create_tables(cursor):

    # Check that the database exist and it is in use.
    try:
        cursor.execute("USE {}".format(DB_NAME))
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database {} does not exist.".format(DB_NAME))
        else:
            print(err)

    # Create tables if not already exist
    for table_name, table_description in TABLES.items():
        try:
            cursor.execute(table_description)
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                print("{} already exists.".format(table_name))
            else:
                print(err.msg)
        else:
            print("table {} create successfully.".format(table_name))


if __name__ == '__main__':

    # Create table if does not already exist
    cnx = connect(DB_NAME)
    cursor = cnx.cursor()

    create_tables(cursor)
    # for t in TABLES.keys():
    #     cursor.execute("Drop table if exists {};".format(t))

    cursor.close()
    cnx.close()
