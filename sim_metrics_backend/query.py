"""stores all the pre-defined query strings."""
from collections import defaultdict
from enum import Enum
import numpy as np
import os
from scipy.io import loadmat

# tags for different queries
prerequisites = {
    "FACT_AV_TRACE": (
        "fact_av_trace", {"FACT_VEHICLE_TRACE"}
    ),
    "TACOMA_FIT_DENOISED_ACCEL": (
        "fact_energy_trace", {"FACT_VEHICLE_TRACE"}
    ),
    "PRIUS_FIT_DENOISED_ACCEL": (
        "fact_energy_trace", {"FACT_VEHICLE_TRACE"}
    ),
    "COMPACT_SEDAN_FIT_DENOISED_ACCEL": (
        "fact_energy_trace", {"FACT_VEHICLE_TRACE"}
    ),
    "MIDSIZE_SEDAN_FIT_DENOISED_ACCEL": (
        "fact_energy_trace", {"FACT_VEHICLE_TRACE"}
    ),
    "RAV4_2019_FIT_DENOISED_ACCEL": (
        "fact_energy_trace", {"FACT_VEHICLE_TRACE"}
    ),
    "MIDSIZE_SUV_FIT_DENOISED_ACCEL": (
        "fact_energy_trace", {"FACT_VEHICLE_TRACE"}
    ),
    "LIGHT_DUTY_PICKUP_FIT_DENOISED_ACCEL": (
        "fact_energy_trace", {"FACT_VEHICLE_TRACE"}
    ),
    "CLASS3_PND_TRUCK_FIT_DENOISED_ACCEL": (
        "fact_energy_trace", {"FACT_VEHICLE_TRACE"}
    ),
    "CLASS8_TRACTOR_TRAILER_FIT_DENOISED_ACCEL": (
        "fact_energy_trace", {"FACT_VEHICLE_TRACE"}
    ),
    "FACT_SAFETY_METRICS_2D": (
        "fact_safety_metrics", {"FACT_VEHICLE_TRACE"}
    ),
    "FACT_SAFETY_METRICS_3D": (
        "fact_safety_metrics", {"FACT_VEHICLE_TRACE"}
    ),
    "FACT_NETWORK_THROUGHPUT_AGG": (
        "fact_network_throughput_agg", {"FACT_VEHICLE_TRACE"}
    ),
    "FACT_NETWORK_INFLOWS_OUTFLOWS": (
        "fact_network_inflows_outflows", {"FACT_VEHICLE_TRACE"}
    ),
    "FACT_NETWORK_SPEED": (
        "fact_network_speed", {"FACT_VEHICLE_TRACE"}
    ),
    "FACT_VEHICLE_METRICS": (
        "fact_vehicle_metrics", {"FACT_VEHICLE_TRACE"}
    ),
    "FACT_SPACE_GAPS_BINNED": (
        "fact_space_gaps_binned", {"FACT_VEHICLE_TRACE"}
    ),
    "FACT_TIME_GAPS_BINNED": (
        "fact_time_gaps_binned", {"FACT_VEHICLE_TRACE"}
    ),
    "FACT_VEHICLE_COUNTS_BY_TIME": (
        "fact_vehicle_counts_by_time", {"FACT_VEHICLE_TRACE"}
    ),
    "FACT_FOLLOWERSTOPPER_ENVELOPE": (
        "fact_followerstopper_envelope", {"FACT_VEHICLE_TRACE"}
    ),
    "FACT_VEHICLE_FUEL_EFFICIENCY_AGG": (
        "fact_vehicle_fuel_efficiency_agg", {"FACT_VEHICLE_TRACE",
                                             "TACOMA_FIT_DENOISED_ACCEL",
                                             "PRIUS_FIT_DENOISED_ACCEL",
                                             "COMPACT_SEDAN_FIT_DENOISED_ACCEL",
                                             "MIDSIZE_SEDAN_FIT_DENOISED_ACCEL",
                                             "RAV4_2019_FIT_DENOISED_ACCEL",
                                             "MIDSIZE_SUV_FIT_DENOISED_ACCEL",
                                             "LIGHT_DUTY_PICKUP_FIT_DENOISED_ACCEL",
                                             "CLASS3_PND_TRUCK_FIT_DENOISED_ACCEL",
                                             "CLASS8_TRACTOR_TRAILER_FIT_DENOISED_ACCEL"}
    ),
    "FACT_NETWORK_METRICS_BY_DISTANCE_AGG": (
        "fact_network_metrics_by_distance_agg", {"FACT_VEHICLE_TRACE",
                                                 "TACOMA_FIT_DENOISED_ACCEL",
                                                 "PRIUS_FIT_DENOISED_ACCEL",
                                                 "COMPACT_SEDAN_FIT_DENOISED_ACCEL",
                                                 "MIDSIZE_SEDAN_FIT_DENOISED_ACCEL",
                                                 "RAV4_2019_FIT_DENOISED_ACCEL",
                                                 "MIDSIZE_SUV_FIT_DENOISED_ACCEL",
                                                 "LIGHT_DUTY_PICKUP_FIT_DENOISED_ACCEL",
                                                 "CLASS3_PND_TRUCK_FIT_DENOISED_ACCEL",
                                                 "CLASS8_TRACTOR_TRAILER_FIT_DENOISED_ACCEL"}
    ),
    "FACT_NETWORK_METRICS_BY_TIME_AGG": (
        "fact_network_metrics_by_time_agg", {"FACT_VEHICLE_TRACE",
                                             "TACOMA_FIT_DENOISED_ACCEL",
                                             "PRIUS_FIT_DENOISED_ACCEL",
                                             "COMPACT_SEDAN_FIT_DENOISED_ACCEL",
                                             "MIDSIZE_SEDAN_FIT_DENOISED_ACCEL",
                                             "RAV4_2019_FIT_DENOISED_ACCEL",
                                             "MIDSIZE_SUV_FIT_DENOISED_ACCEL",
                                             "LIGHT_DUTY_PICKUP_FIT_DENOISED_ACCEL",
                                             "CLASS3_PND_TRUCK_FIT_DENOISED_ACCEL",
                                             "CLASS8_TRACTOR_TRAILER_FIT_DENOISED_ACCEL"}
    ),
    "FACT_INFEASIBLE_FLAGS": (
        "fact_infeasible_flags", {"TACOMA_FIT_DENOISED_ACCEL",
                                  "PRIUS_FIT_DENOISED_ACCEL",
                                  "COMPACT_SEDAN_FIT_DENOISED_ACCEL",
                                  "MIDSIZE_SEDAN_FIT_DENOISED_ACCEL",
                                  "RAV4_2019_FIT_DENOISED_ACCEL",
                                  "MIDSIZE_SUV_FIT_DENOISED_ACCEL",
                                  "LIGHT_DUTY_PICKUP_FIT_DENOISED_ACCEL",
                                  "CLASS3_PND_TRUCK_FIT_DENOISED_ACCEL",
                                  "CLASS8_TRACTOR_TRAILER_FIT_DENOISED_ACCEL"}
    ),
    "FACT_VEHICLE_FUEL_EFFICIENCY_BINNED": (
        "fact_vehicle_fuel_efficiency_binned", {
            "FACT_VEHICLE_FUEL_EFFICIENCY_AGG"}
    ),
    "FACT_NETWORK_FUEL_EFFICIENCY_AGG": (
        "fact_network_fuel_efficiency_agg", {
            "FACT_VEHICLE_FUEL_EFFICIENCY_AGG"}
    ),
    "FACT_SAFETY_METRICS_AGG": (
        "fact_safety_metrics_agg", {"FACT_SAFETY_METRICS_3D"}
    ),
    "FACT_SAFETY_METRICS_BINNED": (
        "fact_safety_metrics_binned", {"FACT_SAFETY_METRICS_3D"}
    ),
    "LEADERBOARD_CHART": (
        "leaderboard_chart", {"FACT_NETWORK_THROUGHPUT_AGG",
                              "FACT_NETWORK_SPEED",
                              "FACT_VEHICLE_METRICS",
                              "FACT_NETWORK_FUEL_EFFICIENCY_AGG",
                              "FACT_SAFETY_METRICS_AGG",
                              "FACT_INFEASIBLE_FLAGS"}
    ),
    "LEADERBOARD_CHART_AGG": (
        "leaderboard_chart_agg", {"LEADERBOARD_CHART"}
    ),
    "FACT_TOP_SCORES": (
        "fact_top_scores", {"LEADERBOARD_CHART_AGG"}
    ),
}

triggers = [
    "FACT_VEHICLE_TRACE",
    "TACOMA_FIT_DENOISED_ACCEL",
    "PRIUS_FIT_DENOISED_ACCEL",
    "COMPACT_SEDAN_FIT_DENOISED_ACCEL",
    "MIDSIZE_SEDAN_FIT_DENOISED_ACCEL",
    "RAV4_2019_FIT_DENOISED_ACCEL",
    "MIDSIZE_SUV_FIT_DENOISED_ACCEL",
    "LIGHT_DUTY_PICKUP_FIT_DENOISED_ACCEL",
    "CLASS3_PND_TRUCK_FIT_DENOISED_ACCEL",
    "CLASS8_TRACTOR_TRAILER_FIT_DENOISED_ACCEL",
    "FACT_INFEASIBLE_FLAGS",
    "FACT_VEHICLE_FUEL_EFFICIENCY_AGG",
    "FACT_SAFETY_METRICS_3D",
    "FACT_NETWORK_THROUGHPUT_AGG",
    "FACT_NETWORK_SPEED",
    "FACT_VEHICLE_METRICS",
    "FACT_NETWORK_FUEL_EFFICIENCY_AGG",
    "FACT_SAFETY_METRICS_AGG",
    "LEADERBOARD_CHART",
    "LEADERBOARD_CHART_AGG"
]

tables = [
    "fact_vehicle_trace",
    "fact_av_trace",
    "fact_energy_trace",
    "fact_infeasible_flags",
    "fact_vehicle_counts_by_time",
    "fact_safety_metrics",
    "fact_safety_metrics_agg",
    "fact_safety_metrics_binned",
    "fact_network_throughput_agg",
    "fact_network_inflows_outflows",
    "fact_network_speed",
    "fact_vehicle_metrics",
    "fact_space_gaps_binned",
    "fact_time_gaps_binned",
    "fact_followerstopper_envelope",
    "fact_vehicle_fuel_efficiency_agg",
    "fact_vehicle_fuel_efficiency_binned",
    "fact_network_metrics_by_distance_agg",
    "fact_network_metrics_by_time_agg",
    "fact_network_fuel_efficiency_agg",
    "leaderboard_chart",
    "leaderboard_chart_agg",
    "fact_top_scores",
    "metadata_table"
]

summary_tables = ["leaderboard_chart_agg", "fact_top_scores"]

network_filters = defaultdict(lambda: {
    'inflow_filter': "x > 500",
    'outflow_filter': "x < 2300",
    'warmup_steps': 500 * 3 * 0.4
})
network_filters['I-210 without Ramps'] = {
    'inflow_filter': "edge_id != 'ghost0'",
    'outflow_filter': "edge_id != '119257908#3'",
    'warmup_steps': 600 * 3 * 0.4
}
network_filters['I-210'] = {
    'inflow_filter': "edge_id != 'ghost0'",
    'outflow_filter': "edge_id != '119257908#3'",
    'warmup_steps': 600 * 3 * 0.4
}
network_filters['I-24_subnetwork'] = {
    'inflow_filter': "edge_id NOT IN ('Eastbound_4', 'Westbound_2')",
    'outflow_filter': "edge_id NOT IN ('Eastbound_8', 'Eastbound_Off_2', 'Westbound_7')",
    'warmup_steps': 10500 * 0.2
}
network_filters['Single-Lane Trajectory'] = {
    'inflow_filter': "source_id IS NOT NULL",
    'outflow_filter': "source_id IS NOT NULL",
    'warmup_steps': 0
}

max_decel = -1.0
leader_max_decel = -2.0

# 42.36 kW = 1g/s
gasoline_g_to_joules = 42360
gasoline_galperhr_to_grampersec = 1.268
# 42.47 kW = 1g/s
diesel_g_to_joules = 42470
diesel_galperhr_to_grampersec = 1.119

DIR_PATH = os.path.join(os.path.dirname(os.path.realpath(
    __file__)), "./energy_models/model_coefficients")


def load_coeffs(filename, mass, conversion=33.43e3, v_max_fit=40):
    """Load in model coefficients from MATLAB files."""
    mat = loadmat(os.path.join(DIR_PATH, filename))
    mat = {key: val.item() for key, val in mat.items()
           if isinstance(val, np.ndarray)}
    mat['mass'] = mass
    mat['conversion'] = conversion
    mat['v_max_fit'] = v_max_fit
    return mat


COMPACT_SEDAN_COEFFS = load_coeffs("Compact_coeffs.mat", mass=1450)
MIDSIZE_SEDAN_COEFFS = load_coeffs("midBase_coeffs.mat", mass=1743)
RAV4_2019_COEFFS = load_coeffs("RAV4_coeffs.mat", mass=1717)
MIDSIZE_SUV_COEFFS = load_coeffs("midSUV_coeffs.mat", mass=1897)
LIGHT_DUTY_PICKUP_COEFFS = load_coeffs("Pickup_coeffs.mat", mass=2173)
CLASS3_PND_TRUCK_COEFFS = load_coeffs("Class3PND_coeffs.mat", mass=5943)
CLASS8_TRACTOR_TRAILER_COEFFS = load_coeffs(
    "Class8Tractor_coeffs.mat", mass=25104)

FET_FIRST_LINES = """
    SELECT
        id,
        time_step,
        speed,
        acceleration,
        road_grade,
        edge_id,
        follower_id,
        distance,"""
FET_COEFFS_FORMAT = """
        GREATEST({coeffs[beta0]}, {coeffs[C0]} +
            {coeffs[C1]} * speed +
            {coeffs[C2]} * POW(speed,2) +
            {coeffs[C3]} * POW(speed,3) +
            {coeffs[p0]} * acceleration +
            {coeffs[p1]} * acceleration * speed +
            {coeffs[p2]} * acceleration * POW(speed,2) +
            {coeffs[q0]} * POW(GREATEST(acceleration,0),2) +
            {coeffs[q1]} * POW(GREATEST(acceleration,0),2) * speed
            ) * {g2J_conversion} AS power,
        GREATEST({coeffs[beta0]}, {coeffs[C0]} +
            {coeffs[C1]} * speed +
            {coeffs[C2]} * POW(speed,2) +
            {coeffs[C3]} * POW(speed,3) +
            {coeffs[p0]} * acceleration +
            {coeffs[p1]} * acceleration * speed +
            {coeffs[p2]} * acceleration * POW(speed,2) +
            {coeffs[q0]} * POW(GREATEST(acceleration,0),2) +
            {coeffs[q1]} * POW(GREATEST(acceleration,0),2) * speed
            ) AS fuel_rate_mass,
        GREATEST({coeffs[beta0]}, {coeffs[C0]} +
            {coeffs[C1]} * speed +
            {coeffs[C2]} * POW(speed,2) +
            {coeffs[C3]} * POW(speed,3) +
            {coeffs[p0]} * acceleration +
            {coeffs[p1]} * acceleration * speed +
            {coeffs[p2]} * acceleration * POW(speed,2) +
            {coeffs[q0]} * POW(GREATEST(acceleration,0),2) +
            {coeffs[q1]} * POW(GREATEST(acceleration,0),2) * speed
            ) * {galph2gps_conversion} AS fuel_rate_vol,
        '{energy_model_id}' AS energy_model_id,
        source_id,
        CASE
            WHEN speed > {coeffs[v_max_fit]} THEN 1
            WHEN acceleration > {coeffs[b1]} *
                POW(speed / {coeffs[v_max_fit]}, {coeffs[b2]}) * POW(1.0 - speed / {coeffs[v_max_fit]}, {coeffs[b3]})
                + {coeffs[b4]} * speed + {coeffs[b5]}
            THEN 1 ELSE 0 END AS infeasible_flag"""
FET_LAST_LINES = """
    FROM {}
    ORDER BY id, time_step
    """

TACOMA_FIT_FINAL_SELECT = FET_FIRST_LINES + """
        GREATEST(0, 2041 * acceleration * speed +
            3405.5481762 +
            83.12392997 * speed +
            6.7650718327 * POW(speed,2) +
            0.7041355229 * POW(speed,3)
            ) + GREATEST(0, 4598.7155 * acceleration + 975.12719 * acceleration * speed) AS power,
        (GREATEST(0, 2041 * acceleration * speed +
            3405.5481762 +
            83.12392997 * speed +
            6.7650718327 * POW(speed,2) +
            0.7041355229 * POW(speed,3)
            ) + GREATEST(0, 4598.7155 * acceleration +
            975.12719 * acceleration * speed)) / 42360 AS fuel_rate_mass,
        (GREATEST(0, 2041 * acceleration * speed +
            3405.5481762 +
            83.12392997 * speed +
            6.7650718327 * POW(speed,2) +
            0.7041355229 * POW(speed,3)
            ) + GREATEST(0, 4598.7155 * acceleration +
            975.12719 * acceleration * speed)) * 1.268 / 42360 AS fuel_rate_vol,
        'TACOMA_FIT_DENOISED_ACCEL' AS energy_model_id,
        source_id,
        0 AS infeasible_flag""" + FET_LAST_LINES

PRIUS_FIT_FINAL_SELECT = """
    , pmod_calculation AS (
        SELECT
            id,
            time_step,
            speed,
            acceleration,
            road_grade,
            edge_id,
            follower_id,
            distance,
            1663 * acceleration * speed +
                1.046 +
                119.166 * speed +
                0.337 * POW(speed,2) +
                0.383 * POW(speed,3) +
                GREATEST(0, 296.66 * acceleration * speed) AS p_mod,
            source_id
        FROM {}
    )""" + FET_FIRST_LINES + """
        GREATEST(p_mod, 0.869 * p_mod, -2338 * speed) AS power,
        GREATEST(p_mod, 0.869 * p_mod, -2338 * speed) / (1.268 * 33700) AS fuel_rate_mass,
        GREATEST(p_mod, 0.869 * p_mod, -2338 * speed) / 33700 AS fuel_rate_vol,
        'PRIUS_FIT_DENOISED_ACCEL' AS energy_model_id,
        source_id,
        0 AS infeasible_flag
    FROM pmod_calculation
    ORDER BY id, time_step
    """

COMPACT_SEDAN_FORMAT = FET_COEFFS_FORMAT.format(coeffs=COMPACT_SEDAN_COEFFS,
                                                g2J_conversion=gasoline_g_to_joules,
                                                galph2gps_conversion=gasoline_galperhr_to_grampersec,
                                                energy_model_id='COMPACT_SEDAN_FIT_DENOISED_ACCEL')
COMPACT_SEDAN_FIT_FINAL_SELECT = FET_FIRST_LINES + \
    COMPACT_SEDAN_FORMAT + FET_LAST_LINES

MIDSIZE_SEDAN_FORMAT = FET_COEFFS_FORMAT.format(coeffs=MIDSIZE_SEDAN_COEFFS,
                                                g2J_conversion=gasoline_g_to_joules,
                                                galph2gps_conversion=gasoline_galperhr_to_grampersec,
                                                energy_model_id='MIDSIZE_SEDAN_FIT_DENOISED_ACCEL')
MIDSIZE_SEDAN_FIT_FINAL_SELECT = FET_FIRST_LINES + \
    MIDSIZE_SEDAN_FORMAT + FET_LAST_LINES

RAV4_2019_FORMAT = FET_COEFFS_FORMAT.format(coeffs=RAV4_2019_COEFFS,
                                            g2J_conversion=gasoline_g_to_joules,
                                            galph2gps_conversion=gasoline_galperhr_to_grampersec,
                                            energy_model_id='RAV4_2019_FIT_DENOISED_ACCEL')
RAV4_2019_FIT_FINAL_SELECT = FET_FIRST_LINES + RAV4_2019_FORMAT + FET_LAST_LINES

MIDSIZE_SUV_FORMAT = FET_COEFFS_FORMAT.format(coeffs=MIDSIZE_SUV_COEFFS,
                                              g2J_conversion=gasoline_g_to_joules,
                                              galph2gps_conversion=gasoline_galperhr_to_grampersec,
                                              energy_model_id='MIDSIZE_SUV_FIT_DENOISED_ACCEL')
MIDSIZE_SUV_FIT_FINAL_SELECT = FET_FIRST_LINES + \
    MIDSIZE_SUV_FORMAT + FET_LAST_LINES

LIGHT_DUTY_PICKUP_FORMAT = FET_COEFFS_FORMAT.format(coeffs=LIGHT_DUTY_PICKUP_COEFFS,
                                                    g2J_conversion=gasoline_g_to_joules,
                                                    galph2gps_conversion=gasoline_galperhr_to_grampersec,
                                                    energy_model_id='LIGHT_DUTY_PICKUP_FIT_DENOISED_ACCEL')
LIGHT_DUTY_PICKUP_FIT_FINAL_SELECT = FET_FIRST_LINES + \
    LIGHT_DUTY_PICKUP_FORMAT + FET_LAST_LINES

CLASS3_PND_TRUCK_FORMAT = FET_COEFFS_FORMAT.format(coeffs=CLASS3_PND_TRUCK_COEFFS,
                                                   g2J_conversion=diesel_g_to_joules,
                                                   galph2gps_conversion=diesel_galperhr_to_grampersec,
                                                   energy_model_id='CLASS3_PND_TRUCK_FIT_DENOISED_ACCEL')
CLASS3_PND_TRUCK_FIT_FINAL_SELECT = FET_FIRST_LINES + \
    CLASS3_PND_TRUCK_FORMAT + FET_LAST_LINES

CLASS8_TRACTOR_TRAILER_FORMAT = FET_COEFFS_FORMAT.format(coeffs=CLASS8_TRACTOR_TRAILER_COEFFS,
                                                         g2J_conversion=diesel_g_to_joules,
                                                         galph2gps_conversion=diesel_galperhr_to_grampersec,
                                                         energy_model_id='CLASS8_TRACTOR_TRAILER_FIT_DENOISED_ACCEL')
CLASS8_TRACTOR_TRAILER_FIT_FINAL_SELECT = FET_FIRST_LINES + \
    CLASS8_TRACTOR_TRAILER_FORMAT + FET_LAST_LINES

DENOISED_ACCEL = """
    INSERT IGNORE INTO fact_energy_trace(id,time_step,speed,acceleration,road_grade,edge_id,follower_id,distance,power,fuel_rate_mass,fuel_rate_vol,energy_model_id,source_id,infeasible_flag)
    WITH denoised_accel_cte AS (
        SELECT
            id,
            time_step,
            speed,
            COALESCE (target_accel_no_noise_with_failsafe,
                      target_accel_no_noise_no_failsafe,
                      realized_accel) AS acceleration,
            road_grade,
            edge_id,
            follower_id,
            distance,
            source_id
        FROM fact_vehicle_trace
        WHERE 1 = 1
            AND source_id=\'{{partition}}\'
    )
    {};"""


class QueryStrings(Enum):
    """An enumeration of all the pre-defined query strings."""

    TACOMA_FIT_DENOISED_ACCEL = \
        DENOISED_ACCEL.format(
            TACOMA_FIT_FINAL_SELECT.format('denoised_accel_cte'))

    PRIUS_FIT_DENOISED_ACCEL = \
        DENOISED_ACCEL.format(
            PRIUS_FIT_FINAL_SELECT.format('denoised_accel_cte'))

    COMPACT_SEDAN_FIT_DENOISED_ACCEL = \
        DENOISED_ACCEL.format(
            COMPACT_SEDAN_FIT_FINAL_SELECT.format('denoised_accel_cte'))

    MIDSIZE_SEDAN_FIT_DENOISED_ACCEL = \
        DENOISED_ACCEL.format(
            MIDSIZE_SEDAN_FIT_FINAL_SELECT.format('denoised_accel_cte'))

    RAV4_2019_FIT_DENOISED_ACCEL = \
        DENOISED_ACCEL.format(
            RAV4_2019_FIT_FINAL_SELECT.format('denoised_accel_cte'))

    MIDSIZE_SUV_FIT_DENOISED_ACCEL = \
        DENOISED_ACCEL.format(
            MIDSIZE_SUV_FIT_FINAL_SELECT.format('denoised_accel_cte'))

    LIGHT_DUTY_PICKUP_FIT_DENOISED_ACCEL = \
        DENOISED_ACCEL.format(
            LIGHT_DUTY_PICKUP_FIT_FINAL_SELECT.format('denoised_accel_cte'))

    CLASS3_PND_TRUCK_FIT_DENOISED_ACCEL = \
        DENOISED_ACCEL.format(
            CLASS3_PND_TRUCK_FIT_FINAL_SELECT.format('denoised_accel_cte'))

    CLASS8_TRACTOR_TRAILER_FIT_DENOISED_ACCEL = \
        DENOISED_ACCEL.format(
            CLASS8_TRACTOR_TRAILER_FIT_FINAL_SELECT.format('denoised_accel_cte'))

    FACT_AV_TRACE = """
        INSERT IGNORE INTO fact_av_trace(source_id, time_step, id, is_av, position, speed, acceleration, space_gap, time_gap, fs_region)
        WITH local_traces AS (
            SELECT
                *,
                SUBSTRING_INDEX(id, '_', 1) IN ('av', 'rl') AS is_av,
                SUBSTRING_INDEX(follower_id, '_', 1) IN ('av', 'rl') AS leads_av,
                4.5 + 1 / (2 * 1.5) * POW(LEAST(leader_rel_speed, 0), 2) + 0.4*speed AS dx_1,
                5.25 + 1 / (2 * 1.0) * POW(LEAST(leader_rel_speed, 0), 2) + 1.2*speed AS dx_2,
                6.0 + 1 / (2 * 0.5) * POW(LEAST(leader_rel_speed, 0), 2) + 1.8*speed AS dx_3
            FROM fact_vehicle_trace
            WHERE 1 = 1
                AND source_id = \'{partition}\'
                AND time_step >= {start_filter}
                AND {inflow_filter}
                AND {outflow_filter}
        )
        SELECT
            source_id,
            time_step,
            id,
            is_av,
            CASE
                WHEN is_av THEN distance
                ELSE distance - 4.5
            END AS position,
            speed,
            COALESCE (target_accel_no_noise_with_failsafe,
                      target_accel_no_noise_no_failsafe,
                      realized_accel) AS acceleration,
            headway AS space_gap,
            headway / speed AS time_gap,
            CASE
                WHEN headway > dx_3 THEN 4
                WHEN headway > dx_2 AND headway <= dx_3 THEN 3
                WHEN headway > dx_1 AND headway <= dx_2 THEN 2
                ELSE 1
            END AS fs_region
        FROM local_traces
        WHERE 1 = 1
            AND (is_av OR leads_av)
    ;"""

    FACT_SAFETY_METRICS_2D = """
        INSERT IGNORE INTO fact_safety_metrics(source_id, safety_model, time_step, id, safety_value)
        WITH sub_vehicle_trace AS (
            SELECT
                source_id,
                id,
                time_step,
                headway,
                leader_rel_speed
            FROM fact_vehicle_trace
            WHERE 1 = 1
                AND source_id = \'{partition}\'
                AND time_step >= {start_filter}
                AND {inflow_filter}
                AND {outflow_filter}
            ORDER BY headway, leader_rel_speed
        )
        SELECT
            vt.source_id,
            'v2D_HJI' AS safety_model,
            vt.time_step,
            vt.id,
            COALESCE((
                value_lower_left*(headway_upper-headway)*(rel_speed_upper-leader_rel_speed) +
                value_lower_right*(headway-headway_lower)*(rel_speed_upper-leader_rel_speed) +
                value_upper_left*(headway_upper-headway)*(leader_rel_speed-rel_speed_lower) +
                value_upper_right*(headway-headway_lower)*(leader_rel_speed-rel_speed_lower)
            ) / ((headway_upper-headway_lower)*(rel_speed_upper-rel_speed_lower)), 200.0) AS safety_value
        FROM sub_vehicle_trace vt
        LEFT OUTER JOIN fact_safety_matrix sm ON 1 = 1
            AND vt.leader_rel_speed BETWEEN sm.rel_speed_lower AND sm.rel_speed_upper
            AND vt.headway BETWEEN sm.headway_lower AND sm.headway_upper
        ;
    """

    FACT_SAFETY_METRICS_3D = """
        INSERT IGNORE INTO fact_safety_metrics(source_id, safety_model, time_step, id, safety_value)
        SELECT
            source_id,
            'v3D' AS safety_model,
            time_step,
            id,
            COALESCE(headway, 1000) + (CASE
                WHEN -speed/{max_decel} > -(speed+COALESCE(leader_rel_speed, 0))/{leader_max_decel} THEN
                    -0.5*POW(COALESCE(leader_rel_speed, 0), 2)/{leader_max_decel} +
                    -0.5*POW(speed,2)/{leader_max_decel} +
                    -speed*COALESCE(leader_rel_speed, 0)/{leader_max_decel} +
                    0.5*POW(speed,2)/{max_decel}
                ELSE
                    -COALESCE(leader_rel_speed, 0)*speed/{max_decel} +
                    0.5*POW(speed,2)*{leader_max_decel}/POW({max_decel},2) +
                    -0.5*POW(speed,2)/{max_decel}
                END) AS safety_value
        FROM fact_vehicle_trace
        WHERE 1 = 1
            AND source_id = \'{partition}\'
            AND leader_id IS NOT NULL
            AND time_step >= {start_filter}
            AND {inflow_filter}
            AND {outflow_filter}
        ;
    """

    FACT_SAFETY_METRICS_AGG = """
        INSERT IGNORE INTO fact_safety_metrics_agg(source_id,safety_rate,safety_value_max)
        SELECT
            source_id,
            SUM(CASE WHEN safety_value > 0 THEN 1.0 ELSE 0.0 END) * 100.0 / COUNT(*) safety_rate,
            MIN(safety_value) AS safety_value_max
        FROM fact_safety_metrics
        WHERE 1 = 1
            AND source_id = \'{partition}\'
            AND safety_model = 'v3D'
        GROUP BY 1
        ;
    """

    FACT_SAFETY_METRICS_BINNED = """
        INSERT IGNORE INTO fact_safety_metrics_binned(source_id,safety_value_bin,count)
        WITH unfilter_bins AS (
            SELECT
                CAST(ROW_NUMBER() OVER() AS SIGNED) - 51 AS lb,
                CAST(ROW_NUMBER() OVER() AS SIGNED) - 50 AS ub
            FROM fact_safety_matrix
        ), bins AS (
            SELECT
                lb,
                ub
            FROM unfilter_bins
            WHERE 1=1
                AND lb >= -5
                AND ub <= 15
        )
        SELECT
            fsm.source_id,
            CONCAT('[', CAST(bins.lb AS CHAR(10)), ', ', CAST(bins.ub AS CHAR(10)), ')') AS safety_value_bin,
            COUNT(*) AS count
        FROM bins
        LEFT JOIN fact_safety_metrics fsm ON 1 = 1
            AND fsm.source_id = \'{partition}\'
            AND fsm.safety_value >= bins.lb
            AND fsm.safety_value < bins.ub
            AND fsm.safety_model = 'v3D'
        GROUP BY 1, 2
        ;
    """

    FACT_NETWORK_THROUGHPUT_AGG = """
        INSERT IGNORE INTO fact_network_throughput_agg(source_id,throughput_per_hour)
        WITH min_time AS (
            SELECT
                source_id,
                id,
                MIN(time_step) AS enter_time
            FROM fact_vehicle_trace
            WHERE 1 = 1
                AND source_id = \'{partition}\'
                AND {inflow_filter}
            GROUP BY 1, 2
        ), agg AS (
            SELECT
                source_id,
                COUNT(DISTINCT id) AS n_vehicles,
                MAX(enter_time) - MIN(enter_time) AS total_time_seconds
            FROM min_time
            WHERE 1 = 1
                AND enter_time >= {start_filter}
            GROUP BY 1
        )
        SELECT
            source_id,
            n_vehicles * 3600 / total_time_seconds AS throughput_per_hour
        FROM agg
        ;"""

    FACT_INFEASIBLE_FLAGS = """
        INSERT IGNORE INTO fact_infeasible_flags(source_id,distribution_model_id,percent_infeasible)
        WITH ranked_agg AS (
            SELECT
                *,
                RANK() OVER(PARTITION BY energy_model_id ORDER BY id) AS 'rank',
                SUBSTRING_INDEX(id, '_', 1) AS id_type
            FROM fact_energy_trace
            WHERE 1 = 1
                AND time_step >= {start_filter}
                AND {inflow_filter}
                AND {outflow_filter}
                AND source_id = \'{partition}\'
        ), dist_agg AS (
            SELECT
                agg.source_id,
                agg.id,
                agg.energy_model_id,
                agg.id_type,
                agg.infeasible_flag,
                dist.all_tacoma_fit,
                dist.all_prius_ev_fit,
                dist.all_midsize_sedan,
                dist.all_midsize_suv,
                dist.distribution_v0,
                dist.all_rav4
            FROM ranked_agg agg
            LEFT JOIN fact_vehicle_distributions dist ON 1 = 1
                AND agg.rank = dist.rank
            WHERE 1 = 1
                AND CASE agg.id_type
                        WHEN 'av' THEN agg.energy_model_id = 'RAV4_2019_FIT_DENOISED_ACCEL'
                        WHEN 'rl' THEN agg.energy_model_id = 'RAV4_2019_FIT_DENOISED_ACCEL'
                        WHEN 'truck' THEN agg.energy_model_id = 'CLASS8_TRACTOR_TRAILER_FIT_DENOISED_ACCEL'
                        ELSE TRUE
                    END
        )
        SELECT
            source_id,
            'ALL_TACOMA' AS distribution_model_id,
            100.0 * SUM(infeasible_flag) / COUNT(infeasible_flag) AS percent_infeasible
        FROM dist_agg
        WHERE 1 = 1
            AND (id_type IN ('av', 'rl', 'truck') OR energy_model_id = all_tacoma_fit)
        GROUP BY 1, 2
        UNION ALL
        SELECT
            source_id,
            'ALL_PRIUS_EV' AS distribution_model_id,
            100.0 * SUM(infeasible_flag) / COUNT(infeasible_flag) AS percent_infeasible
        FROM dist_agg
        WHERE 1 = 1
            AND (id_type IN ('av', 'rl', 'truck') OR energy_model_id = all_prius_ev_fit)
        GROUP BY 1, 2
        UNION ALL
        SELECT
            source_id,
            'ALL_MIDSIZE_SEDAN' AS distribution_model_id,
            100.0 * SUM(infeasible_flag) / COUNT(infeasible_flag) AS percent_infeasible
        FROM dist_agg
        WHERE 1 = 1
            AND (id_type IN ('av', 'rl', 'truck') OR energy_model_id = all_midsize_sedan)
        GROUP BY 1, 2
        UNION ALL
        SELECT
            source_id,
            'ALL_MIDSIZE_SUV' AS distribution_model_id,
            100.0 * SUM(infeasible_flag) / COUNT(infeasible_flag) AS percent_infeasible
        FROM dist_agg
        WHERE 1 = 1
            AND (id_type IN ('av', 'rl', 'truck') OR energy_model_id = all_midsize_suv)
        GROUP BY 1, 2
        UNION ALL
        SELECT
            source_id,
            'DISTRIBUTION_V0' AS distribution_model_id,
            100.0 * SUM(infeasible_flag) / COUNT(infeasible_flag) AS percent_infeasible
        FROM dist_agg
        WHERE 1 = 1
            AND (id_type IN ('av', 'rl', 'truck') OR energy_model_id = distribution_v0)
        GROUP BY 1, 2
        UNION ALL
        SELECT
            source_id,
            'ALL_RAV4' AS distribution_model_id,
            100.0 * SUM(infeasible_flag) / COUNT(infeasible_flag) AS percent_infeasible
        FROM dist_agg
        WHERE 1 = 1
            AND (id_type IN ('av', 'rl', 'truck') OR energy_model_id = all_rav4)
        GROUP BY 1, 2
        ;"""

    FACT_VEHICLE_FUEL_EFFICIENCY_AGG = """
        INSERT IGNORE INTO fact_vehicle_fuel_efficiency_agg(source_id,id,energy_model_id,distance_meters,energy_joules,fuel_grams,fuel_gallons,efficiency_meters_per_joules,efficiency_miles_per_gallon,is_locally_measurable)
        WITH sub_fact_energy_trace AS (
            SELECT
                id,
                source_id,
                energy_model_id,
                MAX(distance) - MIN(distance) AS distance_meters,
                (MAX(time_step) - MIN(time_step)) / (COUNT(DISTINCT time_step) - 1) AS time_step_size_seconds,
                SUM(power) AS power_watts,
                SUM(fuel_rate_mass) AS fuel_rate_g_per_s,
                SUM(fuel_rate_vol) AS fuel_rate_gal_per_hr,
                CASE
                    WHEN id LIKE 'av_%' OR id LIKE 'rl_%' OR
                         follower_id LIKE 'av_%' OR follower_id LIKE 'rl_%' THEN 1
                    ELSE 0
                END AS is_locally_measurable
            FROM fact_energy_trace
            WHERE  1 = 1
                AND time_step >= {start_filter}
                AND source_id = \'{partition}\'
                AND {inflow_filter}
                AND {outflow_filter}
            GROUP BY 1, 2, 3, 9
            HAVING 1 = 1
                AND MAX(distance) - MIN(distance) > 10
                AND COUNT(DISTINCT time_step) > 10
        )
        SELECT
            source_id,
            id,
            energy_model_id,
            distance_meters,
            power_watts * time_step_size_seconds AS energy_joules,
            fuel_rate_g_per_s * time_step_size_seconds AS fuel_grams,
            fuel_rate_gal_per_hr * time_step_size_seconds / 3600.0 AS fuel_gallons,
            distance_meters / (power_watts * time_step_size_seconds) AS efficiency_meters_per_joules,
            3600 / 1609.34 * distance_meters /
                (fuel_rate_gal_per_hr * time_step_size_seconds) AS efficiency_miles_per_gallon,
            is_locally_measurable
        FROM sub_fact_energy_trace
        WHERE 1 = 1
            AND power_watts * time_step_size_seconds != 0
        ;
    """

    FACT_VEHICLE_FUEL_EFFICIENCY_BINNED = """
        INSERT IGNORE INTO fact_vehicle_fuel_efficiency_binned(source_id,distribution_model_id,fuel_efficiency_bin,count)
        WITH unfilter_bins AS (
            SELECT
                ROW_NUMBER() OVER() - 1 AS lb,
                ROW_NUMBER() OVER() AS ub
            FROM fact_safety_matrix
        ), ranked_agg AS (
            SELECT
                *,
                RANK() OVER(PARTITION BY energy_model_id ORDER BY id) AS 'rank',
                SUBSTRING_INDEX(id, '_', 1) AS id_type
            FROM fact_vehicle_fuel_efficiency_agg
            WHERE 1 = 1
                AND source_id = \'{partition}\'
        ), dist_agg AS (
            SELECT
                agg.source_id,
                agg.id,
                agg.energy_model_id,
                agg.efficiency_miles_per_gallon,
                agg.id_type,
                dist.all_tacoma_fit,
                dist.all_prius_ev_fit,
                dist.all_midsize_sedan,
                dist.all_midsize_suv,
                dist.distribution_v0,
                dist.all_rav4
            FROM ranked_agg agg
            LEFT JOIN fact_vehicle_distributions dist ON 1 = 1
                AND agg.rank = dist.rank
        ), tacoma_binned AS (
            SELECT
                bins.lb*2 AS lb,
                bins.ub*2 AS ub,
                source_id,
                COUNT(*) AS count
            FROM unfilter_bins bins
            LEFT JOIN dist_agg ON 1 = 1
                AND dist_agg.efficiency_miles_per_gallon >= bins.lb*2
                AND dist_agg.efficiency_miles_per_gallon < bins.ub*2
                AND CASE dist_agg.id_type
                        WHEN 'av' THEN dist_agg.energy_model_id = 'RAV4_2019_FIT_DENOISED_ACCEL'
                        WHEN 'rl' THEN dist_agg.energy_model_id = 'RAV4_2019_FIT_DENOISED_ACCEL'
                        WHEN 'truck' THEN dist_agg.energy_model_id = 'CLASS8_TRACTOR_TRAILER_FIT_DENOISED_ACCEL'
                        ELSE dist_agg.energy_model_id = dist_agg.all_tacoma_fit
                    END
            GROUP BY 1, 2, 3
        ), prius_binned AS (
            SELECT
                bins.lb*2 AS lb,
                bins.ub*2 AS ub,
                source_id,
                COUNT(*) AS count
            FROM unfilter_bins bins
            LEFT JOIN dist_agg ON 1 = 1
                AND dist_agg.efficiency_miles_per_gallon >= bins.lb*2
                AND dist_agg.efficiency_miles_per_gallon < bins.ub*2
                AND CASE dist_agg.id_type
                        WHEN 'av' THEN dist_agg.energy_model_id = 'RAV4_2019_FIT_DENOISED_ACCEL'
                        WHEN 'rl' THEN dist_agg.energy_model_id = 'RAV4_2019_FIT_DENOISED_ACCEL'
                        WHEN 'truck' THEN dist_agg.energy_model_id = 'CLASS8_TRACTOR_TRAILER_FIT_DENOISED_ACCEL'
                        ELSE dist_agg.energy_model_id = dist_agg.all_prius_ev_fit
                    END
            GROUP BY 1, 2, 3
        ), midsize_sedan_binned AS (
            SELECT
                bins.lb*2 AS lb,
                bins.ub*2 AS ub,
                source_id,
                COUNT(*) AS count
            FROM unfilter_bins bins
            LEFT JOIN dist_agg ON 1 = 1
                AND dist_agg.efficiency_miles_per_gallon >= bins.lb*2
                AND dist_agg.efficiency_miles_per_gallon < bins.ub*2
                AND CASE dist_agg.id_type
                        WHEN 'av' THEN dist_agg.energy_model_id = 'RAV4_2019_FIT_DENOISED_ACCEL'
                        WHEN 'rl' THEN dist_agg.energy_model_id = 'RAV4_2019_FIT_DENOISED_ACCEL'
                        WHEN 'truck' THEN dist_agg.energy_model_id = 'CLASS8_TRACTOR_TRAILER_FIT_DENOISED_ACCEL'
                        ELSE dist_agg.energy_model_id = dist_agg.all_midsize_sedan
                    END
            GROUP BY 1, 2, 3
        ), midsize_suv_binned AS (
            SELECT
                bins.lb*2 AS lb,
                bins.ub*2 AS ub,
                source_id,
                COUNT(*) AS count
            FROM unfilter_bins bins
            LEFT JOIN dist_agg ON 1 = 1
                AND dist_agg.efficiency_miles_per_gallon >= bins.lb*2
                AND dist_agg.efficiency_miles_per_gallon < bins.ub*2
                AND CASE dist_agg.id_type
                        WHEN 'av' THEN dist_agg.energy_model_id = 'RAV4_2019_FIT_DENOISED_ACCEL'
                        WHEN 'rl' THEN dist_agg.energy_model_id = 'RAV4_2019_FIT_DENOISED_ACCEL'
                        WHEN 'truck' THEN dist_agg.energy_model_id = 'CLASS8_TRACTOR_TRAILER_FIT_DENOISED_ACCEL'
                        ELSE dist_agg.energy_model_id = dist_agg.all_midsize_suv
                    END
            GROUP BY 1, 2, 3
        ), distribution_v0_binned AS (
            SELECT
                bins.lb*2 AS lb,
                bins.ub*2 AS ub,
                source_id,
                COUNT(*) AS count
            FROM unfilter_bins bins
            LEFT JOIN dist_agg ON 1 = 1
                AND dist_agg.efficiency_miles_per_gallon >= bins.lb*2
                AND dist_agg.efficiency_miles_per_gallon < bins.ub*2
                AND CASE dist_agg.id_type
                        WHEN 'av' THEN dist_agg.energy_model_id = 'RAV4_2019_FIT_DENOISED_ACCEL'
                        WHEN 'rl' THEN dist_agg.energy_model_id = 'RAV4_2019_FIT_DENOISED_ACCEL'
                        WHEN 'truck' THEN dist_agg.energy_model_id = 'CLASS8_TRACTOR_TRAILER_FIT_DENOISED_ACCEL'
                        ELSE dist_agg.energy_model_id = dist_agg.distribution_v0
                    END
            GROUP BY 1, 2, 3
        ), rav4_binned AS (
            SELECT
                bins.lb*2 AS lb,
                bins.ub*2 AS ub,
                source_id,
                COUNT(*) AS count
            FROM unfilter_bins bins
            LEFT JOIN dist_agg ON 1 = 1
                AND dist_agg.efficiency_miles_per_gallon >= bins.lb*2
                AND dist_agg.efficiency_miles_per_gallon < bins.ub*2
                AND CASE dist_agg.id_type
                        WHEN 'av' THEN dist_agg.energy_model_id = 'RAV4_2019_FIT_DENOISED_ACCEL'
                        WHEN 'rl' THEN dist_agg.energy_model_id = 'RAV4_2019_FIT_DENOISED_ACCEL'
                        WHEN 'truck' THEN dist_agg.energy_model_id = 'CLASS8_TRACTOR_TRAILER_FIT_DENOISED_ACCEL'
                        ELSE dist_agg.energy_model_id = dist_agg.all_rav4
                    END
            GROUP BY 1, 2, 3
        ), tacoma_ratio_to_report AS (
            SELECT
                source_id,
                lb,
                ub,
                100.0 * count / (SUM(count) OVER()) AS count
            FROM tacoma_binned
            ORDER BY lb
        ), prius_ratio_to_report AS (
            SELECT
                source_id,
                lb,
                ub,
                100.0 * count / (SUM(count) OVER()) AS count
            FROM prius_binned
            ORDER BY lb
        ), midsize_sedan_ratio_to_report AS (
            SELECT
                source_id,
                lb,
                ub,
                100.0 * count / (SUM(count) OVER()) AS count
            FROM midsize_sedan_binned
            ORDER BY lb
        ), midsize_suv_ratio_to_report AS (
            SELECT
                source_id,
                lb,
                ub,
                100.0 * count / (SUM(count) OVER()) AS count
            FROM midsize_suv_binned
            ORDER BY lb
        ), distribution_v0_ratio_to_report AS (
            SELECT
                source_id,
                lb,
                ub,
                100.0 * count / (SUM(count) OVER()) AS count
            FROM distribution_v0_binned
        ), rav4_ratio_to_report AS (
            SELECT
                source_id,
                lb,
                ub,
                100.0 * count / (SUM(count) OVER()) AS count
            FROM rav4_binned
        )
        SELECT
            source_id,
            'ALL_TACOMA' AS distribution_model_id,
            CONCAT('[', CAST(lb AS CHAR(10)), ', ', CAST(ub AS CHAR(10)), ')') AS fuel_efficiency_bin,
            count
        FROM tacoma_ratio_to_report
        WHERE 1 = 1
            AND lb >= 0
            AND ub <= 100
        UNION ALL
        SELECT
            source_id,
            'ALL_PRIUS_EV' AS distribution_model_id,
            CONCAT('[', CAST(lb AS CHAR(10)), ', ', CAST(ub AS CHAR(10)), ')') AS fuel_efficiency_bin,
            count
        FROM prius_ratio_to_report
        WHERE 1 = 1
            AND lb >= 0
            AND ub <= 100
        UNION ALL
        SELECT
            source_id,
            'ALL_MIDSIZE_SEDAN' AS distribution_model_id,
            CONCAT('[', CAST(lb AS CHAR(10)), ', ', CAST(ub AS CHAR(10)), ')') AS fuel_efficiency_bin,
            count
        FROM midsize_sedan_ratio_to_report
        WHERE 1 = 1
            AND lb >= 0
            AND ub <= 100
        UNION ALL
        SELECT
            source_id,
            'ALL_MIDSIZE_SUV' AS distribution_model_id,
            CONCAT('[', CAST(lb AS CHAR(10)), ', ', CAST(ub AS CHAR(10)), ')') AS fuel_efficiency_bin,
            count
        FROM midsize_suv_ratio_to_report
        WHERE 1 = 1
            AND lb >= 0
            AND ub <= 100
        UNION ALL
        SELECT
            source_id,
            'DISTRIBUTION_V0' AS distribution_model_id,
            CONCAT('[', CAST(lb AS CHAR(10)), ', ', CAST(ub AS CHAR(10)), ')') AS fuel_efficiency_bin,
            count
        FROM distribution_v0_ratio_to_report
        WHERE 1 = 1
            AND lb >= 0
            AND ub <= 100
        UNION ALL
        SELECT
            source_id,
            'ALL_RAV4' AS distribution_model_id,
            CONCAT('[', CAST(lb AS CHAR(10)), ', ', CAST(ub AS CHAR(10)), ')') AS fuel_efficiency_bin,
            count
        FROM rav4_ratio_to_report
        WHERE 1 = 1
            AND lb >= 0
            AND ub <= 100
    ;"""

    FACT_NETWORK_FUEL_EFFICIENCY_AGG = """
        INSERT IGNORE INTO fact_network_fuel_efficiency_agg(source_id,distribution_model_id,efficiency_meters_per_kilojoules,efficiency_miles_per_gallon,efficiency_meters_per_kilojoules_local,efficiency_miles_per_gallon_local)
        WITH ranked_agg AS (
            SELECT
                *,
                RANK() OVER(PARTITION BY energy_model_id ORDER BY id) AS 'rank',
                SUBSTRING_INDEX(id, '_', 1) AS id_type
            FROM fact_vehicle_fuel_efficiency_agg
            WHERE 1 = 1
                AND source_id = \'{partition}\'
        ), dist_agg AS (
            SELECT
                agg.source_id,
                agg.id,
                agg.energy_model_id,
                agg.distance_meters,
                agg.energy_joules,
                agg.fuel_gallons,
                agg.id_type,
                agg.is_locally_measurable,
                dist.all_tacoma_fit,
                dist.all_prius_ev_fit,
                dist.all_midsize_sedan,
                dist.all_midsize_suv,
                dist.distribution_v0,
                dist.all_rav4
            FROM ranked_agg agg
            LEFT JOIN fact_vehicle_distributions dist ON 1 = 1
                AND agg.rank = dist.rank
        )
        SELECT
            source_id,
            'ALL_TACOMA' AS distribution_model_id,
            1000 * SUM(distance_meters) / SUM(energy_joules) AS efficiency_meters_per_kilojoules,
            SUM(distance_meters) / (1609.34 * SUM(fuel_gallons)) AS efficiency_miles_per_gallon,
            1000 * SUM(CASE WHEN is_locally_measurable = 1 THEN distance_meters ELSE 0 END) /
                SUM(CASE WHEN is_locally_measurable = 1 THEN energy_joules ELSE 0 END)
                AS efficiency_meters_per_kilojoules_local,
            SUM(CASE WHEN is_locally_measurable = 1 THEN distance_meters ELSE 0 END) / (1609.34 *
                SUM(CASE WHEN is_locally_measurable = 1 THEN fuel_gallons ELSE 0 END))
                AS efficiency_miles_per_gallon_local
        FROM dist_agg
        WHERE 1 = 1
            AND CASE dist_agg.id_type
                    WHEN 'av' THEN dist_agg.energy_model_id = 'RAV4_2019_FIT_DENOISED_ACCEL'
                    WHEN 'rl' THEN dist_agg.energy_model_id = 'RAV4_2019_FIT_DENOISED_ACCEL'
                    WHEN 'truck' THEN dist_agg.energy_model_id = 'CLASS8_TRACTOR_TRAILER_FIT_DENOISED_ACCEL'
                    ELSE energy_model_id = all_tacoma_fit
                END
        GROUP BY 1, 2
        HAVING 1 = 1
            AND SUM(energy_joules) != 0
        UNION ALL
        SELECT
            source_id,
            'ALL_PRIUS_EV' AS distribution_model_id,
            1000 * SUM(distance_meters) / SUM(energy_joules) AS efficiency_meters_per_kilojoules,
            SUM(distance_meters) / (1609.34 * SUM(fuel_gallons)) AS efficiency_miles_per_gallon,
            1000 * SUM(CASE WHEN is_locally_measurable = 1 THEN distance_meters ELSE 0 END) /
                SUM(CASE WHEN is_locally_measurable = 1 THEN energy_joules ELSE 0 END)
                AS efficiency_meters_per_kilojoules_local,
            SUM(CASE WHEN is_locally_measurable = 1 THEN distance_meters ELSE 0 END) / (1609.34 *
                SUM(CASE WHEN is_locally_measurable = 1 THEN fuel_gallons ELSE 0 END))
                AS efficiency_miles_per_gallon_local
        FROM dist_agg
        WHERE 1 = 1
            AND CASE dist_agg.id_type
                    WHEN 'av' THEN dist_agg.energy_model_id = 'RAV4_2019_FIT_DENOISED_ACCEL'
                    WHEN 'rl' THEN dist_agg.energy_model_id = 'RAV4_2019_FIT_DENOISED_ACCEL'
                    WHEN 'truck' THEN dist_agg.energy_model_id = 'CLASS8_TRACTOR_TRAILER_FIT_DENOISED_ACCEL'
                    ELSE energy_model_id = all_prius_ev_fit
                END
        GROUP BY 1, 2
        HAVING 1 = 1
            AND SUM(energy_joules) != 0
        UNION ALL
        SELECT
            source_id,
            'ALL_MIDSIZE_SEDAN' AS distribution_model_id,
            1000 * SUM(distance_meters) / SUM(energy_joules) AS efficiency_meters_per_kilojoules,
            SUM(distance_meters) / (1609.34 * SUM(fuel_gallons)) AS efficiency_miles_per_gallon,
            1000 * SUM(CASE WHEN is_locally_measurable = 1 THEN distance_meters ELSE 0 END) /
                SUM(CASE WHEN is_locally_measurable = 1 THEN energy_joules ELSE 0 END)
                AS efficiency_meters_per_kilojoules_local,
            SUM(CASE WHEN is_locally_measurable = 1 THEN distance_meters ELSE 0 END) / (1609.34 *
                SUM(CASE WHEN is_locally_measurable = 1 THEN fuel_gallons ELSE 0 END))
                AS efficiency_miles_per_gallon_local
        FROM dist_agg
        WHERE 1 = 1
            AND CASE dist_agg.id_type
                    WHEN 'av' THEN dist_agg.energy_model_id = 'RAV4_2019_FIT_DENOISED_ACCEL'
                    WHEN 'rl' THEN dist_agg.energy_model_id = 'RAV4_2019_FIT_DENOISED_ACCEL'
                    WHEN 'truck' THEN dist_agg.energy_model_id = 'CLASS8_TRACTOR_TRAILER_FIT_DENOISED_ACCEL'
                    ELSE energy_model_id = all_midsize_sedan
                END
        GROUP BY 1, 2
        HAVING 1 = 1
            AND SUM(energy_joules) != 0
        UNION ALL
        SELECT
            source_id,
            'ALL_MIDSIZE_SUV' AS distribution_model_id,
            1000 * SUM(distance_meters) / SUM(energy_joules) AS efficiency_meters_per_kilojoules,
            SUM(distance_meters) / (1609.34 * SUM(fuel_gallons)) AS efficiency_miles_per_gallon,
            1000 * SUM(CASE WHEN is_locally_measurable = 1 THEN distance_meters ELSE 0 END) /
                SUM(CASE WHEN is_locally_measurable = 1 THEN energy_joules ELSE 0 END)
                AS efficiency_meters_per_kilojoules_local,
            SUM(CASE WHEN is_locally_measurable = 1 THEN distance_meters ELSE 0 END) / (1609.34 *
                SUM(CASE WHEN is_locally_measurable = 1 THEN fuel_gallons ELSE 0 END))
                AS efficiency_miles_per_gallon_local
        FROM dist_agg
        WHERE 1 = 1
            AND CASE dist_agg.id_type
                    WHEN 'av' THEN dist_agg.energy_model_id = 'RAV4_2019_FIT_DENOISED_ACCEL'
                    WHEN 'rl' THEN dist_agg.energy_model_id = 'RAV4_2019_FIT_DENOISED_ACCEL'
                    WHEN 'truck' THEN dist_agg.energy_model_id = 'CLASS8_TRACTOR_TRAILER_FIT_DENOISED_ACCEL'
                    ELSE energy_model_id = all_midsize_suv
                END
        GROUP BY 1, 2
        HAVING 1 = 1
            AND SUM(energy_joules) != 0
        UNION ALL
        SELECT
            source_id,
            'DISTRIBUTION_V0' AS distribution_model_id,
            1000 * SUM(distance_meters) / SUM(energy_joules) AS efficiency_meters_per_kilojoules,
            SUM(distance_meters) / (1609.34 * SUM(fuel_gallons)) AS efficiency_miles_per_gallon,
            1000 * SUM(CASE WHEN is_locally_measurable = 1 THEN distance_meters ELSE 0 END) /
                SUM(CASE WHEN is_locally_measurable = 1 THEN energy_joules ELSE 0 END)
                AS efficiency_meters_per_kilojoules_local,
            SUM(CASE WHEN is_locally_measurable = 1 THEN distance_meters ELSE 0 END) / (1609.34 *
                SUM(CASE WHEN is_locally_measurable = 1 THEN fuel_gallons ELSE 0 END))
                AS efficiency_miles_per_gallon_local
        FROM dist_agg
        WHERE 1 = 1
            AND CASE dist_agg.id_type
                    WHEN 'av' THEN dist_agg.energy_model_id = 'RAV4_2019_FIT_DENOISED_ACCEL'
                    WHEN 'rl' THEN dist_agg.energy_model_id = 'RAV4_2019_FIT_DENOISED_ACCEL'
                    WHEN 'truck' THEN dist_agg.energy_model_id = 'CLASS8_TRACTOR_TRAILER_FIT_DENOISED_ACCEL'
                    ELSE energy_model_id = distribution_v0
                END
        GROUP BY 1, 2
        HAVING 1 = 1
            AND SUM(energy_joules) != 0
        UNION ALL
        SELECT
            source_id,
            'ALL_RAV4' AS distribution_model_id,
            1000 * SUM(distance_meters) / SUM(energy_joules) AS efficiency_meters_per_kilojoules,
            SUM(distance_meters) / (1609.34 * SUM(fuel_gallons)) AS efficiency_miles_per_gallon,
            1000 * SUM(CASE WHEN is_locally_measurable = 1 THEN distance_meters ELSE 0 END) /
                SUM(CASE WHEN is_locally_measurable = 1 THEN energy_joules ELSE 0 END)
                AS efficiency_meters_per_kilojoules_local,
            SUM(CASE WHEN is_locally_measurable = 1 THEN distance_meters ELSE 0 END) / (1609.34 *
                SUM(CASE WHEN is_locally_measurable = 1 THEN fuel_gallons ELSE 0 END))
                AS efficiency_miles_per_gallon_local
        FROM dist_agg
        WHERE 1 = 1
            AND CASE dist_agg.id_type
                    WHEN 'av' THEN dist_agg.energy_model_id = 'RAV4_2019_FIT_DENOISED_ACCEL'
                    WHEN 'rl' THEN dist_agg.energy_model_id = 'RAV4_2019_FIT_DENOISED_ACCEL'
                    WHEN 'truck' THEN dist_agg.energy_model_id = 'CLASS8_TRACTOR_TRAILER_FIT_DENOISED_ACCEL'
                    ELSE energy_model_id = all_rav4
                END
        GROUP BY 1, 2
        HAVING 1 = 1
            AND SUM(energy_joules) != 0
        ;"""

    FACT_NETWORK_SPEED = """
        INSERT IGNORE INTO fact_network_speed(source_id,avg_instantaneous_speed,avg_network_speed,total_vmt)
        WITH vehicle_agg AS (
            SELECT
                id,
                source_id,
                AVG(speed) AS vehicle_avg_speed,
                COUNT(DISTINCT time_step) AS n_steps,
                MAX(time_step) - MIN(time_step) AS time_delta,
                MAX(distance) - MIN(distance) AS distance_delta
            FROM fact_vehicle_trace
            WHERE 1 = 1
                AND source_id = \'{partition}\'
                AND {inflow_filter}
                AND {outflow_filter}
                AND time_step >= {start_filter}
            GROUP BY 1, 2
        )
        SELECT
            source_id,
            SUM(vehicle_avg_speed * n_steps) / SUM(n_steps) AS avg_instantaneous_speed,
            SUM(distance_delta) / SUM(time_delta) AS avg_network_speed,
            SUM(distance_delta) / 1609.344 AS total_vmt
        FROM vehicle_agg
        GROUP BY 1
    ;"""

    FACT_VEHICLE_METRICS = """
        INSERT IGNORE INTO fact_vehicle_metrics(source_id,lane_changes_per_vehicle,space_gap_min,space_gap_max,space_gap_avg,space_gap_stddev,av_space_gap_min,av_space_gap_max,av_space_gap_avg,av_space_gap_stddev,time_gap_min,time_gap_max,time_gap_avg,time_gap_stddev,av_time_gap_min,av_time_gap_max,av_time_gap_avg,av_time_gap_stddev,speed_min,speed_max,speed_avg,speed_stddev,accel_min,accel_max,accel_avg,accel_stddev)
        WITH cte AS (
            SELECT
                source_id,
                id,
                SUBSTRING_INDEX(id, '_', 1) IN ('av', 'rl') AS is_av,
                lane_id,
                LAG(lane_id) OVER (PARTITION BY id ORDER BY time_step) AS lag_lane_id,
                edge_id,
                LAG(edge_id) OVER (PARTITION BY id ORDER BY time_step) AS lag_edge_id,
                headway,
                speed,
                COALESCE (target_accel_no_noise_with_failsafe,
                    target_accel_no_noise_no_failsafe,
                    realized_accel) AS accel
            FROM fact_vehicle_trace
            WHERE 1 = 1
                AND source_id = \'{partition}\'
                AND {inflow_filter}
                AND {outflow_filter}
                AND time_step >= {start_filter}
        )
        SELECT
            source_id,
            SUM(CASE
                    WHEN lane_id != lag_lane_id AND edge_id = lag_edge_id THEN 1
                    ELSE 0
                END) * 100.0 / COUNT(DISTINCT id) AS lane_changes_per_vehicle,
            MIN(headway) AS space_gap_min,
            MAX(headway) AS space_gap_max,
            AVG(headway) AS space_gap_avg,
            STDDEV(headway) AS space_gap_stddev,
            MIN(CASE WHEN is_av THEN headway ELSE NULL END) AS av_space_gap_min,
            MAX(CASE WHEN is_av THEN headway ELSE NULL END) AS av_space_gap_max,
            AVG(CASE WHEN is_av THEN headway ELSE NULL END) AS av_space_gap_avg,
            STDDEV(CASE WHEN is_av THEN headway ELSE NULL END) AS av_space_gap_stddev,
            MIN(headway / speed) AS time_gap_min,
            MAX(headway / speed) AS time_gap_max,
            AVG(headway / speed) AS time_gap_avg,
            STDDEV(headway / speed) AS time_gap_stddev,
            MIN(CASE WHEN is_av THEN headway / speed ELSE NULL END) AS av_time_gap_min,
            MAX(CASE WHEN is_av THEN headway / speed ELSE NULL END) AS av_time_gap_max,
            AVG(CASE WHEN is_av THEN headway / speed ELSE NULL END) AS av_time_gap_avg,
            STDDEV(CASE WHEN is_av THEN headway / speed ELSE NULL END) AS av_time_gap_stddev,
            MIN(speed) AS speed_min,
            MAX(speed) AS speed_max,
            AVG(speed) AS speed_avg,
            STDDEV(speed) AS speed_stddev,
            MIN(accel) AS accel_min,
            MAX(accel) AS accel_max,
            AVG(accel) AS accel_avg,
            STDDEV(accel) AS accel_stddev
        FROM cte
        WHERE 1=1
            AND headway < 1000
        GROUP BY 1
    ;"""

    FACT_SPACE_GAPS_BINNED = """
        INSERT IGNORE INTO fact_space_gaps_binned(source_id,safety_value_bin,space_gap_count,av_space_gap_count)
        WITH unfilter_bins AS (
            SELECT
                ROW_NUMBER() OVER() AS lb,
                ROW_NUMBER() OVER() + 1 AS ub
            FROM fact_safety_matrix
        ), bins AS (
            SELECT
                lb,
                ub
            FROM unfilter_bins
            WHERE 1=1
                AND lb >= 0
                AND ub <= 200
        )
        SELECT
            vt.source_id,
            CAST(bins.lb AS CHAR(10)) AS safety_value_bin,
            COUNT(vt.headway) AS space_gap_count,
            SUM(CASE
                    WHEN
                        SUBSTRING_INDEX(vt.id, '_', 1) IN ('av', 'rl')
                    THEN 1
                    ELSE 0
                END) AS av_space_gap_count
        FROM bins
        LEFT JOIN fact_vehicle_trace vt ON 1 = 1
            AND vt.source_id = \'{partition}\'
            AND vt.headway BETWEEN bins.lb AND bins.ub
        GROUP BY 1, 2
    ;"""

    FACT_TIME_GAPS_BINNED = """
        INSERT IGNORE INTO fact_time_gaps_binned(source_id,safety_value_bin,time_gap_count,av_time_gap_count)
        WITH unfilter_bins AS (
            SELECT
                ROW_NUMBER() OVER() AS lb,
                ROW_NUMBER() OVER() + 1 AS ub
            FROM fact_safety_matrix
        ), bins AS (
            SELECT
                lb * 0.1 AS lb,
                ub * 0.1 AS ub
            FROM unfilter_bins
            WHERE 1=1
                AND lb >= 0
                AND ub <= 50
        )
        SELECT
            vt.source_id,
            CAST(bins.lb AS CHAR(10)) AS safety_value_bin,
            COUNT(vt.headway) AS time_gap_count,
            SUM(CASE
                    WHEN
                        SUBSTRING_INDEX(vt.id, '_', 1) IN ('av', 'rl')
                    THEN 1
                    ELSE 0
                END) AS av_time_gap_count
        FROM bins
        LEFT JOIN fact_vehicle_trace vt ON 1 = 1
            AND vt.source_id = \'{partition}\'
            AND vt.headway / vt.speed BETWEEN bins.lb AND bins.ub
        GROUP BY 1, 2
    ;"""

    FACT_FOLLOWERSTOPPER_ENVELOPE = """
        INSERT IGNORE INTO fact_followerstopper_envelope(source_id,region_1_proportion,region_2_proportion,region_3_proportion,region_4_proportion,av_region_1_proportion,av_region_2_proportion,av_region_3_proportion,av_region_4_proportion)
        WITH cte AS (
            SELECT
                source_id,
                SUBSTRING_INDEX(id, '_', 1) IN ('rl', 'av') AS is_av,
                4.5 + 1 / (2 * 1.5) * POW(LEAST(leader_rel_speed, 0), 2) + 0.4*speed AS dx_1,
                5.25 + 1 / (2 * 1.0) * POW(LEAST(leader_rel_speed, 0), 2) + 1.2*speed AS dx_2,
                6.0 + 1 / (2 * 0.5) * POW(LEAST(leader_rel_speed, 0), 2) + 1.8*speed AS dx_3,
                headway,
                time_step
            FROM fact_vehicle_trace
            WHERE 1 = 1
                AND source_id = \'{partition}\'
                AND {inflow_filter}
                AND {outflow_filter}
                AND time_step >= {start_filter}
        ), region_counts AS (
            SELECT
                source_id,
                SUM(CASE WHEN headway <= dx_1 THEN 1 ELSE 0 END) AS region_1_count,
                SUM(CASE WHEN headway > dx_1 AND headway <= dx_2 THEN 1 ELSE 0 END) AS region_2_count,
                SUM(CASE WHEN headway > dx_2 AND headway <= dx_3 THEN 1 ELSE 0 END) AS region_3_count,
                SUM(CASE WHEN headway > dx_3 THEN 1 ELSE 0 END) AS region_4_count,
                COUNT(*) AS total_count,
                SUM(CASE WHEN is_av AND headway <= dx_1 THEN 1 ELSE 0 END) AS av_region_1_count,
                SUM(CASE WHEN is_av AND headway > dx_1 AND headway <= dx_2 THEN 1 ELSE 0 END) AS av_region_2_count,
                SUM(CASE WHEN is_av AND headway > dx_2 AND headway <= dx_3 THEN 1 ELSE 0 END) AS av_region_3_count,
                SUM(CASE WHEN is_av AND headway > dx_3 THEN 1 ELSE 0 END) AS av_region_4_count,
                SUM(CASE WHEN is_av THEN 1 ELSE 0 END) AS av_count
            FROM cte
            GROUP BY 1
        )
        SELECT
            source_id,
            region_1_count * 100.0 / total_count AS region_1_proportion,
            region_2_count * 100.0 / total_count AS region_2_proportion,
            region_3_count * 100.0 / total_count AS region_3_proportion,
            region_4_count * 100.0 / total_count AS region_4_proportion,
            CASE av_count
                WHEN 0 THEN 0
                ELSE av_region_1_count * 100.0 / av_count
            END AS av_region_1_proportion,
            CASE av_count
                WHEN 0 THEN 0
                ELSE av_region_2_count * 100.0 / av_count
            END AS av_region_2_proportion,
            CASE av_count
                WHEN 0 THEN 0
                ELSE av_region_3_count * 100.0 / av_count
            END AS av_region_3_proportion,
            CASE av_count
                WHEN 0 THEN 0
                ELSE av_region_4_count * 100.0 / av_count
            END AS av_region_4_proportion
        FROM region_counts
    """

    LEADERBOARD_CHART = """
        INSERT IGNORE INTO leaderboard_chart(source_id,
                                            throughput_per_hour,
                                            avg_instantaneous_speed,
                                            avg_network_speed,
                                            total_vmt,
                                            lane_changes_per_vehicle,
                                            space_gap_min,
                                            space_gap_max,
                                            space_gap_avg,
                                            space_gap_stddev,
                                            av_space_gap_min,
                                            av_space_gap_max,
                                            av_space_gap_avg,
                                            av_space_gap_stddev,
                                            time_gap_min,
                                            time_gap_max,
                                            time_gap_avg,
                                            time_gap_stddev,
                                            av_time_gap_min,
                                            av_time_gap_max,
                                            av_time_gap_avg,
                                            av_time_gap_stddev,
                                            speed_min,
                                            speed_max,
                                            speed_avg,
                                            speed_stddev,
                                            accel_min,
                                            accel_max,
                                            accel_avg,
                                            accel_stddev,
                                            safety_rate,
                                            safety_value_max,
                                            prius_percent_infeasible,
                                            tacoma_percent_infeasible,
                                            midsize_sedan_percent_infeasible,
                                            midsize_suv_percent_infeasible,
                                            distribution_v0_percent_infeasible,
                                            rav4_percent_infeasible,
                                            prius_efficiency_meters_per_kilojoules,
                                            tacoma_efficiency_meters_per_kilojoules,
                                            midsize_sedan_efficiency_meters_per_kilojoules,
                                            midsize_suv_efficiency_meters_per_kilojoules,
                                            distribution_v0_efficiency_meters_per_kilojoules,
                                            rav4_efficiency_meters_per_kilojoules,
                                            prius_efficiency_miles_per_gallon,
                                            tacoma_efficiency_miles_per_gallon,
                                            midsize_sedan_efficiency_miles_per_gallon,
                                            midsize_suv_efficiency_miles_per_gallon,
                                            distribution_v0_efficiency_miles_per_gallon,
                                            rav4_efficiency_miles_per_gallon,
                                            prius_efficiency_miles_per_gallon_local,
                                            tacoma_efficiency_miles_per_gallon_local,
                                            midsize_sedan_efficiency_miles_per_gallon_local,
                                            midsize_suv_efficiency_miles_per_gallon_local,
                                            distribution_v0_efficiency_miles_per_gallon_local,
                                            rav4_efficiency_miles_per_gallon_local)
        SELECT
            nt.source_id,
            nt.throughput_per_hour,
            ns.avg_instantaneous_speed,
            ns.avg_network_speed,
            ns.total_vmt,
            vm.lane_changes_per_vehicle,
            vm.space_gap_min,
            vm.space_gap_max,
            vm.space_gap_avg,
            vm.space_gap_stddev,
            vm.av_space_gap_min,
            vm.av_space_gap_max,
            vm.av_space_gap_avg,
            vm.av_space_gap_stddev,
            vm.time_gap_min,
            vm.time_gap_max,
            vm.time_gap_avg,
            vm.time_gap_stddev,
            vm.av_time_gap_min,
            vm.av_time_gap_max,
            vm.av_time_gap_avg,
            vm.av_time_gap_stddev,
            vm.speed_min,
            vm.speed_max,
            vm.speed_avg,
            vm.speed_stddev,
            vm.accel_min,
            vm.accel_max,
            vm.accel_avg,
            vm.accel_stddev,
            sm.safety_rate,
            sm.safety_value_max,
            fif_all_prius_ev.percent_infeasible AS prius_percent_infeasible,
            fif_all_tacoma.percent_infeasible AS tacoma_percent_infeasible,
            fif_all_midsize_sedan.percent_infeasible AS midsize_sedan_percent_infeasible,
            fif_all_midsize_suv.percent_infeasible AS midsize_suv_percent_infeasible,
            COALESCE(fif_distribution_v0.percent_infeasible, NULL) AS distribution_v0_percent_infeasible,
            fif_all_rav4.percent_infeasible AS rav4_percent_infeasible,
            fe_all_prius_ev.efficiency_meters_per_kilojoules AS prius_efficiency_meters_per_kilojoules,
            fe_all_tacoma.efficiency_meters_per_kilojoules AS tacoma_efficiency_meters_per_kilojoules,
            fe_all_midsize_sedan.efficiency_meters_per_kilojoules AS midsize_sedan_efficiency_meters_per_kilojoules,
            fe_all_midsize_suv.efficiency_meters_per_kilojoules AS midsize_suv_efficiency_meters_per_kilojoules,
            fe_distribution_v0.efficiency_meters_per_kilojoules AS distribution_v0_efficiency_meters_per_kilojoules,
            fe_all_rav4.efficiency_meters_per_kilojoules AS rav4_efficiency_meters_per_kilojoules,
            fe_all_prius_ev.efficiency_miles_per_gallon AS prius_efficiency_miles_per_gallon,
            fe_all_tacoma.efficiency_miles_per_gallon AS tacoma_efficiency_miles_per_gallon,
            fe_all_midsize_sedan.efficiency_miles_per_gallon AS midsize_sedan_efficiency_miles_per_gallon,
            fe_all_midsize_suv.efficiency_miles_per_gallon AS midsize_suv_efficiency_miles_per_gallon,
            fe_distribution_v0.efficiency_miles_per_gallon AS distribution_v0_efficiency_miles_per_gallon,
            fe_all_rav4.efficiency_miles_per_gallon AS rav4_efficiency_miles_per_gallon,
            fe_all_prius_ev.efficiency_miles_per_gallon_local AS prius_efficiency_miles_per_gallon_local,
            fe_all_tacoma.efficiency_miles_per_gallon_local AS tacoma_efficiency_miles_per_gallon_local,
            fe_all_midsize_sedan.efficiency_miles_per_gallon_local AS midsize_sedan_efficiency_miles_per_gallon_local,
            fe_all_midsize_suv.efficiency_miles_per_gallon_local AS midsize_suv_efficiency_miles_per_gallon_local,
            fe_distribution_v0.efficiency_miles_per_gallon_local AS distribution_v0_efficiency_miles_per_gallon_local,
            fe_all_rav4.efficiency_miles_per_gallon_local AS rav4_efficiency_miles_per_gallon_local
        FROM fact_network_throughput_agg AS nt
        JOIN fact_network_speed AS ns ON 1 = 1
            AND ns.source_id = \'{partition}\'
            AND nt.source_id = ns.source_id
        JOIN fact_vehicle_metrics AS vm ON 1 = 1
            AND vm.source_id = \'{partition}\'
            AND nt.source_id = vm.source_id
        JOIN fact_network_fuel_efficiency_agg AS fe_all_prius_ev ON 1 = 1
            AND fe_all_prius_ev.source_id = \'{partition}\'
            AND nt.source_id = fe_all_prius_ev.source_id
            AND fe_all_prius_ev.distribution_model_id = 'ALL_PRIUS_EV'
        JOIN fact_network_fuel_efficiency_agg AS fe_all_tacoma ON 1 = 1
            AND fe_all_tacoma.source_id = \'{partition}\'
            AND nt.source_id = fe_all_tacoma.source_id
            AND fe_all_tacoma.distribution_model_id = 'ALL_TACOMA'
        JOIN fact_network_fuel_efficiency_agg AS fe_all_midsize_sedan ON 1 = 1
            AND fe_all_midsize_sedan.source_id = \'{partition}\'
            AND nt.source_id = fe_all_midsize_sedan.source_id
            AND fe_all_midsize_sedan.distribution_model_id = 'ALL_MIDSIZE_SEDAN'
        JOIN fact_network_fuel_efficiency_agg AS fe_all_midsize_suv ON 1 = 1
            AND fe_all_midsize_suv.source_id = \'{partition}\'
            AND nt.source_id = fe_all_midsize_suv.source_id
            AND fe_all_midsize_suv.distribution_model_id = 'ALL_MIDSIZE_SUV'
        JOIN fact_network_fuel_efficiency_agg AS fe_distribution_v0 ON 1 = 1
            AND fe_distribution_v0.source_id = \'{partition}\'
            AND nt.source_id = fe_distribution_v0.source_id
            AND fe_distribution_v0.distribution_model_id = 'DISTRIBUTION_V0'
        JOIN fact_network_fuel_efficiency_agg AS fe_all_rav4 ON 1 = 1
            AND fe_all_rav4.source_id = \'{partition}\'
            AND nt.source_id = fe_all_rav4.source_id
            AND fe_all_rav4.distribution_model_id = 'ALL_RAV4'
        JOIN fact_safety_metrics_agg AS sm ON 1 = 1
            AND sm.source_id = \'{partition}\'
            AND nt.source_id = sm.source_id
        JOIN fact_infeasible_flags AS fif_all_prius_ev ON 1 = 1
            AND fif_all_prius_ev.source_id = \'{partition}\'
            AND nt.source_id = fif_all_prius_ev.source_id
            AND fif_all_prius_ev.distribution_model_id = 'ALL_PRIUS_EV'
        JOIN fact_infeasible_flags AS fif_all_tacoma ON 1 = 1
            AND fif_all_tacoma.source_id = \'{partition}\'
            AND nt.source_id = fif_all_tacoma.source_id
            AND fif_all_tacoma.distribution_model_id = 'ALL_TACOMA'
        JOIN fact_infeasible_flags AS fif_all_midsize_sedan ON 1 = 1
            AND fif_all_midsize_sedan.source_id = \'{partition}\'
            AND nt.source_id = fif_all_midsize_sedan.source_id
            AND fif_all_midsize_sedan.distribution_model_id = 'ALL_MIDSIZE_SEDAN'
        JOIN fact_infeasible_flags AS fif_all_midsize_suv ON 1 = 1
            AND fif_all_midsize_suv.source_id = \'{partition}\'
            AND nt.source_id = fif_all_midsize_suv.source_id
            AND fif_all_midsize_suv.distribution_model_id = 'ALL_MIDSIZE_SUV'
        LEFT JOIN fact_infeasible_flags AS fif_distribution_v0 ON 1 = 1
            AND fif_distribution_v0.source_id = \'{partition}\'
            AND nt.source_id = fif_distribution_v0.source_id
            AND fif_distribution_v0.distribution_model_id = 'DISTRIBUTION_V0'
        JOIN fact_infeasible_flags AS fif_all_rav4 ON 1 = 1
            AND fif_all_rav4.source_id = \'{partition}\'
            AND nt.source_id = fif_all_rav4.source_id
            AND fif_all_rav4.distribution_model_id = 'ALL_RAV4'
        WHERE 1 = 1
            AND nt.source_id = \'{partition}\'
        ;"""

    FACT_NETWORK_INFLOWS_OUTFLOWS = """
        INSERT IGNORE INTO fact_network_inflows_outflows(time_step,source_id,inflow_rate,outflow_rate)
        WITH in_out_time_step AS (
            SELECT
                id,
                source_id,
                MIN(CASE WHEN {inflow_filter} THEN time_step - {start_filter} ELSE 1000000 END) AS inflow_time_step,
                MIN(CASE WHEN {outflow_filter} THEN 1000000 ELSE time_step - {start_filter} END) AS outflow_time_step
            FROM fact_vehicle_trace
            WHERE 1 = 1
                AND source_id = \'{partition}\'
            GROUP BY 1, 2
        ), inflows AS (
            SELECT
                CAST(inflow_time_step / 60 AS SIGNED) * 60 AS time_step,
                source_id,
                60 * COUNT(DISTINCT id) AS inflow_rate
            FROM in_out_time_step
            WHERE inflow_time_step < 1000000
            GROUP BY 1, 2
        ), outflows AS (
            SELECT
                CAST(outflow_time_step / 60 AS SIGNED) * 60 AS time_step,
                source_id,
                60 * COUNT(DISTINCT id) AS outflow_rate
            FROM in_out_time_step
            WHERE outflow_time_step < 1000000
            GROUP BY 1, 2
        )
        SELECT
            COALESCE(i.time_step, o.time_step) AS time_step,
            COALESCE(i.source_id, o.source_id) AS source_id,
            COALESCE(i.inflow_rate, 0) AS inflow_rate,
            COALESCE(o.outflow_rate, 0) AS outflow_rate
        FROM inflows i
        LEFT JOIN outflows o ON 1 = 1
            AND i.time_step = o.time_step
            AND i.source_id = o.source_id
        WHERE 1 = 1
            AND COALESCE(i.time_step, o.time_step) >= 0
        UNION
        SELECT
            COALESCE(i.time_step, o.time_step) AS time_step,
            COALESCE(i.source_id, o.source_id) AS source_id,
            COALESCE(i.inflow_rate, 0) AS inflow_rate,
            COALESCE(o.outflow_rate, 0) AS outflow_rate
        FROM inflows i
        RIGHT JOIN outflows o ON 1 = 1
            AND i.time_step = o.time_step
            AND i.source_id = o.source_id
        WHERE 1 = 1
            AND COALESCE(i.time_step, o.time_step) >= 0
        ORDER BY time_step
        ;"""

    FACT_NETWORK_METRICS_BY_DISTANCE_AGG = """
        INSERT IGNORE INTO fact_network_metrics_by_distance_agg(source_id,distance_meters_bin,cumulative_energy_avg,cumulative_energy_lower_bound,cumulative_energy_upper_bound,speed_avg,speed_upper_bound,speed_lower_bound,accel_avg,accel_upper_bound,accel_lower_bound,instantaneous_energy_avg,instantaneous_energy_upper_bound,instantaneous_energy_lower_bound)
        WITH joined_trace AS (
            SELECT
                id,
                source_id,
                time_step,
                distance - FIRST_VALUE(distance)
                    OVER (PARTITION BY id, source_id ORDER BY time_step ASC) AS distance_meters,
                energy_model_id,
                speed,
                acceleration,
                time_step - LAG(time_step, 1)
                    OVER (PARTITION BY id, source_id ORDER BY time_step ASC) AS sim_step,
                SUM(power)
                    OVER (PARTITION BY id, source_id ORDER BY time_step ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING and CURRENT ROW) AS cumulative_power
            FROM fact_energy_trace
            WHERE 1 = 1
                AND source_id = \'{partition}\'
                AND energy_model_id = 'MIDSIZE_SEDAN_FIT_DENOISED_ACCEL'
                AND {inflow_filter}
                AND {outflow_filter}
                AND time_step >= {start_filter}
        ), cumulative_energy AS (
            SELECT
                id,
                source_id,
                time_step,
                distance_meters,
                energy_model_id,
                speed,
                acceleration,
                cumulative_power * sim_step AS energy_joules
            FROM joined_trace
        ), binned_cumulative_energy AS (
            SELECT
                source_id,
                CAST(distance_meters/10 AS SIGNED) * 10 AS distance_meters_bin,
                AVG(speed) AS speed_avg,
                AVG(speed) + STDDEV(speed) AS speed_upper_bound,
                AVG(speed) - STDDEV(speed) AS speed_lower_bound,
                AVG(acceleration) AS accel_avg,
                AVG(acceleration) + STDDEV(acceleration) AS accel_upper_bound,
                AVG(acceleration) - STDDEV(acceleration) AS accel_lower_bound,
                AVG(energy_joules) AS cumulative_energy_avg,
                AVG(energy_joules) + STDDEV(energy_joules) AS cumulative_energy_upper_bound,
                AVG(energy_joules) - STDDEV(energy_joules) AS cumulative_energy_lower_bound
            FROM cumulative_energy
            GROUP BY 1, 2
            HAVING 1 = 1
                AND COUNT(DISTINCT time_step) > 1
        ), binned_energy_start_end AS (
            SELECT DISTINCT
                source_id,
                id,
                CAST(distance_meters/10 AS SIGNED) * 10 AS distance_meters_bin,
                FIRST_VALUE(energy_joules)
                    OVER (PARTITION BY id, CAST(distance_meters/10 AS SIGNED) * 10
                    ORDER BY time_step ASC) AS energy_start,
                LAST_VALUE(energy_joules)
                    OVER (PARTITION BY id, CAST(distance_meters/10 AS SIGNED) * 10
                    ORDER BY time_step ASC) AS energy_end
            FROM cumulative_energy
        ), binned_energy AS (
            SELECT
                source_id,
                distance_meters_bin,
                AVG(energy_end - energy_start) AS instantaneous_energy_avg,
                AVG(energy_end - energy_start) + STDDEV(energy_end - energy_start) AS instantaneous_energy_upper_bound,
                AVG(energy_end - energy_start) - STDDEV(energy_end - energy_start) AS instantaneous_energy_lower_bound
            FROM binned_energy_start_end
            GROUP BY 1, 2
        )
        SELECT
            bce.source_id AS source_id,
            bce.distance_meters_bin AS distance_meters_bin,
            bce.cumulative_energy_avg,
            bce.cumulative_energy_lower_bound,
            bce.cumulative_energy_upper_bound,
            bce.speed_avg,
            bce.speed_upper_bound,
            bce.speed_lower_bound,
            bce.accel_avg,
            bce.accel_upper_bound,
            bce.accel_lower_bound,
            COALESCE(be.instantaneous_energy_avg, 0) AS instantaneous_energy_avg,
            COALESCE(be.instantaneous_energy_upper_bound, 0) AS instantaneous_energy_upper_bound,
            COALESCE(be.instantaneous_energy_lower_bound, 0) AS instantaneous_energy_lower_bound
        FROM binned_cumulative_energy bce
        JOIN binned_energy be ON 1 = 1
            AND bce.source_id = be.source_id
            AND bce.distance_meters_bin = be.distance_meters_bin
        ORDER BY distance_meters_bin ASC
        ;"""

    FACT_NETWORK_METRICS_BY_TIME_AGG = """
        INSERT IGNORE INTO fact_network_metrics_by_time_agg(source_id,time_seconds_bin,cumulative_energy_avg,cumulative_energy_lower_bound,cumulative_energy_upper_bound,speed_avg,speed_upper_bound,speed_lower_bound,accel_avg,accel_upper_bound,accel_lower_bound,instantaneous_energy_avg,instantaneous_energy_upper_bound,instantaneous_energy_lower_bound)
        WITH joined_trace AS (
            SELECT
                id,
                source_id,
                time_step - FIRST_VALUE(time_step)
                    OVER (PARTITION BY id, source_id ORDER BY time_step ASC) AS time_step,
                energy_model_id,
                speed,
                acceleration,
                time_step - LAG(time_step, 1)
                    OVER (PARTITION BY id, source_id ORDER BY time_step ASC) AS sim_step,
                SUM(power)
                    OVER (PARTITION BY id, source_id ORDER BY time_step ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING and CURRENT ROW) AS cumulative_power
            FROM fact_energy_trace
            WHERE 1 = 1
                AND source_id = \'{partition}\'
                AND energy_model_id = 'MIDSIZE_SEDAN_FIT_DENOISED_ACCEL'
                AND {inflow_filter}
                AND {outflow_filter}
                AND time_step >= {start_filter}
        ), cumulative_energy AS (
            SELECT
                id,
                source_id,
                time_step,
                energy_model_id,
                speed,
                acceleration,
                cumulative_power * sim_step AS energy_joules
            FROM joined_trace
        ), binned_cumulative_energy AS (
            SELECT
                source_id,
                CAST(time_step/10 AS SIGNED) * 10 AS time_seconds_bin,
                AVG(speed) AS speed_avg,
                AVG(speed) + STDDEV(speed) AS speed_upper_bound,
                AVG(speed) - STDDEV(speed) AS speed_lower_bound,
                AVG(acceleration) AS accel_avg,
                AVG(acceleration) + STDDEV(acceleration) AS accel_upper_bound,
                AVG(acceleration) - STDDEV(acceleration) AS accel_lower_bound,
                AVG(energy_joules) AS cumulative_energy_avg,
                AVG(energy_joules) + STDDEV(energy_joules) AS cumulative_energy_upper_bound,
                AVG(energy_joules) - STDDEV(energy_joules) AS cumulative_energy_lower_bound
            FROM cumulative_energy
            GROUP BY 1, 2
            HAVING 1 = 1
                AND COUNT(DISTINCT time_step) > 1
        ), binned_energy_start_end AS (
            SELECT DISTINCT
                source_id,
                id,
                CAST(time_step/10 AS SIGNED) * 10 AS time_seconds_bin,
                FIRST_VALUE(energy_joules)
                    OVER (PARTITION BY id, CAST(time_step/10 AS SIGNED) * 10
                    ORDER BY time_step ASC) AS energy_start,
                LAST_VALUE(energy_joules)
                    OVER (PARTITION BY id, CAST(time_step/10 AS SIGNED) * 10
                    ORDER BY time_step ASC) AS energy_end
            FROM cumulative_energy
        ), binned_energy AS (
            SELECT
                source_id,
                time_seconds_bin,
                AVG(energy_end - energy_start) AS instantaneous_energy_avg,
                AVG(energy_end - energy_start) + STDDEV(energy_end - energy_start) AS instantaneous_energy_upper_bound,
                AVG(energy_end - energy_start) - STDDEV(energy_end - energy_start) AS instantaneous_energy_lower_bound
            FROM binned_energy_start_end
            GROUP BY 1, 2
        )
        SELECT
            bce.source_id AS source_id,
            bce.time_seconds_bin AS time_seconds_bin,
            bce.cumulative_energy_avg,
            bce.cumulative_energy_lower_bound,
            bce.cumulative_energy_upper_bound,
            bce.speed_avg,
            bce.speed_upper_bound,
            bce.speed_lower_bound,
            bce.accel_avg,
            bce.accel_upper_bound,
            bce.accel_lower_bound,
            COALESCE(be.instantaneous_energy_avg, 0) AS instantaneous_energy_avg,
            COALESCE(be.instantaneous_energy_upper_bound, 0) AS instantaneous_energy_upper_bound,
            COALESCE(be.instantaneous_energy_lower_bound, 0) AS instantaneous_energy_lower_bound
        FROM binned_cumulative_energy bce
        JOIN binned_energy be ON 1 = 1
            AND bce.source_id = be.source_id
            AND bce.time_seconds_bin = be.time_seconds_bin
        ORDER BY time_seconds_bin ASC
        ;"""

    FACT_VEHICLE_COUNTS_BY_TIME = """
        INSERT IGNORE INTO fact_vehicle_counts_by_time(source_id,time_step,vehicle_count)
        WITH counts AS (
            SELECT
                vt.source_id,
                vt.time_step,
                COUNT(DISTINCT vt.id) AS vehicle_count
            FROM fact_vehicle_trace vt
            WHERE 1 = 1
                AND vt.source_id = \'{partition}\'
                AND vt.{inflow_filter}
                AND vt.{outflow_filter}
                AND vt.time_step >= {start_filter}
            GROUP BY 1, 2
        )
        SELECT
            source_id,
            time_step - FIRST_VALUE(time_step)
                OVER (PARTITION BY source_id ORDER BY time_step ASC) AS time_step,
            vehicle_count
        FROM counts
    ;
    """

    LEADERBOARD_CHART_AGG = """
        REPLACE INTO leaderboard_chart_agg(submission_date,source_id,submitter_name,strategy,network,is_baseline,is_benchmark,tacoma_efficiency_miles_per_gallon,prius_efficiency_miles_per_gallon,midsize_sedan_efficiency_miles_per_gallon,midsize_suv_efficiency_miles_per_gallon,distribution_v0_efficiency_miles_per_gallon,rav4_efficiency_miles_per_gallon,efficiency,tacoma_efficiency_miles_per_gallon_local,prius_efficiency_miles_per_gallon_local,midsize_sedan_efficiency_miles_per_gallon_local,midsize_suv_efficiency_miles_per_gallon_local,distribution_v0_efficiency_miles_per_gallon_local,rav4_efficiency_miles_per_gallon_local,efficiency_local,percent_infeasible,inflow,speed,vmt,lane_changes,space_gap,av_space_gap,time_gap,av_time_gap,speed_stddev,accel_stddev,safety_rate,safety_value_max)
        WITH agg AS (
            SELECT
                m.submission_date,
                l.source_id,
                m.submitter_name,
                m.strategy,
                m.network,
                m.is_baseline,
                COALESCE (m.penetration_rate, 'x') AS penetration_rate,
                COALESCE (m.version, '2.0') AS version,
                COALESCE (m.road_grade, 0) AS road_grade,
                COALESCE (m.on_ramp, 0) AS on_ramp,
                COALESCE (m.is_benchmark, 0) AS is_benchmark,
                l.prius_percent_infeasible,
                l.tacoma_percent_infeasible,
                l.midsize_sedan_percent_infeasible,
                l.midsize_suv_percent_infeasible,
                l.distribution_v0_percent_infeasible,
                l.rav4_percent_infeasible,
                l.prius_efficiency_meters_per_kilojoules,
                l.tacoma_efficiency_meters_per_kilojoules,
                l.midsize_sedan_efficiency_meters_per_kilojoules,
                l.midsize_suv_efficiency_meters_per_kilojoules,
                l.distribution_v0_efficiency_meters_per_kilojoules,
                l.rav4_efficiency_meters_per_kilojoules,
                l.prius_efficiency_miles_per_gallon,
                l.tacoma_efficiency_miles_per_gallon,
                l.midsize_sedan_efficiency_miles_per_gallon,
                l.midsize_suv_efficiency_miles_per_gallon,
                l.distribution_v0_efficiency_miles_per_gallon,
                l.rav4_efficiency_miles_per_gallon,
                l.prius_efficiency_miles_per_gallon_local,
                l.tacoma_efficiency_miles_per_gallon_local,
                l.midsize_sedan_efficiency_miles_per_gallon_local,
                l.midsize_suv_efficiency_miles_per_gallon_local,
                l.distribution_v0_efficiency_miles_per_gallon_local,
                l.rav4_efficiency_miles_per_gallon_local,
                l.throughput_per_hour,
                l.avg_instantaneous_speed,
                l.avg_network_speed,
                l.total_vmt,
                l.lane_changes_per_vehicle,
                l.space_gap_min,
                l.space_gap_max,
                l.space_gap_avg,
                l.space_gap_stddev,
                l.av_space_gap_min,
                l.av_space_gap_max,
                l.av_space_gap_avg,
                l.av_space_gap_stddev,
                l.time_gap_min,
                l.time_gap_max,
                l.time_gap_avg,
                l.time_gap_stddev,
                l.av_time_gap_min,
                l.av_time_gap_max,
                l.av_time_gap_avg,
                l.av_time_gap_stddev,
                l.speed_min,
                l.speed_max,
                l.speed_avg,
                l.speed_stddev,
                l.accel_min,
                l.accel_max,
                l.accel_avg,
                l.accel_stddev,
                l.safety_rate,
                l.safety_value_max,
                b.source_id AS baseline_source_id
            FROM leaderboard_chart AS l, metadata_table AS m, baseline_table as b
            WHERE 1 = 1
                AND l.source_id = m.source_id
                AND m.network = b.network
                AND m.version = b.version
                AND m.on_ramp = b.on_ramp
                AND m.road_grade = b.road_grade
                AND m.penetration_rate = b.penetration_rate
                AND (m.is_baseline=0
                     OR (m.is_baseline=1
                         AND m.source_id = b.source_id))
        ), joined_cols AS (
            SELECT
                agg.submission_date,
                agg.source_id,
                agg.submitter_name,
                agg.strategy,
                CONCAT (agg.network , ';' ,
                    ' v' , agg.version , ';' ,
                    ' PR: ' , agg.penetration_rate , '%;' ,
                    CASE WHEN agg.on_ramp = 1
                        THEN ' with ramps;'
                        ELSE ' no ramps;' END ,
                    CASE WHEN agg.road_grade= 1
                        THEN ' with grade;'
                        ELSE ' no grade;' END) AS network,
                agg.is_baseline,
                agg.is_benchmark,
                agg.prius_percent_infeasible,
                agg.tacoma_percent_infeasible,
                agg.midsize_sedan_percent_infeasible,
                agg.midsize_suv_percent_infeasible,
                agg.distribution_v0_percent_infeasible,
                agg.rav4_percent_infeasible,
                agg.prius_efficiency_miles_per_gallon,
                agg.tacoma_efficiency_miles_per_gallon,
                agg.midsize_sedan_efficiency_miles_per_gallon,
                agg.midsize_suv_efficiency_miles_per_gallon,
                agg.distribution_v0_efficiency_miles_per_gallon,
                agg.rav4_efficiency_miles_per_gallon,
                IF(agg.prius_efficiency_miles_per_gallon=0,
                NULL,
                100 * (1 -
                    baseline.prius_efficiency_miles_per_gallon /
                    agg.prius_efficiency_miles_per_gallon))
                    AS prius_fuel_economy_improvement,
                IF(agg.tacoma_efficiency_miles_per_gallon=0,
                NULL,
                100 * (1 -
                    baseline.tacoma_efficiency_miles_per_gallon /
                    agg.tacoma_efficiency_miles_per_gallon))
                    AS tacoma_fuel_economy_improvement,
                IF(agg.midsize_sedan_efficiency_miles_per_gallon=0,
                NULL,
                100 * (1 -
                    baseline.midsize_sedan_efficiency_miles_per_gallon /
                    agg.midsize_sedan_efficiency_miles_per_gallon))
                    AS midsize_sedan_fuel_economy_improvement,
                IF(agg.midsize_suv_efficiency_miles_per_gallon=0,
                NULL,
                100 * (1 -
                    baseline.midsize_suv_efficiency_miles_per_gallon /
                    agg.midsize_suv_efficiency_miles_per_gallon))
                    AS midsize_suv_fuel_economy_improvement,
                IF(agg.distribution_v0_efficiency_miles_per_gallon=0,
                NULL,
                100 * (1 -
                    baseline.distribution_v0_efficiency_miles_per_gallon /
                    agg.distribution_v0_efficiency_miles_per_gallon))
                    AS distribution_v0_fuel_economy_improvement,
                IF(agg.rav4_efficiency_miles_per_gallon=0,
                NULL,
                100 * (1 -
                    baseline.rav4_efficiency_miles_per_gallon /
                    agg.rav4_efficiency_miles_per_gallon))
                    AS rav4_fuel_economy_improvement,
                agg.prius_efficiency_miles_per_gallon_local,
                agg.tacoma_efficiency_miles_per_gallon_local,
                agg.midsize_sedan_efficiency_miles_per_gallon_local,
                agg.midsize_suv_efficiency_miles_per_gallon_local,
                agg.distribution_v0_efficiency_miles_per_gallon_local,
                agg.rav4_efficiency_miles_per_gallon_local,
                IF(agg.prius_efficiency_miles_per_gallon_local=0,
                NULL,
                100 * (1 -
                    baseline.prius_efficiency_miles_per_gallon_local /
                    agg.prius_efficiency_miles_per_gallon_local))
                    AS prius_fuel_economy_improvement_local,
                IF(agg.tacoma_efficiency_miles_per_gallon_local=0,
                NULL,
                100 * (1 -
                    baseline.tacoma_efficiency_miles_per_gallon_local /
                    agg.tacoma_efficiency_miles_per_gallon_local))
                    AS tacoma_fuel_economy_improvement_local,
                IF(agg.midsize_sedan_efficiency_miles_per_gallon_local=0,
                NULL,
                100 * (1 -
                    baseline.midsize_sedan_efficiency_miles_per_gallon_local /
                    agg.midsize_sedan_efficiency_miles_per_gallon_local))
                    AS midsize_sedan_fuel_economy_improvement_local,
                IF(agg.midsize_suv_efficiency_miles_per_gallon_local=0,
                NULL,
                100 * (1 -
                    baseline.midsize_suv_efficiency_miles_per_gallon_local /
                    agg.midsize_suv_efficiency_miles_per_gallon_local))
                    AS midsize_suv_fuel_economy_improvement_local,
                IF(agg.distribution_v0_efficiency_miles_per_gallon_local=0,
                NULL,
                100 * (1 -
                    baseline.distribution_v0_efficiency_miles_per_gallon_local /
                    agg.distribution_v0_efficiency_miles_per_gallon_local))
                    AS distribution_v0_fuel_economy_improvement_local,
                IF(agg.rav4_efficiency_miles_per_gallon_local=0,
                NULL,
                100 * (1 -
                    baseline.rav4_efficiency_miles_per_gallon_local /
                    agg.rav4_efficiency_miles_per_gallon_local))
                    AS rav4_fuel_economy_improvement_local,
                agg.throughput_per_hour,
                IF(baseline.throughput_per_hour=0,
                NULL,
                100 * (agg.throughput_per_hour - baseline.throughput_per_hour) /
                    baseline.throughput_per_hour)
                    AS throughput_change,
                agg.avg_network_speed,
                IF(baseline.avg_network_speed=0,
                NULL,
                100 * (agg.avg_network_speed - baseline.avg_network_speed) /
                    baseline.avg_network_speed)
                    AS speed_change,
                agg.total_vmt,
                IF(baseline.total_vmt=0,
                NULL,
                100 * (agg.total_vmt - baseline.total_vmt) /
                    baseline.total_vmt)
                    AS vmt_change,
                agg.lane_changes_per_vehicle,
                IF(baseline.lane_changes_per_vehicle=0,
                NULL,
                100 * (agg.lane_changes_per_vehicle - baseline.lane_changes_per_vehicle) /
                    baseline.lane_changes_per_vehicle)
                    AS lane_changes_change,
                agg.space_gap_min,
                agg.space_gap_max,
                agg.space_gap_avg,
                agg.space_gap_stddev,
                IF(baseline.space_gap_stddev=0,
                NULL,
                100 * (agg.space_gap_stddev - baseline.space_gap_stddev) /
                    baseline.space_gap_stddev)
                    AS space_gap_dev_change,
                agg.av_space_gap_min,
                agg.av_space_gap_max,
                agg.av_space_gap_avg,
                agg.av_space_gap_stddev,
                IF(baseline.av_space_gap_stddev=0,
                NULL,
                100 * (agg.av_space_gap_stddev - baseline.av_space_gap_stddev) /
                    baseline.av_space_gap_stddev)
                    AS av_space_gap_dev_change,
                agg.time_gap_min,
                agg.time_gap_max,
                agg.time_gap_avg,
                agg.time_gap_stddev,
                IF(baseline.time_gap_stddev=0,
                NULL,
                100 * (agg.time_gap_stddev - baseline.time_gap_stddev) /
                    baseline.time_gap_stddev)
                    AS time_gap_dev_change,
                agg.av_time_gap_min,
                agg.av_time_gap_max,
                agg.av_time_gap_avg,
                agg.av_time_gap_stddev,
                IF(baseline.av_time_gap_stddev=0,
                NULL,
                100 * (agg.av_time_gap_stddev - baseline.av_time_gap_stddev) /
                    baseline.av_time_gap_stddev)
                    AS av_time_gap_dev_change,
                agg.speed_min,
                agg.speed_max,
                agg.speed_avg,
                agg.speed_stddev,
                IF(baseline.speed_stddev=0,
                NULL,
                100 * (agg.speed_stddev - baseline.speed_stddev) /
                    baseline.speed_stddev)
                    AS speed_dev_change,
                agg.accel_min,
                agg.accel_max,
                agg.accel_avg,
                agg.accel_stddev,
                IF(baseline.accel_stddev=0,
                NULL,
                100 * (agg.accel_stddev - baseline.accel_stddev) /
                    baseline.accel_stddev)
                    AS accel_dev_change,
                agg.safety_rate,
                agg.safety_value_max
            FROM agg
            JOIN agg AS baseline ON 1 = 1
                AND agg.network = baseline.network
                AND agg.version = baseline.version
                AND agg.on_ramp = baseline.on_ramp
                AND agg.road_grade = baseline.road_grade
                AND baseline.is_baseline = 1
                AND agg.baseline_source_id = baseline.source_id
        )
        SELECT DISTINCT
            submission_date,
            source_id,
            submitter_name,
            strategy,
            network,
            is_baseline,
            is_benchmark,
            tacoma_efficiency_miles_per_gallon,
            prius_efficiency_miles_per_gallon,
            midsize_sedan_efficiency_miles_per_gallon,
            midsize_suv_efficiency_miles_per_gallon,
            distribution_v0_efficiency_miles_per_gallon,
            rav4_efficiency_miles_per_gallon,
            CONCAT(
            CASE
                WHEN network = 'straight-road'
                THEN CONCAT(CAST(ROUND(rav4_efficiency_miles_per_gallon, 1) AS CHAR(10)) ,
                ' (' , (CASE WHEN SIGN(rav4_fuel_economy_improvement) = 1 THEN '+' ELSE '' END) ,
                CAST(ROUND(rav4_fuel_economy_improvement, 1) AS CHAR(10)))
                ELSE CONCAT(CAST(ROUND(distribution_v0_efficiency_miles_per_gallon, 1) AS CHAR(10)) ,
                ' (' , (CASE WHEN SIGN(distribution_v0_fuel_economy_improvement) = 1 THEN '+' ELSE '' END) ,
                CAST(ROUND(distribution_v0_fuel_economy_improvement, 1) AS CHAR(10)))
            END , '%)') AS efficiency,
            tacoma_efficiency_miles_per_gallon_local,
            prius_efficiency_miles_per_gallon_local,
            midsize_sedan_efficiency_miles_per_gallon_local,
            midsize_suv_efficiency_miles_per_gallon_local,
            distribution_v0_efficiency_miles_per_gallon_local,
            rav4_efficiency_miles_per_gallon_local,
            CASE
                WHEN network = 'straight-road'
                THEN CAST(ROUND(rav4_efficiency_miles_per_gallon_local, 1) AS CHAR(10))
                ELSE CAST(ROUND(distribution_v0_efficiency_miles_per_gallon_local, 1) AS CHAR(10))
            END AS efficiency_local,
            CONCAT(
            CASE
                WHEN network = 'straight-road'
                THEN CAST(ROUND(rav4_percent_infeasible, 1) AS CHAR(10))
                ELSE CAST(ROUND(distribution_v0_percent_infeasible, 1) AS CHAR(10))
            END , '%') AS percent_infeasible,
            CONCAT(
            CAST(ROUND(throughput_per_hour, 1) AS CHAR(10)) ,
                ' (' , (CASE WHEN SIGN(throughput_change) = 1 THEN '+' ELSE '' END) ,
                CAST(ROUND(throughput_change, 1) AS CHAR(10)) , '%)') AS inflow,
            CONCAT(
            CAST(ROUND(avg_network_speed, 1) AS CHAR(10)) ,
                ' (' , (CASE WHEN SIGN(speed_change) = 1 THEN '+' ELSE '' END) ,
                CAST(ROUND(speed_change, 1) AS CHAR(10)) , '%)') AS speed,
            CONCAT(
            CAST(ROUND(total_vmt, 1) AS CHAR(10)) ,
                ' (' , (CASE WHEN SIGN(vmt_change) = 1 THEN '+' ELSE '' END) ,
                CAST(ROUND(vmt_change, 1) AS CHAR(10)) , '%)') AS vmt,
            CONCAT(
            CAST(ROUND(lane_changes_per_vehicle, 1) AS CHAR(10)) ,
                ' (' , (CASE WHEN SIGN(lane_changes_change) = 1 THEN '+' ELSE '' END) ,
                CAST(ROUND(lane_changes_change, 1) AS CHAR(10)) , '%)') AS lane_changes,
            CONCAT(
            '[' , CAST(ROUND(space_gap_min, 1) AS CHAR(10)) , '-' ,
                CAST(ROUND(space_gap_max, 1) AS CHAR(10)) , '] <' ,
                CAST(ROUND(space_gap_avg, 1) AS CHAR(10)) , '>') AS space_gap,
            CONCAT(
            '[' , CAST(ROUND(av_space_gap_min, 1) AS CHAR(10)) , '-' ,
                CAST(ROUND(av_space_gap_max, 1) AS CHAR(10)) , '] <' ,
                CAST(ROUND(av_space_gap_avg, 1) AS CHAR(10)) , '>') AS av_space_gap,
            CONCAT(
            '[' , CAST(ROUND(time_gap_min, 1) AS CHAR(10)) , '-' ,
                CAST(ROUND(time_gap_max, 1) AS CHAR(10)) , '] <' ,
                CAST(ROUND(time_gap_avg, 1) AS CHAR(10)) , '>') AS time_gap,
            CONCAT(
            '[' , CAST(ROUND(av_time_gap_min, 1) AS CHAR(10)) , '-' ,
                CAST(ROUND(av_time_gap_max, 1) AS CHAR(10)) , '] <' ,
                CAST(ROUND(av_time_gap_avg, 1) AS CHAR(10)) , '>') AS av_time_gap,
            CONCAT(
            CAST(ROUND(speed_stddev, 1) AS CHAR(10)) ,
                ' (' , (CASE WHEN SIGN(speed_dev_change) = 1 THEN '+' ELSE '' END) ,
                CAST(ROUND(speed_dev_change, 1) AS CHAR(10)) , '%)\n[' ,
                CAST(ROUND(speed_min, 1) AS CHAR(10)) , '-' , CAST(ROUND(speed_max, 1) AS CHAR(10)) ,
                '] <' , CAST(ROUND(speed_avg, 1) AS CHAR(10)) , '>') AS speed_stddev,
            CONCAT(
            CAST(ROUND(accel_stddev, 1) AS CHAR(10)) ,
                ' (' , (CASE WHEN SIGN(accel_dev_change) = 1 THEN '+' ELSE '' END) ,
                CAST(ROUND(accel_dev_change, 1) AS CHAR(10)) , '%)\n[' ,
                CAST(ROUND(accel_min, 1) AS CHAR(10)) , '-' , CAST(ROUND(accel_max, 1) AS CHAR(10)) ,
                '] <' , CAST(ROUND(accel_avg, 1) AS CHAR(10)) , '>') AS accel_stddev,
            ROUND(safety_rate, 1) AS safety_rate,
            ROUND(safety_value_max, 1) AS safety_value_max
        FROM joined_cols
        ;"""

    FACT_TOP_SCORES = """
        REPLACE INTO fact_top_scores(network,submission_date,tacoma_max_score,prius_max_score,midsize_sedan_max_score,midsize_suv_max_score,distribution_v0_max_score,rav4_max_score)
        WITH curr_max AS (
            SELECT
                network,
                submission_date,
                MAX(tacoma_efficiency_miles_per_gallon)
                    OVER (PARTITION BY network ORDER BY submission_date ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING and CURRENT ROW) AS tacoma_max_score,
                MAX(prius_efficiency_miles_per_gallon)
                    OVER (PARTITION BY network ORDER BY submission_date ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING and CURRENT ROW) AS prius_max_score,
                MAX(midsize_sedan_efficiency_miles_per_gallon)
                    OVER (PARTITION BY network ORDER BY submission_date ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING and CURRENT ROW) AS midsize_sedan_max_score,
                MAX(midsize_suv_efficiency_miles_per_gallon)
                    OVER (PARTITION BY network ORDER BY submission_date ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING and CURRENT ROW) AS midsize_suv_max_score,
                MAX(distribution_v0_efficiency_miles_per_gallon)
                    OVER (PARTITION BY network ORDER BY submission_date ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING and CURRENT ROW) AS distribution_v0_max_score,
                MAX(rav4_efficiency_miles_per_gallon)
                    OVER (PARTITION BY network ORDER BY submission_date ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING and CURRENT ROW) AS rav4_max_score
            FROM leaderboard_chart_agg
            WHERE 1 = 1
                AND is_baseline = 0
        ), prev_max AS (
            SELECT
                network,
                submission_date,
                LAG(tacoma_max_score, 1)
                    OVER (PARTITION BY network ORDER BY submission_date ASC) AS tacoma_max_score,
                LAG(prius_max_score, 1)
                    OVER (PARTITION BY network ORDER BY submission_date ASC) AS prius_max_score,
                LAG(midsize_sedan_max_score, 1)
                    OVER (PARTITION BY network ORDER BY submission_date ASC) AS midsize_sedan_max_score,
                LAG(midsize_suv_max_score, 1)
                    OVER (PARTITION BY network ORDER BY submission_date ASC) AS midsize_suv_max_score,
                LAG(distribution_v0_max_score, 1)
                    OVER (PARTITION BY network ORDER BY submission_date ASC) AS distribution_v0_max_score,
                LAG(rav4_max_score, 1)
                    OVER (PARTITION BY network ORDER BY submission_date ASC) AS rav4_max_score
            FROM curr_max
        ), unioned AS (
            SELECT * FROM curr_max
            UNION ALL
            SELECT * FROM prev_max
        )
        SELECT DISTINCT *
        FROM unioned
        WHERE 1 = 1
            AND tacoma_max_score IS NOT NULL
            AND prius_max_score IS NOT NULL
            AND midsize_sedan_max_score IS NOT NULL
            AND midsize_suv_max_score IS NOT NULL
            AND distribution_v0_max_score IS NOT NULL
            AND rav4_max_score IS NOT NULL
        ORDER BY 1, 2, 3
        ;"""
