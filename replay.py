from utils.db.simulation_db import SimulationDB
from utils.df_utils import plot_dataframe, convert_cols_to_numeric, get_df_from_csv, get_formatted_df_for_llm
from utils.trajectory.trajectory_container import TrajectoryChunker, TrajectoryWindows
from models.llm_agent import LLM_Agent, OpenAiModel, GroqModel
import pandas as pd
import numpy as np
import re
from sklearn.metrics import mean_squared_error

db_path = "data/simulate/1743479815_01Apr25_03h56m55s/2021-04-22-12-47-13_2T3MWRFVXLW056972_masterArray_0_7050_run_0.db"

# Example usage of using the database context manager
with SimulationDB(db_path) as db:
    # rl_av_data = db.get_vehicle_data("1_rl_av", ["speed", "leader_speed", "position", "headway"])
    pass

trajectory_df = get_df_from_csv("data/simulate/1743374502_30Mar25_22h41m42s/emissions/emissions_1.csv")

trajectory_df = trajectory_df[trajectory_df["id"] == "1_rl_av"]
trajectory_df = trajectory_df[["step", "speed", "headway", "leader_speed"]]

trajectory_segmented = TrajectoryChunker(
    trajectory_df,
    chunk_col="step",
    chunk_indices=[575, 1000, 1500],
    sort_col="step"
)
trajectory_chunks = trajectory_segmented.get_chunks()
trajectory_windows = TrajectoryWindows(
    trajectory_chunks[0],
    window_size=10,
    indexing_col="step"
)
trajectory_windows = trajectory_windows.get_windows()

system_prompt = """
You are an AI assistant specializing in traffic flow analysis and vehicle dynamics. Your task is to analyze the initial 3 timesteps of vehicle trajectory data and predict the next 7 timesteps for all three variables: speed (m/s), headway (m), and leader_speed (m/s).

The data represents real-world traffic measurements from a vehicle equipped with adaptive cruise control that aims to smooth traffic flow. You must understand the relationship between the following vehicle, its speed adjustments, and the leader vehicle ahead.

For each prediction, you must:
1. First analyze the initial data inside <reasoning> tags to explain your thought process
2. Provide precise numerical predictions for each variable
3. Format your predictions in separate tags:
   - <future_speeds>speed_t3,speed_t4,speed_t5,speed_t6,speed_t7,speed_t8,speed_t9</future_speeds>
   - <future_headway>headway_t3,headway_t4,headway_t5,headway_t6,headway_t7,headway_t8,headway_t9</future_headway>
   - <future_leader_speed>leader_t3,leader_t4,leader_t5,leader_t6,leader_t7,leader_t8,leader_t9</future_leader_speed>

In your reasoning, analyze:
1. The behavior of the leader vehicle compared to our vehicle
2. The apparent advised speed for traffic smoothing
3. The actual set speed of the vehicle 
4. Whether future acceleration or deceleration is expected

You are explaining what a self-driving car is doing to smooth traffic flow. Your predictions must adhere to basic physics and vehicle dynamics. Make your explanation clear, grounded in the data, and avoid any hallucinations. Calculate the L2 norm of your predictions as part of your evaluation.
"""

def extract_tag_content(response, tag_name):
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip(), True
    else:
        return [], False


def extract_values(response, tag_name):
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        values_str = match.group(1).strip()
        try:
            values = [float(v.strip()) for v in values_str.split(',') if v.strip()]
            return values, True
        except ValueError:
            return [], False
    return [], False

def calculate_l2_norm(predictions, ground_truth):
    if len(predictions) != len(ground_truth):
        return float('nan')
    return np.sqrt(mean_squared_error(ground_truth, predictions))

def evaluate_window(window_data, start_idx, llm_agent):
    first_three_steps = window_data.iloc[:3]
    rest_steps = window_data.iloc[3:]
    
    formatted_first_three = get_formatted_df_for_llm(first_three_steps, precision=2)
    
    user_prompt = f"""
    Given the following initial 3 timesteps of trajectory data:
    
    {formatted_first_three}
    
    Predict the next 7 timesteps (steps 3-9) for all three variables. Explain your reasoning and provide the predictions in the required format.
    
    Think through these steps:
    1. What is the behavior of the leader vehicle? Compare our velocity to the leader's velocity.
    2. What is the advised speed set by flowmo?
    3. What is the actual set speed of the vehicle (flowmo-set-speed)?
    4. Look at the current advised speed and the current set speed. Do we expect to slow down in the future?
    """
    
    response = llm_agent.get_response(system_prompt, user_prompt)
    
    speed_values, speed_extractable = extract_values(response, "future_speeds")
    headway_values, headway_extractable = extract_values(response, "future_headway")
    leader_speed_values, leader_extractable = extract_values(response, "future_leader_speed")
    reasoning, reasoning_extractable = extract_tag_content(response, "reasoning")
    
    all_extractable = speed_extractable and headway_extractable and leader_extractable
    
    gt_speeds = rest_steps['speed'].values
    gt_headways = rest_steps['headway'].values
    gt_leader_speeds = rest_steps['leader_speed'].values
    
    speed_l2 = calculate_l2_norm(speed_values, gt_speeds) if speed_extractable else float('nan')
    headway_l2 = calculate_l2_norm(headway_values, gt_headways) if headway_extractable else float('nan')
    leader_l2 = calculate_l2_norm(leader_speed_values, gt_leader_speeds) if leader_extractable else float('nan')
    
    if all_extractable:
        overall_l2 = (speed_l2 + headway_l2 + leader_l2) / 3
    else:
        overall_l2 = float('nan')
    
    result = {
        'start_index': start_idx,
        'user_prompt': user_prompt,
        'response': response,
        'reasoning': reasoning if reasoning_extractable else "Not extractable",
        'speed_values': speed_values if speed_extractable else [],
        'headway_values': headway_values if headway_extractable else [],
        'leader_speed_values': leader_speed_values if leader_extractable else [],
        'all_extractable': all_extractable,
        'speed_extractable': speed_extractable,
        'headway_extractable': headway_extractable,
        'leader_extractable': leader_extractable,
        'speed_l2': speed_l2,
        'headway_l2': headway_l2,
        'leader_l2': leader_l2,
        'overall_l2': overall_l2
    }
    return result

llm_agent = GroqModel()

results = []

for i, window in enumerate(trajectory_windows):
    print(f"Processing window {i+1}/{len(trajectory_windows)}")
    start_idx = window.iloc[0]['step']
    result = evaluate_window(window, start_idx, llm_agent)
    results.append(result)

results_df = pd.DataFrame(results)
results_df.to_json("trajectory_prediction_results.json", orient="records", indent=2)


plot_dataframe(
    trajectory_df,
    x_axis="step",
    y_values=["speed", "leader_speed"],
    save_path="rl_av_data_from_emissions_chunked.png",
    chunker=trajectory_segmented,
    shaded_regions=[(10, 20, "orange", 0.2)],
)
