from utils.db.simulation_db import SimulationDB
from utils.df_utils import plot_dataframe, convert_cols_to_numeric, get_df_from_csv, get_formatted_df_for_llm
from utils.trajectory.trajectory_container import TrajectoryChunker, TrajectoryWindows
from models.llm_agent import LLM_Agent, OpenAiModel, GroqModel
import pandas as pd
import numpy as np
import re
from sklearn.metrics import mean_squared_error
import asyncio

# Loading in the prompts that we want to use 
from prompts.ashwin_04_05 import user_prompt, system_prompt

USE_DB = True # currently the csv doesn't have reward so ideally use db

if USE_DB:
    db_path = "data/simulate/1743365635_30Mar25_20h13m55s/2021-04-22-12-47-13_2T3MWRFVXLW056972_masterArray_0_7050_run_0.db"
    with SimulationDB(db_path) as db:
        trajectory_df = db.get_vehicle_data("1_rl_av", ["speed", "leader_speed", "position", "headway"])
    chunk_val = "timestep"
    trajectory_df['timestep'] = pd.to_numeric(trajectory_df['timestep'])
    columns_to_rename = {col: col.replace('1_rl_av__', '') for col in trajectory_df.columns if '1_rl_av__' in col}
    trajectory_df = trajectory_df.rename(columns=columns_to_rename)
    print(trajectory_df)
else:
    trajectory_df = get_df_from_csv("data/simulate/1743374502_30Mar25_22h41m42s/emissions/emissions_1.csv")
    trajectory_df = trajectory_df[trajectory_df["id"] == "1_rl_av"]
    trajectory_df = trajectory_df[["step", "speed", "headway", "leader_speed", "position"]] # TODO: position is the placeholder for reward for the current moment
    chunk_val = "step"

trajectory_segmented = TrajectoryChunker(
    trajectory_df,
    chunk_col=chunk_val,
    chunk_indices=[575, 1000, 1500],
    sort_col=chunk_val
)
trajectory_chunks = trajectory_segmented.get_chunks()
trajectory_windows = TrajectoryWindows(
    trajectory_chunks[0],
    window_size=10,
    indexing_col=chunk_val
)
trajectory_windows = trajectory_windows.get_windows()

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
    first_six_steps = window_data.iloc[:6]
    rest_steps = window_data.iloc[6:]
    formatted_first_six = get_formatted_df_for_llm(first_six_steps, precision=2)
    
    response = llm_agent.get_response(system_prompt, user_prompt.format(formatted_first_six), temperature=1, num_samples=1)
    speed_values, speed_ok = extract_values(response, "future_speeds")
    headway_values, headway_ok = extract_values(response, "future_headway")
    leader_values, leader_ok = extract_values(response, "future_leader_speed")
    reward_values, reward_ok = extract_values(response, "future_rewards")
    coeff_str, coeff_ok = extract_tag_content(response, "reward_coefficients")
    reasoning, reasoning_ok = extract_tag_content(response, "reasoning")

    try:
        reward_coefficients = [float(x.strip()) for x in coeff_str.split(',')]
    except:
        reward_coefficients = []
    
    all_ok = speed_ok and headway_ok and leader_ok and reward_ok
    
    gt = rest_steps
    gt_speeds = gt["speed"].values
    gt_headways = gt["headway"].values
    gt_leader_speed = gt["leader_speed"].values
    gt_rewards = gt["position"].values
    
    result = {
        "start_index": start_idx,
        "user_prompt": user_prompt,
        "response": response,
        "reasoning": reasoning if reasoning_ok else "Not extractable",
        "reward_coefficients": reward_coefficients if coeff_ok else [],
        "reward_values": reward_values if reward_ok else [],
        "speed_values": speed_values if speed_ok else [],
        "headway_values": headway_values if headway_ok else [],
        "leader_speed_values": leader_values if leader_ok else [],
        "all_extractable": all_ok,
        "speed_l2": calculate_l2_norm(speed_values, gt_speeds) if speed_ok else float('nan'),
        "headway_l2": calculate_l2_norm(headway_values, gt_headways) if headway_ok else float('nan'),
        "leader_l2": calculate_l2_norm(leader_values, gt_leader_speed) if leader_ok else float('nan'),
        "reward_l2": calculate_l2_norm(reward_values, gt_rewards) if reward_ok else float('nan')
    }
    
    result["overall_l2"] = np.nanmean([
        result["speed_l2"], result["headway_l2"], result["leader_l2"]
    ]) if all_ok else float("nan")

    return result

llm_agent = OpenAiModel()

results = []

for i, window in enumerate(trajectory_windows):
    if i ==2:
        break
    print(f"Processing window {i+1}/{len(trajectory_windows)}")
    start_idx = window.iloc[0][chunk_val]
    result = evaluate_window(window, start_idx, llm_agent)
    results.append(result)

results_df = pd.DataFrame(results)
results_df.to_json("trajectory_prediction_results.json", orient="records", indent=2)


plot_dataframe(
    trajectory_df,
    x_axis=chunk_val,
    y_values=["speed", "leader_speed"],
    save_path="rl_av_data_from_emissions_chunked.png",
    chunker=trajectory_segmented,
    shaded_regions=[(10, 20, "orange", 0.2)],
)
