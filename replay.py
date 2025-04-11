import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime
import pprint

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.db.simulation_db import SimulationDB
from utils.df_utils import (
    get_df_from_csv,
    get_formatted_df_for_llm,
    plot_dataframe
)
from utils.formatters.response_extractors import (
    calculate_l2_norm,
    extract_tag_content,
    extract_values
)
from utils.rag.embedding_models import EnvironmentEmbeddingModel
from utils.rag.read_embedding_db import RAG_Database
from utils.trajectory.trajectory_container import TrajectoryChunker, TrajectoryWindows

# Load prompts
from prompts.ashwin_04_05 import user_prompt
from prompts.adrien_04_07 import rag_user_prompt, system_prompt

# Loading in the util functions that help us parse/extract the response from the LLM.
from utils.formatters.response_extractors import extract_values, extract_tag_content, calculate_l2_norm

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run simulation with specified experiment configuration")
    parser.add_argument(
        "--experiment", 
        type=str, 
        default="experiments/experiment_1.py",
        help="Path to the experiment configuration file (e.g., 'experiments/experiment_1.py')"
    )
    return parser.parse_args()


def load_experiment_config(experiment_path):
    """Dynamically load experiment configuration from file"""
    print(f"Loading experiment configuration from: {experiment_path}")
    
    spec = importlib.util.spec_from_file_location("experiment", experiment_path)
    experiment_module = importlib.util.module_from_spec(spec)
    sys.modules["experiment"] = experiment_module
    spec.loader.exec_module(experiment_module)
    config = experiment_module.config
    
    print(f"Experiment metadata: {config.get('experiment_metadata', 'No metadata provided')}")
    return config


def create_output_directories(config):
    """Create output directories with timestamp and return paths"""
    # Create base output directory if it doesn't exist
    os.makedirs(config["output_dir"], exist_ok=True)

    # Create timestamped run directory
    current_datetime = datetime.now()
    hour = current_datetime.hour
    am_pm = "_am" if hour < 12 else "_pm"
    formatted_datetime = current_datetime.strftime('%m_%d_%y_%H%M%S') + am_pm
    run_dir = os.path.join(config["output_dir"], formatted_datetime)
    os.makedirs(run_dir, exist_ok=True)

    # Set full output paths
    results_path = os.path.join(run_dir, config["results_filename"])
    plot_path = os.path.join(run_dir, config["plot_filename"])
    
    return run_dir, results_path, plot_path, current_datetime


def save_config_files(config, run_dir, experiment_path, current_datetime):
    """Save experiment configuration as JSON and text files"""
    # Create a serializable copy of the config
    save_config = config.copy()
    if "llm_model" in save_config:
        save_config["llm_model"] = str(type(save_config["llm_model"]).__name__)

    # Save as JSON
    with open(os.path.join(run_dir, "experiment_config.json"), "w") as f:
        json.dump(save_config, f, indent=4, default=str)
    
    # Save as pretty-printed text
    with open(os.path.join(run_dir, "experiment_config.txt"), "w") as f:
        f.write(f"Experiment source: {experiment_path}\n")
        f.write(f"Run timestamp: {current_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Configuration:\n")
        f.write(pprint.pformat(save_config, indent=4, width=100))


def load_trajectory_data(config):
    """Load trajectory data from database or CSV file"""
    if config["use_db"]:
        # Load from database
        db_path = config["db_path"]
        with SimulationDB(db_path) as db:
            trajectory_df = db.get_vehicle_data(config["vehicle_id"], config["db_columns"])
        
        # Process database data
        chunk_val = config["chunk_col_db"]
        trajectory_df['timestep'] = pd.to_numeric(trajectory_df['timestep'])
        columns_to_rename = {col: col.replace('1_rl_av__', '') 
                            for col in trajectory_df.columns if '1_rl_av__' in col}
        trajectory_df = trajectory_df.rename(columns=columns_to_rename)
        print(trajectory_df)
    else:
        # Load from CSV
        trajectory_df = get_df_from_csv(config["csv_path"])
        trajectory_df = trajectory_df[trajectory_df["id"] == config["vehicle_id"]]
        trajectory_df = trajectory_df[config["csv_columns"]]  # TODO: position is the placeholder for reward
        chunk_val = config["chunk_col_csv"]
    
    return trajectory_df, chunk_val


def process_trajectory(trajectory_df, chunk_val, config):
    """Process trajectory data into chunks and windows"""
    # Create trajectory segments
    trajectory_segmented = TrajectoryChunker(
        trajectory_df,
        chunk_col=chunk_val,
        chunk_indices=config["chunk_indices"],
        sort_col=chunk_val
    )
    trajectory_chunks = trajectory_segmented.get_chunks()
    
    # Create trajectory windows
    trajectory_windows = TrajectoryWindows(
        trajectory_chunks[0],
        window_size=config["window_size"],
        indexing_col=chunk_val
    )
    
    return trajectory_segmented, trajectory_windows.get_windows()


def evaluate_window(window_data, start_idx, llm_agent, config, rag_db_path=None):
    """Evaluate a single trajectory window using the LLM agent"""
    first_six_steps = window_data.iloc[:6]
    rest_steps = window_data.iloc[6:]
    formatted_first_six = get_formatted_df_for_llm(first_six_steps, precision=2)

    # Generate response based on whether RAG is used
    if rag_db_path is not None:
        # Retrieve similar situations from the database
        db = RAG_Database(rag_db_path) 
        embedding_model = EnvironmentEmbeddingModel()
        top_k_situations = db.get_top_k_situations(
            first_six_steps, 
            embedding_model, 
            k=5, 
            columns=["headway", "speed", "leader_speed"], 
            apply_normalization=False
        )
        
        # Format retrieved situations
        retrieved_situations = ""
        for index, sit in enumerate(top_k_situations):
            retrieved_situations += f"Situation {index + 1}:\n{sit[2]}\n"
        
        # Generate prompt and get response
        provided_user_prompt = rag_user_prompt.format(formatted_first_six, retrieved_situations)
    else:
        # Standard prompt without RAG
        provided_user_prompt = user_prompt.format(formatted_first_six)
    
    # Get response from LLM
    response = llm_agent.get_response(
        system_prompt, 
        provided_user_prompt, 
        temperature=config["temperature"], 
        num_samples=config["num_samples"]
    )

    # Extract values from response
    speed_values, speed_ok = extract_values(response, "future_speeds")
    headway_values, headway_ok = extract_values(response, "future_headway")
    leader_values, leader_ok = extract_values(response, "future_leader_speed")
    reward_values, reward_ok = extract_values(response, "future_rewards")
    coeff_str, coeff_ok = extract_tag_content(response, "reward_coefficients")
    reasoning, reasoning_ok = extract_tag_content(response, "reasoning")

    # Process reward coefficients
    try:
        reward_coefficients = [float(x.strip()) for x in coeff_str.split(',')]
    except:
        reward_coefficients = []
    
    # Check if all values were extracted successfully
    all_ok = speed_ok and headway_ok and leader_ok and reward_ok
    
    # Get ground truth values
    gt = rest_steps
    gt_speeds = gt["speed"].values
    gt_headways = gt["headway"].values
    gt_leader_speed = gt["leader_speed"].values
    gt_rewards = gt["position"].values

    # Create result dictionary
    result = {
        "start_index": start_idx,
        "user_prompt": provided_user_prompt,
        "response": response,
        "reasoning": reasoning if reasoning_ok else "Not extractable",
        "reward_coefficients": reward_coefficients if coeff_ok else [],
        "reward_values": reward_values if reward_ok else [],
        "speed_values": speed_values if speed_ok else [],
        "headway_values": headway_values if headway_ok else [],
        "leader_speed_values": leader_values if leader_ok else [],
        "all_extractable": all_ok,
        "speed_l2": calculate_l2_norm(np.array(speed_values), np.array(gt_speeds)) if speed_ok else float('nan'),
        "headway_l2": calculate_l2_norm(np.array(headway_values), np.array(gt_headways)) if headway_ok else float('nan'),
        "leader_l2": calculate_l2_norm(np.array(leader_values), np.array(gt_leader_speed)) if leader_ok else float('nan'),
        "reward_l2": calculate_l2_norm(np.array(reward_values), np.array(gt_rewards)) if reward_ok else float('nan')
    }
    
    # Calculate overall L2 norm
    result["overall_l2"] = np.nanmean([
        result["speed_l2"], result["headway_l2"], result["leader_l2"]
    ]) if all_ok else float("nan")

    return result


def process_windows(trajectory_windows, chunk_val, llm_agent, config):
    """Process all trajectory windows and collect results"""
    results = []
    
    for i, window in enumerate(tqdm(trajectory_windows, desc="Processing window")):
        if i == 2:  # Early stopping for testing/debugging
            break
        
        start_idx = window.iloc[0][chunk_val]
        result = evaluate_window(
            window, 
            start_idx, 
            llm_agent, 
            config,
            rag_db_path=config["rag_db_path"]
        )
        results.append(result)
    
    return pd.DataFrame(results)


def save_results_and_plot(results_df, results_path, trajectory_df, chunk_val, 
                          trajectory_segmented, plot_path):
    """Save results to JSON and create visualization plot"""
    # Save results
    results_df.to_json(results_path, orient="records", indent=2)
    print(f"Results saved in '{results_path}'")

    # Create and save plot
    plot_dataframe(
        trajectory_df,
        x_axis=chunk_val,
        y_values=["speed", "leader_speed"],
        save_path=plot_path,
        chunker=trajectory_segmented,
        shaded_regions=[(10, 20, "orange", 0.2)],
    )
    print(f"Plot saved in '{plot_path}'")


def main():
    """Main function to run the replay script"""
    # Parse arguments and load configuration
    args = parse_arguments()
    config = load_experiment_config(args.experiment)
    
    # Create output directories
    run_dir, results_path, plot_path, current_datetime = create_output_directories(config)
    
    # Save configuration files
    save_config_files(config, run_dir, args.experiment, current_datetime)
    
    # Load and process trajectory data
    trajectory_df, chunk_val = load_trajectory_data(config)
    trajectory_segmented, trajectory_windows = process_trajectory(trajectory_df, chunk_val, config)
    
    # Get LLM agent from config
    llm_agent = config["llm_model"]
    
    # Process windows and collect results
    results_df = process_windows(trajectory_windows, chunk_val, llm_agent, config)
    
    # Save results and create plot
    save_results_and_plot(results_df, results_path, trajectory_df, chunk_val, 
                          trajectory_segmented, plot_path)


if __name__ == "__main__":
    main()