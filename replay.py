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

# Loading in the utils for the LLM evaluation.
from utils.llm_eval_utils import run_llm_on_window

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run simulation with specified experiment configuration")
    parser.add_argument(
        "--experiment", 
        type=str,
        required=True,
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


def train_on_windows(trajectory_windows, chunk_val, llm_agent, config):
    """Process all trajectory windows and collecting the reuslts. 
    
    This is the main training loop for the LLM. Ideally, we want to be using a random shuffling / batching
    of the windows when training the database of the LLM.
    """
    results = []
    
    # Get the number of windows to process
    # This means that the user wants to run the LLM for a specific NUMBER of iterations.
    if config["percent_of_trajectory"] > 1: 
        num_windows = config["percent_of_trajectory"]
    else:
        num_windows = int(len(trajectory_windows) * config["percent_of_trajectory"])
    
    for i, window in enumerate(tqdm(trajectory_windows, desc="Processing window")):
        if i == num_windows:
            break
        
        start_idx = window.iloc[0][chunk_val]
        result = run_llm_on_window(
            window, 
            start_idx, 
            llm_agent, 
            rag_db_path=None, 
            config=config
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
    results_df = train_on_windows(trajectory_windows, chunk_val, llm_agent, config)
    
    # Save results and create plot
    save_results_and_plot(results_df, results_path, trajectory_df, chunk_val, 
                          trajectory_segmented, plot_path)


if __name__ == "__main__":
    main()