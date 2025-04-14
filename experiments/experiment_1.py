from models.llm_agent import GeminiModel
from prompts.ashwin_04_11 import user_prompt, system_prompt, corrective_system_prompt, corrective_user_prompt

config = {
    # Experiment metadata so we can easily track the run. 
    
    "experiment_metadata": "Finished implementation of corrective loop in the RAG framework.",
    
    # Database settings
    "use_db": True,  # Whether to use database or CSV
    "db_path": "data/simulate/1743365635_30Mar25_20h13m55s/2021-04-22-12-47-13_2T3MWRFVXLW056972_masterArray_0_7050_run_0.db",
    "csv_path": "data/simulate/1743374502_30Mar25_22h41m42s/emissions/emissions_1.csv",
    "trajectory_processor": lambda x: x[100:],
    
    # Vehicle and data settings
    "vehicle_id": "1_rl_av",
    "db_columns": ["speed", "leader_speed", "position", "headway", "accel", "time", "reward", "realized_accel"],
    "csv_columns": ["time""step", "speed", "headway", "leader_speed", "position", "accel", "realized_accel"],
    
    # Chunking and windowing settings
    "chunk_col_db": "timestep",
    "chunk_col_csv": "step",
    "chunk_indices": [575, 1000, 1500],
    "window_size": 10,
    
    # Runs the LLM loop on this much percentage of the trajectory. 
    # For example, if the trajectory is 1000 timesteps, and this is set to 0.1, the LLM will run for 100 timesteps.
    "percent_of_trajectory": 50,
    
    # Random seed for reproducibility. Shuffling the training windows.
    "random_seed": 42,
    "shuffle_windows": True,
    
    # RAG settings
    "rag_db_path": None,
    "checkpoint_rag_db_path": "rag_documents/pkl_db/completed_loop_db.pkl",
    
    # LLM Model Settings,
    "llm_model": GeminiModel(),
    "temperature": 1,
    "num_samples": 1,
    
    # Prompt settings
    "user_prompt": user_prompt,
    "system_prompt": system_prompt,
    
    # Corrective prompt settings. 
    "corrective_system_prompt": corrective_system_prompt,
    "corrective_user_prompt": corrective_user_prompt,
    
    # Output settings
    "output_dir": "outputs",  # Directory to store all output files
    "results_filename": "traj_results.json",  # Standard name for results JSON
    "plot_filename": "trajectory_plot.png"  # Standard name for plot
}