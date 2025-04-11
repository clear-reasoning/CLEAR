from models.llm_agent import OpenAiModel

config = {
    # Experiment metadata so we can easily track the run. 
    
    "experiment_metadata": "Testing out the end to end integration of the RAG and LLM framework with the new experiment set up framework.",
    
    # Database settings
    "use_db": True,  # Whether to use database or CSV
    "db_path": "data/simulate/1743365635_30Mar25_20h13m55s/2021-04-22-12-47-13_2T3MWRFVXLW056972_masterArray_0_7050_run_0.db",
    "csv_path": "data/simulate/1743374502_30Mar25_22h41m42s/emissions/emissions_1.csv",
    
    # Vehicle and data settings
    "vehicle_id": "1_rl_av",
    "db_columns": ["speed", "leader_speed", "position", "headway"],
    "csv_columns": ["step", "speed", "headway", "leader_speed", "position"],
    
    # Chunking and windowing settings
    "chunk_col_db": "timestep",
    "chunk_col_csv": "step",
    "chunk_indices": [575, 1000, 1500],
    "window_size": 10,
    
    # RAG settings
    "rag_db_path": "rag_documents/pkl_db/semantic_db_examples.pkl",
    
    # LLM Model Settings,
    "llm_model": OpenAiModel(),
    "temperature": 1,
    "num_samples": 1,
    
    # Output settings
    "output_dir": "outputs",  # Directory to store all output files
    "results_filename": "traj_results.json",  # Standard name for results JSON
    "plot_filename": "trajectory_plot.png"  # Standard name for plot
}