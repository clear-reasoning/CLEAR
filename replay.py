from utils.db.simulation_db import SimulationDB
from utils.df_utils import plot_dataframe, convert_cols_to_numeric, get_df_from_csv, get_formatted_df_for_llm
from utils.trajectory.trajectory_container import TrajectoryChunker, TrajectoryWindows
from models.llm_agent import LLM_Agent, OpenAiModel

# Path to your database
db_path = "data/simulate/1743365635_30Mar25_20h13m55s/2021-04-22-12-47-13_2T3MWRFVXLW056972_masterArray_0_7050_run_0.db"
# Example usage of using the database context manager
with SimulationDB(db_path) as db:
    # rl_av_data = db.get_vehicle_data("1_rl_av", ["speed", "leader_speed", "position", "headway"])
    pass

# Getting information about the RL AV. 
trajectory_df = get_df_from_csv("data/simulate/1743365635_30Mar25_20h13m55s/emissions/emissions_1.csv")
# only keeping information about the autonomus vehicle (AV``)
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

print(get_formatted_df_for_llm(trajectory_windows[0], precision=2))

# call to the LLM with the trajectory data stringified
# llm_agent = OpenAiModel()
# system_prompt = "You are a data analyst. You are given a dataframe with trajectory data. Your task is to analyze the data and provide insights."
# user_prompt = f"Here is the data:\n{get_formatted_df_for_llm(trajectory_windows[0], precision=2)}\nPlease provide insights about the data."    
# response = llm_agent.get_response(system_prompt, user_prompt)
    

# rl_av_data = convert_cols_to_numeric(rl_av_data, ["1_rl_av__speed", "1_rl_av__leader_speed", "1_rl_av__headway"])        
plot_dataframe(
    trajectory_df,
    x_axis="step",
    y_values=["speed", "leader_speed"],
    save_path="rl_av_data_from_emissions_chunked.png",
    chunker=trajectory_segmented,
    shaded_regions=[(10, 20, "orange", 0.2)],
)

print("hello world")