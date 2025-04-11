import random as rd
import json
import pandas as pd

trajectory_df = pd.read_csv("../data/simulate/1743374502_30Mar25_22h41m42s/emissions/emissions_1.csv", na_values=["NULL", "NaN", "N/A", ""])
trajectory_df = trajectory_df[trajectory_df["id"] == "1_rl_av"]
trajectory_df = trajectory_df[["step", "speed", "headway", "leader_speed", "position"]] # TODO: position is the placeholder for reward for the current moment

nbr_timesteps = len(trajectory_df)
indices = []
for i in range(100):
    idx = rd.randint(5, nbr_timesteps-4)
    while idx in indices:
        idx = rd.randint(0, nbr_timesteps-4)
    indices.append(idx)

samples = []
for idx in indices:
    sample_1 = trajectory_df.iloc[idx-5:idx+1].reset_index(drop=True)
    sample_2 = trajectory_df.iloc[idx+1:idx+5].reset_index(drop=True)
    dict_sample_1 = sample_1.to_dict(orient="list")
    dict_sample_2 = sample_2.to_dict(orient="list")
    sample_entry = {
        "last_timesteps": dict_sample_1,
        "explanation": "",
        "next_timesteps": dict_sample_2
    }
    samples.append(sample_entry)

    json_path = "6_timesteps_4_predictions_no_explanation.json"
    with open(json_path, "w") as json_file:
        json.dump(samples, json_file, indent=4)