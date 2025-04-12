import re
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from typing import Dict, Any, Optional

from models.llm_agent import LLM_Agent

# Importing the utils for the RAG.
from utils.rag.embedding_models import EnvironmentEmbeddingModel
from utils.rag.read_embedding_db import RAG_Database

# Importing the prompts.
from prompts.ashwin_04_05 import user_prompt
from prompts.adrien_04_07 import rag_user_prompt, system_prompt


def extract_tag_content(response: str, tag_name: str):
    """Extract the content within a specific XML-style tag from a response."""
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, response, re.DOTALL)
    return (match.group(1).strip(), True) if match else ([], False)

def extract_values(response: str, tag_name: str):
    """Extract a list of float values from a specific tag."""
    raw, found = extract_tag_content(response, tag_name)
    if not found:
        return [], False
    try:
        values = [float(v.strip()) for v in raw.split(',') if v.strip()]
        return values, True
    except ValueError:
        return [], False

def calculate_l2_norm(predictions, ground_truth):
    """Compute the L2 norm (RMSE) between predictions and ground truth."""
    if len(predictions) != len(ground_truth):
        return float('nan')
    return np.sqrt(mean_squared_error(ground_truth, predictions))

def run_llm_on_window(trajectory_window: pd.DataFrame, start_index: int, llm_agent: LLM_Agent, rag_db_path: Optional[str], config: dict) -> Dict[str, Any]:
    """
    This is a helper function that runs the LLM on a single trajectory window.
    
    This function should be able to handle both RAG and non-RAG cases. This should run the workflow and return back some dictionary of statistics/losses.
    """
    breakpoint()
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