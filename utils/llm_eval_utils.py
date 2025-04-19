import re
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from typing import Dict, Any, Optional

from models.llm_agent import LLM_Agent

# Importing the utils for the RAG.
from utils.rag.embedding_models import BaseEmbeddingModel
from utils.rag.read_embedding_db import RAG_Database
from utils.rag.similarity_functions import l2_loss


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
        values = [float(v.strip()) for v in raw.split(",") if v.strip()]
        return values, True
    except ValueError:
        return [], False


def calculate_l2_norm(predictions, ground_truth):
    """Compute the L2 norm (RMSE) between predictions and ground truth."""
    if len(predictions) != len(ground_truth):
        return float("nan")
    return np.sqrt(mean_squared_error(ground_truth, predictions))


def run_llm_on_window(
    trajectory_window: pd.DataFrame,
    hypothetical_situation: str,
    start_idx: int,
    llm_agent: LLM_Agent,
    db: Optional[RAG_Database],
    embedding_model: Optional[BaseEmbeddingModel],
    config: dict,
) -> Dict[str, Any]:
    """
    This is a helper function that runs the LLM on a single trajectory window.

    This function should be able to handle both RAG and non-RAG cases. This should run the workflow and return back some dictionary of statistics/losses.
    """
    # Getting the first observation/row within the trajectory window.
    current_situation = trajectory_window.iloc[0]
    # Only keeping the columns that the RL controller is trained on.
    current_situation = current_situation[
        ["speed", "headway", "leader_speed"]
    ].to_dict()

    # If RAG is enabled, retrieve and format similar situations
    if db is not None and embedding_model is not None:

        if len(db.json_contents) == 0:
            # no document retrieved
            text_top_k_documents = "No similar situations found in the database."
            document_indices = []

        if len(db.json_contents) > 0:
            ## Retrieve documents from the database depending on the length of the database.
            intermediate_k = min(30, len(db.json_contents))
            k = min(5, len(db.json_contents))
            top_k_documents = db.get_top_k_docs(
                hypothetical_situation,
                current_situation,
                embedding_model,
                intermediate_k,
                k,
            )

            # Format the retrieved documents
            document_indices = [doc["original_index"] for doc in top_k_documents]
            text_top_k_documents = ""
            for index, document in enumerate(top_k_documents):
                text_top_k_documents += (
                    "Situation n:"
                    + str(index)
                    + " \n"
                    + ", ".join(f"{key}: {value}" for key, value in document.items())
                    + "\n\n"
                )

        # Update the user prompt with the current situation, the hypothetical situation and the retrieved documents
        text_current_situation = ", ".join(
            f"{key}: {value}" for key, value in current_situation.items()
        )
        formated_user_prompt = config["user_prompt"].format(
            text_current_situation,
            hypothetical_situation,
            text_top_k_documents,
        )

    # Getting the response from the LLM
    response = llm_agent.get_response(
        config["system_prompt"],
        formated_user_prompt,
        temperature=config["temperature"],
        num_samples=config["num_samples"],
    )

    task_1_content, _ = extract_tag_content(response, "task1")
    task_2_content, _ = extract_tag_content(response, "task2")

    # Extracting the action for task1 from the LLM response
    llm_task1_action, _ = extract_values(task_1_content, "action")

    return {
        # user_prompt is the prompt that was used to generate the response.
        "user_prompt": formated_user_prompt,
        # llm_response is the response from the LLM.
        "llm_response": response,
        "action_explanation": task_1_content,
        "situation_analysis": task_2_content,
        # llm_task1_action is the action that the LLM would have taken for the task1.
        "llm_task1_action": llm_task1_action,
        # true_accel is the acceleration that the RL controller would have taken. realized_accel is fairly similar to true_accel.
        "true_accel": trajectory_window.iloc[0]["accel"],
        "true_realized_accel": trajectory_window.iloc[0]["realized_accel"],
        # Storing the indices of the retrieved documents so we can later replace if needed.
        "retrieved_document_indices": document_indices,
    }


def run_llm_correction(
    trajectory_llm_results: Dict[str, Any],
    llm_agent: LLM_Agent,
    config: dict,
    db: Optional[RAG_Database],
    embedding_model: Optional[BaseEmbeddingModel],
) -> Dict[str, Any]:
    """
    This function looks at the previous explanation/response created by the LLM and tries to come up
    with a new explanation/response that is more accurate.
    """
    # Getting the explanation/response from the generation LLM.
    generated_explanation = trajectory_llm_results["llm_response"]
    generated_explanation_for_task_1 = extract_tag_content(
        generated_explanation, "task1"
    )
    generated_explanation_for_task_2 = extract_tag_content(
        generated_explanation, "task2"
    )

    # Getting only the reasoning related to task 1 from the previous explanation.

    # Pointing out the fallacies in the reasoning chain through the deduction/observation method
    # TODO

    # Formatting the corrective prompt for task 1 related reasoning.
    task_1_ground_truth_action = trajectory_llm_results["true_accel"]
    corrective_user_prompt = config["corrective_user_prompt"].format(
        generated_explanation_for_task_1, task_1_ground_truth_action
    )

    # Getting the response from the corrective LLM.
    corrective_response_task_1 = llm_agent.get_response(
        config["corrective_system_prompt"],
        corrective_user_prompt,
        temperature=config["temperature"],
        num_samples=config["num_samples"],
    )

    # TODO. Here there would be some ranking / eviction policy to make sure that the database doesn't grow too much.

    # Temporarily, we append with 50% probability while we replace a random pulled document with the corrective response.\
    retrieved_document_indices = trajectory_llm_results["retrieved_document_indices"]
    if np.random.rand() < 0.5 or len(retrieved_document_indices) == 0:
        # Appending the corrective response to the database.
        db.add_document(
            {
                "environment_key": trajectory_llm_results["first_observation_as_str"],
                "environment_explanation": corrective_response_task_1,
            },
            embedding_model,
        )
    else:
        # Randomly selecting one of the retrieved documents and replacing it with the corrective response.
        random_index = np.random.choice(retrieved_document_indices)
        db.replace_document(
            random_index,
            {
                "environment_key": trajectory_llm_results["first_observation_as_str"],
                "environment_explanation": corrective_response_task_1,
            },
            embedding_model,
        )

    return {
        "corrected_explanation_for_task_1": corrective_response_task_1,
    }


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
            apply_normalization=False,
        )

        # Format retrieved situations
        retrieved_situations = ""
        for index, sit in enumerate(top_k_situations):
            retrieved_situations += f"Situation {index + 1}:\n{sit[2]}\n"

        # Generate prompt and get response
        provided_user_prompt = rag_user_prompt.format(
            formatted_first_six, retrieved_situations
        )
    else:
        # Standard prompt without RAG
        provided_user_prompt = user_prompt.format(formatted_first_six)

    # Get response from LLM
    response = llm_agent.get_response(
        system_prompt,
        provided_user_prompt,
        temperature=config["temperature"],
        num_samples=config["num_samples"],
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
        reward_coefficients = [float(x.strip()) for x in coeff_str.split(",")]
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
        "speed_l2": calculate_l2_norm(np.array(speed_values), np.array(gt_speeds))
        if speed_ok
        else float("nan"),
        "headway_l2": calculate_l2_norm(np.array(headway_values), np.array(gt_headways))
        if headway_ok
        else float("nan"),
        "leader_l2": calculate_l2_norm(
            np.array(leader_values), np.array(gt_leader_speed)
        )
        if leader_ok
        else float("nan"),
        "reward_l2": calculate_l2_norm(np.array(reward_values), np.array(gt_rewards))
        if reward_ok
        else float("nan"),
    }

    # Calculate overall L2 norm
    result["overall_l2"] = (
        np.nanmean([result["speed_l2"], result["headway_l2"], result["leader_l2"]])
        if all_ok
        else float("nan")
    )

    return result
