import pandas as pd
import numpy as np
import re
import asyncio
from sklearn.metrics import mean_squared_error
from pathlib import Path
import logging
from tqdm.asyncio import tqdm_asyncio
from utils.db.simulation_db import SimulationDB
from utils.df_utils import (
    plot_dataframe, convert_cols_to_numeric,
    get_df_from_csv, get_formatted_df_for_llm
)
from utils.trajectory.trajectory_container import TrajectoryChunker, TrajectoryWindows
from utils.llm_eval_utils import (
    extract_tag_content, extract_values,
    calculate_l2_norm
)
from models.llm_agent import LLM_Agent, OpenAiModel, GroqModel, OpenRouterModel
import random

# Params
USE_DB = True
DB_PATH = Path("data/simulate/1743744527_04Apr25_05h28m47s/2021-04-22-12-47-13_2T3MWRFVXLW056972_masterArray_0_7050_run_0.db")
CSV_PATH = Path("data/simulate/1743374502_30Mar25_22h41m42s/emissions/emissions_1.csv")
CHUNK_INDICES = [575, 1000, 1500]
WINDOW_SIZE = 10
MAX_WINDOWS = 2 # Set low for debugging
CHUNK_COL_DB = "timestep"
CHUNK_COL_CSV = "step"
PLOT_RESULTS = True
MAX_RETRIES = 2 # number of back and forth actions
SEMAPHORE_LIMIT = 5  # more for groq cause of the rate limiting
SELF_CONSISTENCY_RESPONSES = 4 # self consistency repsonses going in bracket
JUDGE_ATTEMPTS = 1 # number of times judge evalutes response


SYSTEM_PROMPT = """
You are an AI assistant specializing in traffic flow analysis and vehicle dynamics. Your task is to analyze the initial 6 timesteps of vehicle trajectory data and predict the next 4 timesteps for all four variables: reward(float), speed (m/s), headway (m), and leader_speed (m/s).

The data represents real-world traffic measurements from a vehicle equipped with an rl control policy that aims to smooth traffic flow. You must understand the relationship between the following vehicle, its speed adjustments, and the leader vehicle ahead.

---

The RL policy taken by the car is determined by 4 core objectives:
1. **Safety**: A crash indicator computed as `1 if headway < 0 else 0`
2. **Efficiency**: Estimated as `follower_speed × abs(action)`
3. **Comfort**: Measured as `action ** 2` (penalizing abrupt accelerations)
4. **Cohesion**: Measured as `abs(leader_speed - follower_speed) + headway` (how well the AV follows)

The reward is a **linear combination** of these components:

\\[
\text{reward} = w_1 \cdot \text{safety} + w_2 \cdot \text{efficiency} + w_3 \cdot \text{comfort} + w_4 \cdot \text{cohesion}
\\]

---

For each prediction, you must:
1. First analyze the initial data inside <reasoning> tags to explain your thought process
2. Provide precise numerical predictions for each variable
3. Provide the reward coefficents in seperate tags:
    -<reward_coefficients>w1,w2,w3,w4</reward_coefficients>
4. Format your predictions in separate tags:
   - <future_speeds>speed_t6,speed_t7,speed_t8,speed_t9</future_speeds>
   - <future_headway>headway_t6,headway_t7,headway_t8,headway_t9</future_headway>
   - <future_leader_speed>leader_t6,leader_t7,leader_t8,leader_t9</future_leader_speed>
   - <future_reward>reward_t6,reward_t7,reward_t8,reward_t9</future_reward>

---

In your reasoning, analyze:
1. The behavior of the leader vehicle compared to our vehicle
2. The apparent advised speed for traffic smoothing
3. The actual set speed of the vehicle 
4. Whether future acceleration or deceleration is expected

You are effectively explaining what a self-driving car is doing to smooth traffic flow. Your predictions must obey physical constraints and vehicle dynamics (e.g., speeds should evolve gradually, headway depends on relative speed). Avoid hallucinated values.
"""

async def verify_response(response: str, judge_agent) -> dict:
    tags_to_check = {
        "future_speeds": 4,
        "future_headway": 4,
        "future_leader_speed": 4,
        "future_reward": 4,
        "reward_coefficients": 4,
    }

    format_ok = True
    for tag, expected_count in tags_to_check.items():
        values, ok = extract_values(response, tag)
        if not ok or len(values) != expected_count:
            print(f"Format error in <{tag}>")
            format_ok = False

    reasoning, reasoning_ok = extract_tag_content(response, "reasoning")
    if not reasoning_ok or len(reasoning.strip()) < 5:
        print("Missing or too short <reasoning> tag")
        format_ok = False

    if not format_ok:
        return {
            "verdict": "no",
            "reason": "Malformed output format",
            "user_friendly_score": -1,
            "coherence_score": -1,
            "raw_verification_response": "Format check failed"
        }
    verification_system_prompt = f"""
        You are a judge specialized in evaluating AI-generated predictions for traffic flow analysis and vehicle dynamics.

    When evaluating each response, focus carefully on the following criteria:

    1. **Correctness:** Ensure the numerical predictions for reward, speed (m/s), headway (m), and leader_speed (m/s) are physically plausible, consistent, and adhere to realistic vehicle dynamics.
    2. **Structure:** Ensure the response contains all required tags with correct formats:
       - <reasoning>
       - <future_speeds>
       - <future_headway>
       - <future_leader_speed>
       - <future_reward>
       - <reward_coefficients>
    3. **Coherence:** The reasoning should logically follow from the given initial trajectory data, clearly explaining the behavior of the leader vehicle compared to the following vehicle, advised speed for traffic smoothing, actual speed adjustments, and future acceleration or deceleration expectations.
    4. **Clarity and Detail:** Evaluate if the response clearly explains the RL policy's objectives, which include Safety, Efficiency, Comfort, and Cohesion, and how these affect the predicted variables.
   
   ---
    Give scores from 0 to 10 for:
    1. <user_friendly_score>: Is this reasoning accessible, readable, and non-offensive to non-technical users?
    2. <coherence_score>: Does the reasoning logically follow from the initial data step by step?

    Respond ONLY with these two tags:
    <user_friendly_score>X</user_friendly_score>
    <coherence_score>Y</coherence_score>
   """
    verification_prompt = f"""
            You are an expert evaluator of AI-generated traffic reasoning.

            Below is the model's full response:
            ---
            {response}
            ---

            Give scores from 0 to 10 for:
            1. <user_friendly_score>: Is this reasoning accessible, readable, and non-offensive to non-technical users?
            2. <coherence_score>: Does the reasoning logically follow from the initial data step by step?

            Respond ONLY with these two tags:
            <user_friendly_score>X</user_friendly_score>
            <coherence_score>Y</coherence_score>
            """

    judge_response = await judge_agent.get_response_async(
        system_prompt=verification_system_prompt,
        user_prompt=verification_prompt,
        temperature=0.0
    )

    uf_str, ok1 = extract_tag_content(judge_response, "user_friendly_score")
    coh_str, ok2 = extract_tag_content(judge_response, "coherence_score")

    try:
        uf_score = int(uf_str) if ok1 else -1
        coh_score = int(coh_str) if ok2 else -1
    except:
        uf_score, coh_score = -1, -1

    passed = uf_score >= 7 and coh_score >= 7 # threshold set right now

    return {
        "verdict": "yes" if passed else "no",
        "reason": "Scored by LLM judge",
        "user_friendly_score": uf_score,
        "coherence_score": coh_score,
        "raw_verification_response": judge_response
    }



def check_physical_constraints(speed_vals, headway_vals, leader_vals): # Would need to shrink this with some more evals on data
    if any(s < 0 or s > 50 for s in speed_vals):
        return False, "Invalid speed detected"
    if any(h < 0 or h > 200 for h in headway_vals):
        return False, "Invalid headway detected"
    if any(ls < 0 or ls > 50 for ls in leader_vals):
        return False, "Invalid leader speed detected"
    return True, ""

async def get_multiple_responses(llm_agent, system_prompt, user_prompt, num_responses=4):
    tasks = [llm_agent.get_response_async(system_prompt, user_prompt, temperature=0.7) for _ in range(num_responses)]
    responses = await asyncio.gather(*tasks)
    return responses

async def judge_battle(response_a, response_b, judge_agent, judge_attempts=3):
    judge_system_prompt = """
        You are a judge specialized in evaluating AI-generated predictions for traffic flow analysis and vehicle dynamics.

        When evaluating each response, focus carefully on the following criteria:

        1. **Correctness:** Ensure the numerical predictions for reward, speed (m/s), headway (m), and leader_speed (m/s) are physically plausible, consistent, and adhere to realistic vehicle dynamics.
        2. **Structure:** Ensure the response contains all required tags with correct formats:
        - <reasoning>
        - <future_speeds>
        - <future_headway>
        - <future_leader_speed>
        - <future_reward>
        - <reward_coefficients>
        3. **Coherence:** The reasoning should logically follow from the given initial trajectory data, clearly explaining the behavior of the leader vehicle compared to the following vehicle, advised speed for traffic smoothing, actual speed adjustments, and future acceleration or deceleration expectations.
        4. **Clarity and Detail:** Evaluate if the response clearly explains the RL policy's objectives, which include Safety, Efficiency, Comfort, and Cohesion, and how these affect the predicted variables.

        Choose the overall better response strictly based on these guidelines.

        Respond strictly in the following format:
        <winner>A</winner> or <winner>B</winner>
        """

    prompt = f"""
    Response A:
    {response_a}

    Response B:
    {response_b}

    Which response better satisfies the correctness, structure, coherence, and clarity requirements described above?
    """
    results = []
    for i in range(judge_attempts):
        try:
            logging.info(f"[Judge Battle Attempt {i+1}] Sending request to judge...")
            judge_response = await judge_agent.get_response_async(judge_system_prompt, prompt, temperature=0.0)
            logging.info(f"[Judge Battle Attempt {i+1}] Received response:\n{judge_response}")

            winner_tag, winner_ok = extract_tag_content(judge_response, "winner")
            if winner_ok and winner_tag in ['A', 'B']:
                results.append(winner_tag)
        except Exception as e:
            logging.warning(f"[Judge Battle Attempt {i+1}] Exception occurred: {e}")

    if results:
        most_common = max(set(results), key=results.count)
        if results.count('A') == results.count('B'):
            logging.info("Judge battle resulted in a tie. Selecting randomly.")
            return random.choice([response_a, response_b])
        logging.info(f"Final winner after {judge_attempts} attempts: {most_common}")
        return response_a if most_common == 'A' else response_b

    logging.warning("No valid winner tags returned. Falling back to random choice.")
    return random.choice([response_a, response_b])

async def run_bracket(responses, judge_agent):
    round_responses = responses[:]

    while len(round_responses) > 1:
        next_round = []
        pairs = [(round_responses[i], round_responses[i+1]) for i in range(0, len(round_responses), 2)]

        for idx, (resp_a, resp_b) in enumerate(pairs):
            winner = await judge_battle(resp_a, resp_b, judge_agent, JUDGE_ATTEMPTS)
            print(f"Pair {idx+1} winner selected.")
            next_round.append(winner)

        round_responses = next_round

    return round_responses[0]

async def evaluate_window_self_consistency(window_data, start_idx, llm_agent, judge_agent=None, semaphore=None):
    async with semaphore:
        first_six = window_data.iloc[:6]
        future_steps = window_data.iloc[6:]
        prompt_data = get_formatted_df_for_llm(first_six, precision=2)

        user_prompt = f"""
        Below are the initial 6 timesteps of trajectory data from a vehicle operating under an RL-based traffic smoothing policy:

        {prompt_data}

        Based on this history, please:

        1. Predict the next 4 timesteps (t6–t9) for all four variables:
        - reward (float)
        - speed (m/s)
        - headway (m)
        - leader_speed (m/s)

        2. Use the standard output tag format for each variable:
        - <future_speeds>...</future_speeds>
        - <future_headway>...</future_headway>
        - <future_leader_speed>...</future_leader_speed>
        - <future_reward>...</future_reward>

        3. Provide your estimated reward coefficients in:
        - <reward_coefficients>w1,w2,w3,w4</reward_coefficients>

        4. Include your reasoning in <reasoning> tags, reflecting on dynamics, expected changes, and whether the vehicle will accelerate or decelerate.

        Be sure to keep your predictions physically plausible and consistent with vehicle dynamics.
        """

        for attempt in range(MAX_RETRIES):
            responses = await get_multiple_responses(llm_agent, SYSTEM_PROMPT, user_prompt, SELF_CONSISTENCY_RESPONSES)
            best_response = await run_bracket(responses, judge_agent)

            verdict = await verify_response(best_response, judge_agent)
            if verdict["verdict"] != "yes":
                logging.warning(f"[Attempt {attempt+1}] Judge rejected best response: {verdict}")
                continue

            speed_vals, speed_ok = extract_values(best_response, "future_speeds")
            headway_vals, headway_ok = extract_values(best_response, "future_headway")
            leader_vals, leader_ok = extract_values(best_response, "future_leader_speed")
            reward_vals, reward_ok = extract_values(best_response, "future_reward")
            coeff_str, coeff_ok = extract_tag_content(best_response, "reward_coefficients")
            reasoning, reasoning_ok = extract_tag_content(best_response, "reasoning")

            valid, message = check_physical_constraints(speed_vals, headway_vals, leader_vals)
            if not valid:
                logging.error(f"Attempt {attempt+1}: {message}")
                continue

            try:
                reward_coeffs = [float(x.strip()) for x in coeff_str.split(',')]
            except:
                reward_coeffs = []

            gt = future_steps
            result = {
                "start_index": start_idx,
                "reasoning": reasoning if reasoning_ok else "Not extractable",
                "reward_coefficients": reward_coeffs if coeff_ok else [],
                "reward_values": reward_vals if reward_ok else [],
                "speed_values": speed_vals if speed_ok else [],
                "headway_values": headway_vals if headway_ok else [],
                "leader_speed_values": leader_vals if leader_ok else [],
                "all_extractable": speed_ok and headway_ok and leader_ok and reward_ok,
                "speed_l2": calculate_l2_norm(speed_vals, gt["speed"].values) if speed_ok else float('nan'),
                "headway_l2": calculate_l2_norm(headway_vals, gt["headway"].values) if headway_ok else float('nan'),
                "leader_l2": calculate_l2_norm(leader_vals, gt["leader_speed"].values) if leader_ok else float('nan'),
                "reward_l2": calculate_l2_norm(reward_vals, gt["reward"].values) if reward_ok else float('nan'),
                "overall_l2": np.nan
            }


            result.update({
                "user_friendly_score": verdict["user_friendly_score"],
                "coherence_score": verdict["coherence_score"],
                "judge_response": verdict["raw_verification_response"]
            })
            result["overall_l2"] = np.nanmean([
                result["speed_l2"], result["headway_l2"], result["leader_l2"]
            ])
            return result

        return {"start_index": start_idx, "reasoning": "No valid response", "all_extractable": False}

if USE_DB:
    with SimulationDB(DB_PATH) as db:
        df = db.get_vehicle_data("1_rl_av", ["speed", "leader_speed", "headway"])
        df[CHUNK_COL_DB] = pd.to_numeric(df[CHUNK_COL_DB])
        df.rename(columns={col: col.replace("1_rl_av__", "") for col in df.columns if "1_rl_av__" in col}, inplace=True)
        chunk_col = CHUNK_COL_DB
else:
    df = get_df_from_csv(str(CSV_PATH))
    df = df[df["id"] == "1_rl_av"][["step", "speed", "headway", "leader_speed", "position"]]
    chunk_col = CHUNK_COL_CSV
    logging.warning("Using CSV fallback — reward uses 'position' as a placeholder.")
chunker = TrajectoryChunker(df, chunk_col=chunk_col, chunk_indices=CHUNK_INDICES, sort_col=chunk_col)
chunks = chunker.get_chunks()
windows = TrajectoryWindows(chunks[0], window_size=WINDOW_SIZE, indexing_col=chunk_col).get_windows()




llm_agent = OpenRouterModel()
judge_agent = OpenRouterModel()
async def main():
    semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)
    tasks = [
        evaluate_window_self_consistency(window, window.iloc[0][chunk_col], llm_agent, judge_agent, semaphore)
        for window in windows[:MAX_WINDOWS]
    ]
    results = []

    for coro in tqdm_asyncio.as_completed(tasks, desc="Evaluating trajectory windows"):
        result = await coro
        results.append(result)
        logging.info(f"Finished evaluation for start index: {result.get('start_index', 'Unknown')}")

    pd.DataFrame(results).to_json("trajectory_prediction_results.json", orient="records", indent=2)

    if PLOT_RESULTS:
        plot_dataframe(
            df,
            x_axis=chunk_col,
            y_values=["speed", "leader_speed"],
            save_path="rl_av_data_from_emissions_chunked.png",
            chunker=chunker,
            shaded_regions=[(10, 20, "orange", 0.2)],
        )

if __name__ == "__main__":
    asyncio.run(main())
